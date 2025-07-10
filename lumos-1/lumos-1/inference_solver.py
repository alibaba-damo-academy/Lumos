import argparse
import copy
import math
from typing import List, Optional, Union

from PIL import Image
import torch
import transformers
from transformers import GenerationConfig, TextStreamer
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList, LogitsWarper

from data.item_processor import FlexARItemProcessor
from model.chameleon import ChameleonForConditionalGeneration


class LLMImageStartTriggeredUnbatchedClassifierFreeGuidanceLogitsProcessor(LogitsProcessor):
    r"""
    Logits processor for Classifier-Free Guidance (CFG). The processors computes a weighted average across scores
    from prompt conditional and prompt unconditional (or negative) logits, parameterized by the `guidance_scale`.
    The unconditional scores are computed internally by prompting `model` with the `unconditional_ids` branch.

    See [the paper](https://arxiv.org/abs/2306.17806) for more information.
    """

    def __init__(
        self,
        guidance_scale: float,
        model,
        image_start_token_id,
        image_end_token_id,
        image_next_line_token_id,
        patch_size,
        unconditional_ids: Optional[torch.LongTensor] = None,
        unconditional_attention_mask: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = True,
    ):
        '''
        Parameters:
            - guidance_scale: This is a float that controls the mix between conditional and unconditional logits. 
                            A value of 1.0 means no guidance, while values greater than 1.0 will emphasize the prompt-conditioning.
            - model: This is the underlying model used to compute logits.
            - image_start_token_id, image_end_token_id, image_next_line_token_id: These are token identifiers for specific image-related tokens.
                                                                                They help the processor identify image boundaries.
            - patch_size: Size of the image patches. Used in processing images.
            - unconditional_ids and unconditional_attention_mask: These are used to compute unconditional logits, i.e., logits generated without specific prompt conditioning.
            - use_cache: A boolean that, when set to True, uses cached attention mechanisms for faster generation.
            - unconditional_context_backup: This dictionary stores the initial context for generating unconditional logits.
            - Attributes like h_latent_dim, w_latent_dim, etc.: Used for storing latent dimensions when processing the image tokens.
        '''
        self.guidance_scale = guidance_scale
        self.model = model
        self.unconditional_context_backup = {
            "input_ids": unconditional_ids,
            "attention_mask": unconditional_attention_mask,
            "use_cache": use_cache,
            "past_key_values": transformers.DynamicCache() if use_cache else None,
            "first_pass": True,
        }
        self.unconditional_context = None

        self.nums_image_start_tokens = None

        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.image_next_line_token_id = image_next_line_token_id
        self.image_start_token_id_index = None
        self.patch_size = patch_size
        self.h_latent_dim = None
        self.w_latent_dim = None

    def get_unconditional_logits(self, input_ids, image_start_token_id_index):
        '''
            Generate unconditional logits using the model, without the influence of the specific prompt.
        '''

        # If this is the first pass, initialize the input_ids and attention mask for the unconditional context
        if self.unconditional_context["first_pass"]:
            if self.unconditional_context["input_ids"] is None:
                # Set the input IDs to start from the last image start token index
                self.unconditional_context["input_ids"] = input_ids[:, image_start_token_id_index:]
            if self.unconditional_context["attention_mask"] is None:
                # Create an attention mask of ones with the same shape as the input_ids
                self.unconditional_context["attention_mask"] = torch.ones_like(
                    self.unconditional_context["input_ids"], dtype=torch.long
                )
            
            # Update input_ids and attention_mask with the initialized values
            input_ids = self.unconditional_context["input_ids"]
            attention_mask = self.unconditional_context["attention_mask"]

            # Set first_pass to False after initializing
            self.unconditional_context["first_pass"] = False
        else:
            # For subsequent passes, update the attention mask to reflect the new context
            attention_mask = torch.cat(
                [
                    self.unconditional_context["attention_mask"],
                    torch.ones_like(input_ids[:, -1:], dtype=torch.long),
                ],
                dim=1,
            )

            # Update the input_ids based on whether caching is used
            if not self.unconditional_context["use_cache"]:
                input_ids = torch.cat([self.unconditional_context["input_ids"], input_ids[:, -1:]], dim=1)
            else:
                input_ids = input_ids[:, -1:] # Use only the latest input ID
            
            # Update the unconditional context with the new input_ids and attention mask
            self.unconditional_context["input_ids"] = input_ids
            self.unconditional_context["attention_mask"] = attention_mask

        out = self.model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=self.unconditional_context["use_cache"],
            past_key_values=self.unconditional_context["past_key_values"],
        )
        self.unconditional_context["past_key_values"] = out.get("past_key_values", None)

        return out.logits

    def __call__(self, input_ids, scores):
        '''
            The main function that processes logits during generation to apply Classifier-Free Guidance
        '''

        # Count the number of image start and end tokens in the input sequence.
        num_image_start_tokens = (input_ids[0] == self.image_start_token_id).sum()
        num_image_end_tokens = (input_ids[0] == self.image_end_token_id).sum()

        # If the number of start tokens and end tokens are equal, it means we're not in the middle of generating an image.
        if num_image_start_tokens == num_image_end_tokens:
            # Reset latent dimensions, token index, and unconditional context
            self.h_latent_dim, self.w_latent_dim = None, None
            self.image_start_token_id_index = None
            self.unconditional_context = None
            return scores  # Return the original scores without modification

        # If there is one more start token than end tokens, we're currently generating an image
        elif num_image_start_tokens == num_image_end_tokens + 1:
            # Identify the index of the last image start token if not already set
            if self.image_start_token_id_index is None:
                self.image_start_token_id_index = torch.where(input_ids[0] == self.image_start_token_id)[0][-1].item()
            
            # Determine the number of tokens that have been generated since the image start token
            new_token_num = len(input_ids[0][self.image_start_token_id_index + 1 :])
            if new_token_num >= 2:
                # Calculate the latent dimensions if they haven't been set yet
                if self.h_latent_dim is None or self.w_latent_dim is None:
                    h_grids, w_grids = (
                        input_ids[0][self.image_start_token_id_index + 1] - 8804,
                        input_ids[0][self.image_start_token_id_index + 2] - 8804,
                    )
                    self.h_latent_dim, self.w_latent_dim = h_grids * 2, w_grids * 2

                # Set the unconditional context for the first time if it hasn't been set yet
                if self.unconditional_context is None:
                    self.unconditional_context = copy.deepcopy(self.unconditional_context_backup)

                # If the guidance scale is 1.0, there is no need to modify the scores
                if self.guidance_scale == 1.0:
                    return scores

                # Compute unconditional logits using the model
                unconditional_logits = self.get_unconditional_logits(input_ids, self.image_start_token_id_index)[:, -1]

                # Apply Classifier-Free Guidance to adjust the scores
                scores_processed = self.guidance_scale * (scores - unconditional_logits) + unconditional_logits
                return scores_processed

        else:
            print("Something wrong in the decoding process.")

        return scores


class MultiModalLogitsProcessor(LogitsProcessor):
    '''
        Initialize the logits processor with necessary image-specific tokens and settings
    '''

    def __init__(
        self,
        image_start_token_id=None,
        image_end_token_id=None,
        image_next_line_token_id=None,
        patch_size=None,
        voc_size=None,
    ):
        # Token IDs that identify the start, end, and next line of an image section
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.image_next_line_token_id = image_next_line_token_id
        self.image_start_token_id_index = None
        self.patch_size = patch_size
        self.h_latent_dim = None
        self.w_latent_dim = None

        # Create lists and tensors used for token suppression
        # `vocab_list` is a list of all token IDs in the vocabulary
        self.vocab_list = [i for i in range(voc_size)]
        # `image_token_list` represents token IDs used for image representation
        self.image_token_list = [i for i in range(4, 8195 + 1)]
        # `suppress_tokens` is a list of tokens not relevant for image data
        self.suppress_tokens = torch.tensor(
            [x for x in self.vocab_list if x not in self.image_token_list], device="cuda"
        )

        # Create tensors used to mask scores of tokens not needed in the current generation context
        self.vocab_tensor = torch.arange(voc_size, device="cuda")
        # Mask used to suppress non-image tokens in image generation
        self.suppress_token_mask = torch.isin(self.vocab_tensor, self.suppress_tokens)
        # Mask used to force the new line token
        self.new_line_force_token_mask = torch.isin(
            self.vocab_tensor, torch.tensor([self.image_next_line_token_id], device="cuda")
        )
        # Mask used to force the end-of-image token
        self.eos_image_force_token_mask = torch.isin(
            self.vocab_tensor, torch.tensor([self.image_end_token_id], device="cuda")
        )

        # Additional flags and counters to track the generation state
        self.flag = False
        self.num_image_start_tokens = None
        self.num_image_end_tokens = None

    # @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        # Count the number of image start and end tokens in the current input sequence
        self.num_image_start_tokens = (input_ids[0] == self.image_start_token_id).sum()
        self.num_image_end_tokens = (input_ids[0] == self.image_end_token_id).sum()

        # print(self.num_image_start_tokens, self.num_image_end_tokens)

        # If the number of start and end tokens is equal, no image section is being processed
        if self.num_image_start_tokens == self.num_image_end_tokens:
            # Reset latent dimensions and start token index
            self.h_latent_dim, self.w_latent_dim = None, None
            self.image_start_token_id_index = None
            return scores

        # If we have one more start token than end tokens, it means an image section is being generated
        elif self.num_image_start_tokens == self.num_image_end_tokens + 1:
            # Find the index of the last image start token if not already determine
            if self.image_start_token_id_index is None:
                self.image_start_token_id_index = torch.where(input_ids[0] == self.image_start_token_id)[0]
                print(self.image_start_token_id_index)
                self.image_start_token_id_index = torch.where(input_ids[0] == self.image_start_token_id)[0][-1].item()

            # Calculate the number of new tokens generated after the image start token
            new_token_num = len(input_ids[0][self.image_start_token_id_index + 1 :])
            # print(f"num new tokens: {new_token_num}")

            # If at least two tokens have been generated after the image start token
            if new_token_num >= 2:
                # Determine the height and width of the latent image representation
                if self.h_latent_dim is None or self.w_latent_dim is None:
                    h_grids, w_grids = (
                        input_ids[0][self.image_start_token_id_index + 1] - 8804,
                        input_ids[0][self.image_start_token_id_index + 2] - 8804,
                    )
                    # print(f"h_grids: {h_grids}, w_grids: {w_grids}")
                    self.h_latent_dim, self.w_latent_dim = h_grids * 2, w_grids * 2
                    print(f"h_latent_dim: {self.h_latent_dim}, w_latent_dim: {self.w_latent_dim}")

                # Process the tokens generated for the image
                tokens = input_ids[0][self.image_start_token_id_index + 3 :]

                # If the number of tokens matches the width for a new line
                if (len(tokens) + 1) % (self.w_latent_dim + 1) == 0:
                    # Force the model to output the next line token
                    new_line_constrained_scores = torch.full_like(scores, -math.inf)
                    new_line_constrained_scores[:, self.image_next_line_token_id] = 0
                    print(f"new line: {len(tokens)+1}")
                    return new_line_constrained_scores
                
                # If the number of tokens matches the entire image height and width （Image generation）
                elif (len(tokens) + 1) == (self.w_latent_dim + 1) * self.h_latent_dim + 1:
                    # Force the model to output the end-of-image token
                    eos_image_constrained_scores = torch.full_like(scores, -math.inf)
                    eos_image_constrained_scores[:, self.image_end_token_id] = 0
                    print(f"eos image: {len(tokens)+1}")
                    return eos_image_constrained_scores
                
                # If we're still generating tokens within the current row
                elif (len(tokens) + 1) % (self.w_latent_dim + 1) != 0:
                    # Constrain the scores to suppress non-image tokens
                    image_constrained_scores = torch.where(self.suppress_token_mask, -float("inf"), scores)
                    # This line uses torch.where() to replace scores of all tokens that are not image-related with negative infinity (-float("inf")).
                    # Setting a score to negative infinity effectively means that the model will not select that token because its probability will become zero after applying softmax.
                    # This ensures that only image-related tokens can be chosen while generating tokens within an image row.
                    return image_constrained_scores
        else:
            print("Something wrong in the decoding process.")

        return scores


class InterleavedTopKLogitsWarper(LogitsWarper):
    r"""
    [`LogitsWarper`] that performs top-k, i.e. restricting to the k highest probability elements. Often used together
    with [`TemperatureLogitsWarper`] and [`TopPLogitsWarper`].
    """

    def __init__(
        self,
        image_top_k: int,
        text_top_k: int,
        image_start_token_id=None,
        image_end_token_id=None,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
    ):  
        # Validate that image_top_k and text_top_k are strictly positive integers.
        if not isinstance(text_top_k, int) or text_top_k <= 0:
            raise ValueError(f"`text_top_k` has to be a strictly positive integer, but is {text_top_k}")
        if not isinstance(image_top_k, int) or text_top_k <= 0:
            raise ValueError(f"`image_top_k` has to be a strictly positive integer, but is {image_top_k}")

        # Set the values for image and text top-k, ensuring a minimum number of tokens are kept.
        self.image_top_k = max(image_top_k, min_tokens_to_keep)
        self.text_top_k = max(text_top_k, min_tokens_to_keep)
        self.filter_value = filter_value

        # Set the token IDs for identifying image boundaries.
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id

        # Initialize additional attributes used for tracking the number of image tokens.
        self.flag = False
        self.num_image_start_tokens = None
        self.num_image_end_tokens = None

    # @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        The core logic of the logits processor. Depending on the current context (image or text generation),
        this function will restrict the scores to only the top-k tokens with the highest probabilities.
        
        Args:
            input_ids (torch.LongTensor): Input IDs representing the tokens that have been generated so far.
            scores (torch.FloatTensor): The logits output from the model, representing scores for all tokens.

        Returns:
            torch.FloatTensor: The scores after applying the top-k restriction.
        """
        # Count the number of image start and end tokens in the current sequence.
        self.num_image_start_tokens = (input_ids[0] == self.image_start_token_id).sum()
        self.num_image_end_tokens = (input_ids[0] == self.image_end_token_id).sum()

        # Determine whether we are in the process of generating image-related tokens or text-related tokens.
        if self.num_image_start_tokens == self.num_image_end_tokens + 1:
            # If the number of image start tokens is exactly one more than the end tokens,
            # it implies that we are still generating within an image section.
            top_k = min(self.image_top_k, scores.size(-1))
        else:
            # Otherwise, we assume we are generating text, so use the text_top_k setting.
            top_k = min(self.text_top_k, scores.size(-1))  # Safety check

        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
        scores_processed = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores_processed





class FlexARInferenceSolver:
    @classmethod
    def get_args_parser(cls):
        parser = argparse.ArgumentParser("xllmx Inference", add_help=False)
        parser.add_argument("--model_path", type=str)
        parser.add_argument("--precision", type=str, choices=["fp16", "bf16", "tf32"], default="bf16")

        return parser

    def __init__(self, model_path, precision, target_size=512):
        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        self.model = ChameleonForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            device_map="cuda",
        )
        self.item_processor = FlexARItemProcessor(target_size=target_size)

    def get_streamer(self):
        return TextStreamer(self.item_processor.tokenizer)

    @torch.no_grad()
    def generate(
        self,
        images: Image.Image | str | List[Union[Image.Image, str]],
        qas,
        max_gen_len,
        temperature,
        logits_processor=None,
        streamer=None,
    ):
        ### Prepare Conversations 
        conversations = []
        for q, a in qas:
            conversations.append(
                {
                    "from": "human",
                    "value": q,
                }
            )
            conversations.append(
                {
                    "from": "gpt",
                    "value": a,
                }
            )
        item = {"image": images, "conversations": conversations}

        ### Convert the images and conversations into a format that the model can understand
        _prompt = self.item_processor.process_item(item)
        prompt = []
        for value in _prompt:
            if isinstance(value, int):
                prompt.append(value)
            else: # which means it‘s a media dict, we extract its "input_ids"
                prompt += value["input_ids"]
        prompt_len = len(prompt)
        prompt = torch.tensor(prompt, dtype=torch.int64, device=self.model.device).unsqueeze(0)

        generation_config = GenerationConfig(
            max_new_tokens=max_gen_len,
            max_length=self.model.config.max_position_embeddings,
            temperature=temperature,
            top_k=None,
            do_sample=True,
            eos_token_id=[8710],
        )

        if logits_processor is None:
            logits_processor = self.create_logits_processor()
        
        with torch.cuda.amp.autocast(dtype=self.dtype):
            # Original generate() function.
            # https://github.com/huggingface/transformers/blob/617b21273a349bd3a94e2b3bfb83f8089f45749b/src/transformers/generation/utils.py#L1860
            generation_result = self.model.generate(
                prompt, 
                generation_config, 
                logits_processor=logits_processor, 
                streamer=streamer
            )[0][prompt_len:].tolist()
            if len(generation_result) > 0 and generation_result[-1] == 8710:
                generation_result = generation_result[:-1]
        
        return self.decode_ids(generation_result)

    def decode_ids(self, tokens: List[int]):
        generated_images = []
        generation_result_processed = []
        i = 0
        while i < len(tokens):
            token_id = tokens[i]
            if token_id == self.item_processor.token2id(self.item_processor.image_start_token):
                cache = []
                for j in range(i + 1, len(tokens)):
                    if tokens[j] != self.item_processor.token2id(self.item_processor.image_end_token):
                        cache.append(tokens[j])
                        i = j + 1
                    else:
                        image = self.decode_image(cache)
                        generated_images.append(image)
                        generation_result_processed.append(self.item_processor.token2id("<|image|>"))
                        i = j + 1
                        break
            else:
                generation_result_processed.append(token_id)
                i += 1

        generated = self.item_processor.tokenizer.decode(generation_result_processed)

        return generated, generated_images

    def decode_image(self, tokens: List[int]):
        return self.item_processor.decode_image(tokens)

    @staticmethod
    def create_image_grid(images, rows, cols):
        width, height = images[0].size

        grid_img = Image.new("RGB", (cols * width, rows * height))

        for i, img in enumerate(images):
            row = i // cols
            col = i % cols
            grid_img.paste(img, (col * width, row * height))

        return grid_img

    def create_logits_processor(self, cfg=3.0, image_top_k=2000, text_top_k=10):
        logits_processor = LogitsProcessorList()

        cfg_processor = LLMImageStartTriggeredUnbatchedClassifierFreeGuidanceLogitsProcessor(
            guidance_scale=cfg,
            model=self.model,
            image_start_token_id=self.item_processor.token2id(self.item_processor.image_start_token),
            image_end_token_id=self.item_processor.token2id(self.item_processor.image_end_token),
            image_next_line_token_id=self.item_processor.token2id(self.item_processor.new_line_token),
            patch_size=32,
        )

        candidate_processor = MultiModalLogitsProcessor(
            image_start_token_id=self.item_processor.token2id(self.item_processor.image_start_token),
            image_end_token_id=self.item_processor.token2id(self.item_processor.image_end_token),
            image_next_line_token_id=self.item_processor.token2id(self.item_processor.new_line_token),
            patch_size=32,
            voc_size=self.model.config.vocab_size,
        )

        topk_processor = InterleavedTopKLogitsWarper(
            image_top_k=image_top_k,
            text_top_k=text_top_k,
            image_start_token_id=self.item_processor.token2id(self.item_processor.image_start_token),
            image_end_token_id=self.item_processor.token2id(self.item_processor.image_end_token),
        )

        logits_processor.append(cfg_processor)
        logits_processor.append(candidate_processor)
        logits_processor.append(topk_processor)

        return logits_processor


if __name__ == "__main__":
    parser = FlexARInferenceSolver.get_args_parser()
    args = parser.parse_args()
    solver = FlexARInferenceSolver(**vars(args))
