import argparse
import copy
import math
import time
from typing import List, Optional, Union
import json
import cv2
import numpy as np
import os
import random

from PIL import Image
import torch
import transformers
from transformers import GenerationConfig, TextStreamer
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList, LogitsWarper
from transformers import ChameleonConfig

# from data.item_processor import FlexARItemProcessor
from data.item_processor import FlexARItemProcessor2
from model.chameleon import ChameleonForConditionalGeneration, ChameleonMMRoPEForConditionalGeneration
from model.utils import calculate_resolution_from_token_list, mask_append, create_attention_mask_t2v
from model.sampling import cosine_schedule, mask_by_random_topk



class LLMVideoStartTriggeredUnbatchedClassifierFreeGuidanceLogitsProcessor(LogitsProcessor):
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
        video_start_token_id,
        video_end_token_id,
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

        self.video_start_token_id = video_start_token_id
        self.video_end_token_id = video_end_token_id
        self.image_start_token_id_index = None
        self.video_start_token_id_index = None
        self.patch_size = patch_size

        self.h_latent_dim = None
        self.w_latent_dim = None
        self.frame_latent_dim = None

    def get_unconditional_logits(self, input_ids, media_start_token_id_index):
        '''
            Generate unconditional logits using the model, without the influence of the specific prompt.
        '''

        # If this is the first pass, initialize the input_ids and attention mask for the unconditional context
        if self.unconditional_context["first_pass"]:
            if self.unconditional_context["input_ids"] is None:
                # Set the input IDs to start from the last media start token index
                self.unconditional_context["input_ids"] = input_ids[:, media_start_token_id_index:]
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
        
        # Count the number of video start and end tokens in the input sequence.
        num_video_start_tokens = (input_ids[0] == self.video_start_token_id).sum()
        num_video_end_tokens = (input_ids[0] == self.video_end_token_id).sum()

        # If the number of start tokens and end tokens are equal, it means we're not in the middle of generating an image.
        if num_video_start_tokens == num_video_end_tokens:
            # Reset latent dimensions, token index, and unconditional context
            self.h_latent_dim, self.w_latent_dim, self.frame_latent_dim = None, None, None
            self.video_start_token_id_index = None
            self.unconditional_context = None
            return scores  # Return the original scores without modification
        
        # If there is one more start token than end tokens, we're currently generating a video
        elif num_video_start_tokens == num_video_end_tokens + 1:

            # Identify the index of the last video start token if not already set
            if self.video_start_token_id_index is None:
                self.video_start_token_id_index = torch.where(input_ids[0] == self.video_start_token_id)[0][-1].item()
            
            # Determine the number of tokens that have been generated since the video start token
            new_token_num = len(input_ids[0][self.video_start_token_id_index + 1 :])
            
            if new_token_num >= 2:
                # Calculate the latent dimensions if they haven't been set yet
                if self.frame_latent_dim is None:
                    duration_grids, fps_grids = (
                        input_ids[0][self.video_start_token_id_index + 1] - 8804,
                        input_ids[0][self.video_start_token_id_index + 2] - 8804,
                    )
                    self.frame_latent_dim = duration_grids * fps_grids
                

                # Set the unconditional context for the first time if it hasn't been set yet
                if self.unconditional_context is None:
                    self.unconditional_context = copy.deepcopy(self.unconditional_context_backup)

                # If the guidance scale is 1.0, there is no need to modify the scores
                if self.guidance_scale == 1.0:
                    return scores

                # Compute unconditional logits using the model
                unconditional_logits = self.get_unconditional_logits(input_ids, self.video_start_token_id_index)[:, -1]

                # Apply Classifier-Free Guidance to adjust the scores
                scores_processed = self.guidance_scale * (scores - unconditional_logits) + unconditional_logits
                return scores_processed
        else:
            print("Something wrong in the decoding process.")

        return scores



class VideoLogitsProcessor(LogitsProcessor):
    '''
        Initialize the logits processor with necessary image-specific tokens and settings
    '''
    def __init__(
        self,
        image_start_token_id=None,
        image_end_token_id=None,
        image_next_line_token_id=None,
        video_start_token_id=None,
        video_end_token_id=None,
        patch_size=None,
        voc_size=None,
        visual_tokenizer="Chameleon",
    ):
        # Token IDs that identify the start, end, and next line of an image section
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.image_next_line_token_id = image_next_line_token_id
        self.video_start_token_id = video_start_token_id
        self.video_end_token_id = video_end_token_id

        self.image_start_token_id_index = None  # By default, the last image_start_token.
        self.first_image_start_token_id_index = None # By default, the first image_start_token.
        self.video_start_token_id_index = None
        self.patch_size = patch_size
        self.h_latent_dim = None
        self.w_latent_dim = None
        self.frame_latent_dim = None

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

        self.visual_tokenizer = visual_tokenizer
        if "Cosmos-Tokenizer" in self.visual_tokenizer:
            # `cosmos_video_token_list` represents token IDs used for cosmos video representation
            self.cosmos_video_token_list = [i for i in range(65536, 129536)]
            # `suppress_cosmos_video_tokens` is a list of tokens not relevant for cosmos video data
            self.suppress_cosmos_video_tokens = torch.tensor(
                [x for x in self.vocab_list if x not in self.cosmos_video_token_list], device="cuda"
            )
            # Mask used to suppress non-cosmos video tokens in video generation
            self.suppress_cosmos_video_token_mask = torch.isin(self.vocab_tensor, self.suppress_cosmos_video_tokens)

        # Additional flags and counters to track the generation state
        self.flag = False
        self.num_image_start_tokens = None
        self.num_image_end_tokens = None
        self.num_video_start_tokens = None
        self.num_video_end_tokens = None

    # @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        
        # Count the number of video start and end tokens in the input sequence.
        self.num_video_start_tokens = (input_ids[0] == self.video_start_token_id).sum()
        self.num_video_end_tokens = (input_ids[0] == self.video_end_token_id).sum()

        # If the number of start tokens and end tokens are equal, it means we're not in the middle of generating an image.
        if self.num_video_start_tokens == self.num_video_end_tokens:
            # Reset latent dimensions, token index, and unconditional context
            self.h_latent_dim, self.w_latent_dim, self.frame_latent_dim = None, None, None
            self.video_start_token_id_index = None
            self.unconditional_context = None
            return scores  # Return the original scores without modification
        
        # If we have one more start token than end tokens, it means a video section is being generated
        elif self.num_video_start_tokens == self.num_video_end_tokens + 1:

            # Identify the index of the last image start token if not already set
            if self.video_start_token_id_index is None:
                self.video_start_token_id_index = torch.where(input_ids[0] == self.video_start_token_id)[0][-1].item()

            # Calculate the number of new tokens generated after the image start token
            new_token_num = len(input_ids[0][self.video_start_token_id_index + 1 :])

            if new_token_num >= 2:
                # Calculate the latent dimensions if they haven't been set yet
                if self.frame_latent_dim is None:
                    duration_grids, fps_grids = (
                        input_ids[0][self.video_start_token_id_index + 1] - 8804,
                        input_ids[0][self.video_start_token_id_index + 2] - 8804,
                    )
                    self.frame_latent_dim = duration_grids * fps_grids
                    print(f"frame_latent_dim: {self.frame_latent_dim}")


                # Count the number of image start and end tokens in the current video
                self.num_image_start_tokens = (input_ids[0][self.video_start_token_id_index + 3 :] == self.image_start_token_id).sum()
                self.num_image_end_tokens = (input_ids[0][self.video_start_token_id_index + 3 :] == self.image_end_token_id).sum()



                if self.num_image_start_tokens == self.num_image_end_tokens == self.frame_latent_dim:
                    # Force the model to output the end-of-video token
                    eos_video_constrained_scores = torch.full_like(scores, -math.inf)
                    eos_video_constrained_scores[:, self.video_end_token_id] = 0
                    # print(f"eos video: {len(tokens)+1}")
                    print(f"eos video: {len(input_ids[0])+1}")
                    
                    return eos_video_constrained_scores

                elif self.num_image_start_tokens == self.num_image_end_tokens:
                    # Force the model to output the start image token
                    new_image_constrained_scores = torch.full_like(scores, -math.inf)
                    new_image_constrained_scores[:, self.image_start_token_id] = 0
                    self.image_start_token_id_index = None
                    self.first_image_start_token_id_index = None
                    # print(f"new frame: {len(tokens)+1}")
                    print(f"Starting {self.num_image_start_tokens+1}th frame: {len(input_ids[0])+1}")

                    return new_image_constrained_scores
                
                elif self.num_image_start_tokens == self.num_image_end_tokens + 1:

                    # Find the index of the last image start token if not already determined
                    if self.image_start_token_id_index is None:
                        ### Find the image_start_token 
                        self.image_start_token_id_index = torch.where(input_ids[0] == self.image_start_token_id)[0][-1].item()
                    if self.first_image_start_token_id_index is None:
                        ### Find the first image_start_token after video_start_token
                        all_image_start_token_id_index = torch.where(input_ids[0] == self.image_start_token_id)[0]
                        for ais in all_image_start_token_id_index:
                            if ais.item() > self.video_start_token_id_index:
                                self.first_image_start_token_id_index = ais.item()
                                break
                    
                    # Calculate the number of new tokens generated after the image start token
                    new_image_token_num = len(input_ids[0][self.image_start_token_id_index + 1 :])
                    

                    ### New version for reading h and w
                    if new_image_token_num == 0:
                        if self.num_image_start_tokens >= 2:  # which means the second or later frames.
                            if self.h_latent_dim is None: # read out the h_latent_dim from the given token list 
                                h_grids = input_ids[0][self.first_image_start_token_id_index + 1] - 8804
                                self.h_latent_dim = h_grids * 2
                                print(f"h_grids: {h_grids}, h_latent_dim: {self.h_latent_dim}")

                            h_token = self.h_latent_dim//2 + 8804
                            h_token_constrained_scores = torch.full_like(scores, -math.inf)
                            h_token_constrained_scores[:, h_token] = 0
                            print(f"new frame h: {self.h_latent_dim}")
                            return h_token_constrained_scores


                    elif new_image_token_num == 1:
                        if self.num_image_start_tokens >= 2:  # which means the second or later frames.
                            if self.w_latent_dim is None: # read out the w_latent_dim from the given token list 
                                # 需要自己写，因为很可能第一帧就定好了，但是第一帧不是生成的，所以我们这里需要重新读取
                                w_grids = input_ids[0][self.first_image_start_token_id_index + 2] - 8804
                                self.w_latent_dim = w_grids * 2
                                print(f"w_grids: {w_grids}, w_latent_dim: {self.w_latent_dim}")
                        
                            w_token = self.w_latent_dim//2 + 8804
                            w_token_constrained_scores = torch.full_like(scores, -math.inf)
                            w_token_constrained_scores[:, w_token] = 0
                            print(f"new frame w: {self.w_latent_dim}")
                            return w_token_constrained_scores

                            # TODO Using this version.
                            # def force_token(scores, token_id, token_type):
                            #     constrained_scores = torch.full_like(scores, -math.inf)
                            #     constrained_scores[:, token_id] = 0
                            #     print(f"new frame {token_type}: Forcing generation of {token_type} token.")
                            #     return constrained_scores


                    elif new_image_token_num >= 2:
                    
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
                            # Constrain the scores to suppress non-image/video tokens
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




class FlexARVidInferenceSolver:
    @classmethod
    def get_args_parser(cls):
        parser = argparse.ArgumentParser("xllmx Inference", add_help=False)
        parser.add_argument("--model_path", type=str)
        parser.add_argument("--precision", type=str, choices=["fp16", "bf16", "tf32"], default="bf16")

        return parser

    def __init__(self, model_path, precision, target_fps, duration, visual_tokenizer="Chameleon", vae_st_compress=None, target_size=512):
        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        ### Model init with pre-trained weights.
        # self.model = ChameleonForConditionalGeneration.from_pretrained(
        self.model = ChameleonMMRoPEForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            device_map="cuda",
        )

        # self.item_processor = FlexARItemProcessor(target_size=target_size)
        self.item_processor = FlexARItemProcessor2(
            target_size=target_size, 
            target_fps=target_fps, 
            duration=duration,
            inference_mode=True,
            visual_tokenizer=visual_tokenizer,
            cosmos_dtype=self.dtype if "Cosmos-Tokenizer" in visual_tokenizer else None,
        )

        self.vae_st_compress = vae_st_compress # [16, 1] for Chameleon
        
        self.visual_tokenizer = visual_tokenizer
        if "Cosmos-Tokenizer" in self.visual_tokenizer:
            self.temporal_compression, self.spatial_compression = [int(i) for i in self.visual_tokenizer.split('DV')[1].split('x')[:2]]

    def get_streamer(self):
        return TextStreamer(self.item_processor.tokenizer)

    @torch.no_grad()
    def generate(
        self,
        partial_videos: Union[str, List[Image.Image], List[Union[str, List[Image.Image]]]],
        qas,
        max_gen_len,
        temperature,
        output_video_path,
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
        # item = {"image": images, "conversations": conversations}
        item = {"partial_video": partial_videos, "conversations": conversations}

        ### Convert the images and conversations into a format that the model can understand
        _prompt = self.item_processor.process_item(item)

        prompt = []
        for value in _prompt:
            if isinstance(value, int):
                prompt.append(value)
            else: # which means it‘s a media dict, we extract its "input_ids"
                # prompt += value["input_ids"]
                input_ids = value["input_ids"]
                prompt += input_ids

        assert prompt[-1] == self.item_processor.token2id(self.item_processor.sep_token)
        prompt = prompt[:-1]
        assert prompt[-1] == self.item_processor.token2id(self.item_processor.video_end_token)
        prompt = prompt[:-1]

        prompt_len = len(prompt)
        prompt = torch.tensor(prompt, dtype=torch.int64, device=self.model.device).unsqueeze(0)

        generation_config = GenerationConfig(
            max_new_tokens=max_gen_len,
            max_length=self.model.config.max_position_embeddings,
            temperature=temperature,
            top_k=None,
            do_sample=True,  # Whether or not to use sampling ; use greedy decoding otherwise.
            eos_token_id=[8710],
        )

        if logits_processor is None:
            logits_processor = self.create_logits_processor()
        
        with torch.cuda.amp.autocast(dtype=self.dtype):
            start_time = time.time()  # Start timing

            # Original generate() function.
            # https://github.com/huggingface/transformers/blob/617b21273a349bd3a94e2b3bfb83f8089f45749b/src/transformers/generation/utils.py#L1860
            generation_result = self.model.generate(
                prompt,
                generation_config,
                logits_processor=logits_processor,
                streamer=streamer,
            )[0].tolist()  # [prompt_len:].tolist()
        
            video_start_token = self.item_processor.token2id(self.item_processor.video_start_token)
            video_start_indices = [index for index, value in enumerate(generation_result) if value == video_start_token]
            generation_result = generation_result[video_start_indices[-1]:]

            if len(generation_result) > 0 and generation_result[-1] == 8710:
                generation_result = generation_result[:-1]
            
            # End timing
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time taken for generation: {elapsed_time:.2f} seconds")
        

        return self.decode_video_ids(generation_result, output_video_path)
    
        

    ### V8: NOTE to be completed.
    @torch.no_grad()
    def generate_maskedAR_mmrope_v8_newcache(
        self,
        qas,
        max_gen_len,
        temperature,
        output_video_path,
        fps_duration,
        generation_task,
        # iterations,
        use_kv_cache=True,
        timesteps=18,  # ideal number of steps is 18 in maskgit paper
        guidance_scale=None,
        noise_schedule=cosine_schedule,
        generator: torch.Generator = None,
        video_resolution=None, # like [448, 256] "448x256"
        mask_history_ratio=None,
        partial_videos: Union[str, List[Image.Image], List[Union[str, List[Image.Image]]]]=[],
        logits_processor=None,
        streamer=None,
    ):

        ### Prepare token_id
        image_start_token_id = self.item_processor.token2id(self.item_processor.image_start_token)
        image_end_token_id   = self.item_processor.token2id(self.item_processor.image_end_token)
        video_start_token_id = self.item_processor.token2id(self.item_processor.video_start_token)
        video_end_token_id   = self.item_processor.token2id(self.item_processor.video_end_token)
        sep_token_id         = self.item_processor.token2id(self.item_processor.sep_token)
        new_line_token_id    = self.item_processor.token2id(self.item_processor.new_line_token)
        mask_token_id        = self.item_processor.token2id(self.item_processor.mask_token)
        special_token = {
            "image_start_token": image_start_token_id, # "<racm3:break>" # fixed tokens for start and end, so can hardcode
            "image_end_token": image_end_token_id,     # "<eoss>" 
            "new_line_token": new_line_token_id,       # "<reserved08799>"
            "sep_token": sep_token_id,                 # "<reserved08706>"
            "video_start_token": video_start_token_id, # "<reserved09000>"
            "video_end_token": video_end_token_id,     # "<reserved09001>"
            "mask_token": mask_token_id,               # "<reserved08999>"
        }

        ### Replace <video_resolution_placeholder> string
        if generation_task=="t2v":
            video_resolution_str = f"{int(video_resolution[0])}x{int(video_resolution[1])}" # like "448x256"
            qas[0][0] = qas[0][0].replace("<video_resolution_placeholder>", video_resolution_str) # the question of the first qa.
            # replace is not an in-place operation.
        elif generation_task=="vp":
            pass # <video_resolution_placeholder> will be replaced in replace_resolution_token().

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
        # item = {"image": images, "conversations": conversations}
        item = {"partial_video": partial_videos, "conversations": conversations}

        ### Convert the images and conversations into a format that the model can understand
        _prompt = self.item_processor.process_item(item)

        prompt = []
        for value in _prompt:
            if isinstance(value, int):
                prompt.append(value)
            else: # which means it‘s a media dict, we extract its "input_ids"
                # prompt += value["input_ids"]
                input_ids = value["input_ids"]
                prompt += input_ids


        if generation_task=="t2v":
            # Drop the final sep_token generated by the empty gpt answer.
            assert prompt[-1] == sep_token_id
            prompt = prompt[:-1]
            # For t2v, we need to add video start tokens.
            video_init_tokens = [
                video_start_token_id,
                self.item_processor.token2id(self.item_processor.get_n_grids_token(fps_duration[1])),
                self.item_processor.token2id(self.item_processor.get_n_grids_token(fps_duration[0]))
            ] # video start token: [video_start_token, duration_token, fps_token,...]
            prompt += video_init_tokens
        elif generation_task=="vp":
            # We need to drop the final sep_token and video_end_token because frames are read from the video.
            assert prompt[-1] == sep_token_id
            prompt = prompt[:-1]
            assert prompt[-1] == video_end_token_id
            prompt = prompt[:-1]

        prompt_len = len(prompt)
        prompt = torch.tensor(prompt, dtype=torch.int64, device=self.model.device).unsqueeze(0) # unsqueeze to obtain the batch dimension.

        ### Read out resolution stat from the token list.
        if generation_task=="t2v":
            pass
        elif generation_task=="vp":
            video_resolution = calculate_resolution_from_token_list(
                sequence=prompt, 
                soi_id=image_start_token_id, 
                eoi_id=image_end_token_id,
                new_line_id=new_line_token_id,
                spatial_compress_ratio=self.vae_st_compress[0],
            )
        resolution_token = [self.item_processor.token2id(self.item_processor.get_n_grids_token(video_resolution[1]//self.item_processor.patch_size)),
                            self.item_processor.token2id(self.item_processor.get_n_grids_token(video_resolution[0]//self.item_processor.patch_size))]
                            # NOTE resolution_token is [h, w] rather than [w,h].
        resolution_token = torch.tensor(resolution_token, dtype=prompt.dtype, device=prompt.device)

        with torch.cuda.amp.autocast(dtype=self.dtype):
            start_time  = time.time()  # Start timing
            if (prompt == image_start_token_id).sum() == (prompt == image_end_token_id).sum():
                start_frame = (prompt == image_start_token_id).sum().item()
            else:
                assert False, "The number of frame start tokens and end tokens are different."
            end_frame = fps_duration[0]*fps_duration[1]
            # end_frame = 9 # for debugging
            if "Cosmos-Tokenizer" in self.visual_tokenizer:
                end_frame = math.ceil((end_frame - 1)/self.temporal_compression) + 1
            print(f"Latent frame: {start_frame} to {end_frame}.")

            past_kv = transformers.DynamicCache() if use_kv_cache else None
            uncond_past_kv = transformers.DynamicCache() if use_kv_cache and (guidance_scale is not None and guidance_scale != 1) \
                             else None
            for frame_idx in range(start_frame, end_frame):
                ### Mask token append function
                prompt, prompt_mask = mask_append(
                    sequence               = prompt,                  # Tensor: Input sequence of tokens to which a masked frame will be appended.
                    resolution             = video_resolution,        # Tuple[int, int]: Original resolution (width, height) of the input image.
                    resolution_token       = resolution_token,        # Tensor: Token that represents the compressed resolution of the image.
                    spatial_compress_ratio = self.vae_st_compress[0], # int: Ratio used to compress the resolution of the image spatially.
                    mask_token_id          = mask_token_id,           # int: Token ID representing a masked element.
                    new_line_id            = new_line_token_id,       # int: Token ID representing a new line (used when appending rows).
                    image_start_token_id   = image_start_token_id,    # int: Token ID representing the start of an image in the sequence.
                    image_end_token_id     = image_end_token_id       # int: Token ID representing the start of an image in the sequence.
                )
                assert prompt_mask.sum().item() == video_resolution[0]*video_resolution[1]//self.vae_st_compress[0]**2, \
                    "prompt_mask does not have correct number of masks."

                ### Attention mask generation
                attention_mask = create_attention_mask_t2v(
                    sequence=prompt, special_token=special_token, pad_id=0, 
                    rm_pad_in_image=False, return_inverse_mask=True
                )

                ### Prepare for uncond data for classifier-free gudiance
                if guidance_scale is not None and guidance_scale != 1:
                    uncond_prompt, uncond_prompt_mask, uncond_attention_mask = \
                        self.uncond_prompt_from_prompt(prompt, attention_mask, video_start_token_id, mask_token_id)


                ### Prepare position_ids, which is shared for all timesteps (for a given frame).
                position_ids = self.generate_mmrope_position_ids(prompt)
                uncond_position_ids = self.generate_mmrope_position_ids(uncond_prompt)
                last_image_start = torch.where(prompt[0]==image_start_token_id)[-1][-1]
                uncond_last_image_start = torch.where(uncond_prompt[0]==image_start_token_id)[-1][-1]
                # preserve full position_ids to run at the first step to get KV for texts
                position_ids_full = position_ids
                uncond_position_ids_full = uncond_position_ids
                position_ids_last_image = self.crop_mmrope_position_ids(position_ids, last_image_start)
                uncond_position_ids_last_image = self.crop_mmrope_position_ids(uncond_position_ids, uncond_last_image_start)
                
                ### Default
                timesteps_list = list(range(timesteps))

                for step in timesteps_list:
                    # print("step ", step)
                    if guidance_scale is not None and guidance_scale != 1: # 1 means no guidance.
                        # ###### V1: without KV cache
                        # cond_logits = self.model(prompt, attention_mask=attention_mask)[0] # shape like [1, 4409, 65536]
                        # uncond_logits = self.model(uncond_prompt, attention_mask=uncond_attention_mask)[0]
                        # logits = guidance_scale * (cond_logits[prompt_mask] - uncond_logits[uncond_prompt_mask]) \
                        #         + uncond_logits[uncond_prompt_mask] # shape like [448, 65536] 

                        ###### V3: with KV cache (Correct version)
                        ### Prepare prompt and attention mask to be compatible with dropped kv cache
                        if use_kv_cache:
                            if step==0 and frame_idx==start_frame:
                                # cond prompt and attention mask
                                prompt_for_cache = prompt.clone()
                                attention_mask_for_cache = attention_mask.clone()
                                # uncond prompt and attention mask
                                uncond_prompt_for_cache = uncond_prompt.clone()
                                uncond_attention_mask_for_cache = uncond_attention_mask.clone()
                                position_ids_cache = position_ids_full
                                uncond_position_ids_cache = uncond_position_ids_full

                                # generate mask_pattern only once (step==0 and frame_idx==start_frame)
                                last_image_start = torch.where(prompt[0] == image_start_token_id)[-1][-1]
                                history_length = prompt[:, last_image_start:].shape[-1] - 4 # "-4" means we eliminate image_start_token, h_grid_token, w_grid_token and image_end_token.   
                                # mask_pattern = self.mask_history_pattern(history_length, prompt.device)
                                mask_pattern = self.mask_history_pattern(history_length, prompt.device, mask_history_ratio) if mask_history_ratio is not None \
                                               else self.mask_history_pattern(history_length, prompt.device)

                                ### Video prediction-specific operations.
                                # mask the history if we are doing video prediction.
                                if frame_idx == start_frame and start_frame >= 1: # This means that this is video prediction and we should mask these frames as well.
                                    print(f"{(prompt == mask_token_id).sum()} mask tokens before masking.")
                                    prompt_no_mask_vp = prompt.clone()
                                    for prev_idx in range(0, start_frame):
                                        prev_image_start, prev_image_end = torch.where(prompt[0]==image_start_token_id)[-1][prev_idx:(prev_idx + 2)]
                                        prev_prompt = prompt[:,prev_image_start:prev_image_end]
                                        prev_prompt_partial, _ = self.mask_partial_histroy(prev_prompt, mask_token_id, mask_pattern)
                                        prompt[:,prev_image_start:prev_image_end] = prev_prompt_partial
                                        print(f"{(prompt[:,prev_image_start:prev_image_end] == mask_token_id).sum()} mask tokens.")
                                    print(f"{(prompt == mask_token_id).sum()} mask tokens after masking.")

                            else:
                                ### Video prediction-specific operations.
                                # replace the masked prompt with the unmasked one.
                                if frame_idx == start_frame and start_frame >= 1 and step==1: # This means that this is video prediction.
                                    last_image_start = torch.where(prompt[0]==image_start_token_id)[-1][-1]
                                    prompt[:,:last_image_start] = prompt_no_mask_vp[:,:last_image_start]

                                # cond prompt and attention mask
                                last_image_start = torch.where(prompt[0]==image_start_token_id)[-1][-1]
                                prompt_for_cache = prompt[:,last_image_start:]
                                attention_mask_for_cache = attention_mask[:,:,last_image_start:,:]
                                # uncond prompt and attention mask
                                uncond_last_image_start = torch.where(uncond_prompt[0]==image_start_token_id)[-1][-1]
                                uncond_prompt_for_cache = uncond_prompt[:,uncond_last_image_start:]
                                uncond_attention_mask_for_cache = uncond_attention_mask[:,:,uncond_last_image_start:,:]
                                # position_ids cache
                                position_ids_cache = position_ids_last_image
                                uncond_position_ids_cache = uncond_position_ids_last_image
                        else:
                            pass
                        
                        ### Model inference.
                        cond_output = self.model(
                            input_ids=prompt if not use_kv_cache else prompt_for_cache,
                            attention_mask=attention_mask if not use_kv_cache else attention_mask_for_cache,
                            past_key_values=past_kv,  # Use past key values for caching
                            use_cache=use_kv_cache,   # Enable caching
                            position_ids=position_ids_cache,
                        )                             # Assuming the model returns (logits, past_key_values)
                        cond_logits = cond_output.get("logits", None)
                        uncond_output = self.model(
                            input_ids=uncond_prompt if not use_kv_cache else uncond_prompt_for_cache,
                            attention_mask=uncond_attention_mask if not use_kv_cache else uncond_attention_mask_for_cache,
                            past_key_values=uncond_past_kv,
                            use_cache=use_kv_cache,
                            position_ids=uncond_position_ids_cache,
                        )
                        uncond_logits = uncond_output.get("logits", None)


                        ### Extract logits corresponding to the last image.
                        # NOTE logits in cond_output or uncond_output follows the length of “uncond_prompt_for_cache” and "uncond_attention_mask_for_cache".
                        # they are obtained by cutting form uncond_prompt and uncond_attention_mask to make use of past_kv.
                        # Therefore, it follows the logic of this step "Prepare prompt and attention mask to be compatible with dropped kv cache".
                        if use_kv_cache:
                            if step==0 and frame_idx==start_frame:
                                logits = guidance_scale * (cond_logits[prompt_mask] - uncond_logits[uncond_prompt_mask]) \
                                    + uncond_logits[uncond_prompt_mask] # shape like [448, 65536]
                            else:
                                last_image_start = torch.where(prompt[0]==image_start_token_id)[-1][-1]
                                last_image_content_mask = prompt_mask[:,last_image_start:]
                                uncond_last_image_start = torch.where(uncond_prompt[0]==image_start_token_id)[-1][-1]
                                uncond_last_image_content_mask = uncond_prompt_mask[:,uncond_last_image_start:]
                                logits = guidance_scale * (cond_logits[last_image_content_mask] - uncond_logits[uncond_last_image_content_mask]) \
                                         + uncond_logits[uncond_last_image_content_mask] # shape like [448, 65536]
                        else:
                            logits = guidance_scale * (cond_logits[prompt_mask] - uncond_logits[uncond_prompt_mask]) \
                                     + uncond_logits[uncond_prompt_mask] # shape like [448, 65536]


                        ### Drop the cache for the current image since it should not be used for next-step inference.
                        if use_kv_cache:
                            last_image_start = torch.where(prompt[0]==image_start_token_id)[-1][-1]
                            past_kv.crop(last_image_start)
                            uncond_last_image_start = torch.where(uncond_prompt[0]==image_start_token_id)[-1][-1]
                            uncond_past_kv.crop(uncond_last_image_start)

                    else:
                        assert False, "Not implemented when cfg is not used."


                    logits = logits.unsqueeze(dim=0) # shape like [1, 448, 65536]
                    current_frame_token = prompt[prompt_mask].unsqueeze(dim=0) # shape like [1, 448]

                    # NOTE Suppress the non-image token
                    if self.visual_tokenizer == "Chameleon":
                        logits = torch.where(logits_processor[1].suppress_token_mask, -float("inf"), logits)
                    elif "Cosmos-Tokenizer" in self.visual_tokenizer:
                        logits = torch.where(logits_processor[1].suppress_cosmos_video_token_mask, -float("inf"), logits)
                    else:
                        raise NotImplementedError(f"{self.visual_tokenizer} is not implemented.")

                    # Apply softmax to convert logits into probabilities
                    probs = logits.softmax(dim=-1)
                    # Flatten and sample from the distribution to get new token IDs
                    sampled = probs.reshape(-1, logits.size(-1))
                    sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1]) # shape like [1, 448]

                    # Create an unknown map to identify which tokens are still masked
                    unknown_map = current_frame_token == mask_token_id
                    # print(unknown_map.sum())
                    # Replace the masked tokens with sampled values
                    sampled_ids = torch.where(unknown_map, sampled_ids, current_frame_token)

                    # Defines the mask ratio for the next round. The number to mask out is
                    # determined by mask_ratio * unknown_number_in_the_beginning.
                    ratio = 1.0 * (step + 1) / timesteps
                    mask_ratio = noise_schedule(torch.tensor(ratio))
                    # Computes the probabilities of each selected tokens.
                    selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None]) # shape like [1, 448, 1]
                    selected_probs = selected_probs.squeeze(-1) # shape like [1, 448]

                    # Ignore tokens that were not masked by setting their confidence to max
                    selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
                    # Gets mask lens for each sample in the batch according to the mask ratio.
                    # Determine how many tokens should be masked for the next step
                    mask_len = (current_frame_token.shape[-1] * mask_ratio).floor().unsqueeze(0).to(logits.device)
                    # Keeps at least one of prediction in this round and also masks out at least
                    # one and for the next iteration
                    mask_len = torch.max(
                        torch.tensor([1], device=logits.device), 
                        torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
                    ) # like [[446.]]

                    # Apply masking with some randomness using temperature to control exploration
                    temperature = temperature * (1.0 - ratio)
                    masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)
                    # Update the input_ids by masking low-confidence tokens and replacing them with `mask_token_id`
                    if step == timesteps_list[-1]:
                    # if step == timesteps-1: # When using .
                        current_frame_token = sampled_ids
                    else:
                        current_frame_token = torch.where(
                            masking, 
                            mask_token_id,
                            sampled_ids) # shape like [1, 448]
                    
                    prompt[prompt_mask] = current_frame_token.squeeze(0)


                # NOTE: We comment this because we want the uncond history to be complete masks.
                ### Update KV cache for the newest predicted frame (after running all timesteps)
                # if guidance_scale is not None and guidance_scale != 1:
                #     ### Set uncond_prompt 
                #     # NOTE: We do not need to set uncond_prompt_mask, uncond_attention_mask since they should not be changed within timesteps_list.
                #     uncond_prompt = self.uncond_prompt_from_prompt(prompt, attention_mask, video_start_token_id, mask_token_id, only_prompt=True)
                
                # cond prompt and attention mask
                last_image_start = torch.where(prompt[0]==image_start_token_id)[-1][-1]
                prompt_for_cache = prompt[:, last_image_start:]
                # Mask partial prompt_for_cache so that we only observe partial history.
                prompt_for_cache_partial, _ = self.mask_partial_histroy(prompt_for_cache, mask_token_id, mask_pattern)
                attention_mask_for_cache = attention_mask[:,:,last_image_start:,:]
                position_ids_cache = position_ids_last_image
                # Update cache for the previous whole frame
                cond_output = self.model(
                    input_ids=prompt_for_cache_partial,
                    attention_mask=attention_mask_for_cache,
                    past_key_values=past_kv,  
                    use_cache=use_kv_cache,   
                    position_ids=position_ids_cache,
                )                             
                # uncond prompt and attention mask
                uncond_last_image_start = torch.where(uncond_prompt[0]==image_start_token_id)[-1][-1]
                uncond_prompt_for_cache = uncond_prompt[:,uncond_last_image_start:]
                # Mask partial uncond_prompt_for_cache using the same mask pattern with prompt_for_cache.
                uncond_prompt_for_cache_partial, _ = self.mask_partial_histroy(uncond_prompt_for_cache, mask_token_id, mask_pattern)
                uncond_attention_mask_for_cache = uncond_attention_mask[:,:,uncond_last_image_start:,:]
                uncond_position_ids_cache = uncond_position_ids_last_image
                # Update cache for the previous whole frame
                uncond_output = self.model(
                    input_ids=uncond_prompt_for_cache_partial,
                    attention_mask=uncond_attention_mask_for_cache,
                    past_key_values=uncond_past_kv,
                    use_cache=use_kv_cache,
                    position_ids=uncond_position_ids_cache,
                )


                # NOTE update attention_mask, prompt, prompt_mask and their uncond fusion.
                if frame_idx == end_frame - 1:
                # if frame_idx == fps_duration[0]*fps_duration[1] - 1:
                    video_end = torch.ones((prompt.shape[0],1), dtype=prompt.dtype, device=prompt.device) * video_end_token_id
                    prompt = torch.cat([prompt, video_end], dim=1)
                
                print("Mask token num: ", (prompt==mask_token_id).sum().item())
                print("prompt len: ", prompt.shape[-1])
                elapsed_time = time.time() - start_time
                print(f"Time taken for generation ({len(timesteps_list)} steps): {elapsed_time:.2f} seconds")
            
            ### Prepare for decoding
            generation_result = prompt[0].tolist()
            video_start_indices = [index for index, value in enumerate(generation_result) if value == video_start_token_id]
            generation_result = generation_result[video_start_indices[-1]:]

            if len(generation_result) > 0 and generation_result[-1] == 8710:
                generation_result = generation_result[:-1]

        return self.decode_video_ids(generation_result, output_video_path)

  
    
    def frame_mask_token_append():
        pass


    def decode_video_ids(self, tokens: List[int], output_video_path: str):
        """
        Decode a list of tokens into a video.
        
        Parameters:
            - tokens: List[int], list of tokens representing video information.
            - output_video_path: str, path where the output video will be saved.
            
        Returns:
            - generated_text: str, the decoded text output.
        """
        generated_frames = []
        generation_result_processed = []
        i = 0
        fps = None
        video_start_token_id = self.item_processor.token2id(self.item_processor.video_start_token)
        video_end_token_id = self.item_processor.token2id(self.item_processor.video_end_token)
        image_start_token_id = self.item_processor.token2id(self.item_processor.image_start_token)
        image_end_token_id = self.item_processor.token2id(self.item_processor.image_end_token)

        while i < len(tokens):
            token_id = tokens[i]

            # Handle Video Start
            if token_id == video_start_token_id:
                current_frame_tokens = []

                # Extract FPS token (which is the second token after video start token)
                fps_token = tokens[i + 2]
                fps = fps_token - 8804

                # Iterate over tokens to collect frames
                for j in range(i + 3, len(tokens)):  # Start after fps token
                    frame_token_id = tokens[j]

                    if frame_token_id == image_start_token_id:
                        print("Start one frame...")
                        # Gather tokens for the frame until the end image token is reached
                        current_frame_tokens = [frame_token_id]
                        for k in range(j + 1, len(tokens)):
                            if tokens[k] != image_end_token_id:
                                current_frame_tokens.append(tokens[k])
                            else:
                                current_frame_tokens.append(tokens[k])
                                if self.visual_tokenizer == "Chameleon":
                                    # Decode the frame and add it to the frames list
                                    frame_image = self.decode_image(current_frame_tokens)
                                    generated_frames.append(frame_image)
                                    print("Decode one image.")
                                elif "Cosmos-Tokenizer" in self.visual_tokenizer:
                                    generated_frames.append(current_frame_tokens)
                                else:
                                    raise NotImplementedError(f"{self.visual_tokenizer} is not implemented.")
                                j = k  # Update index after finishing the frame
                                break
                        i = k + 1
                    elif frame_token_id == video_end_token_id:
                        i = j + 1
                        break
                    else:
                        i += 1

            else:
                generation_result_processed.append(token_id)
                i += 1
        

        if "Cosmos-Tokenizer" in self.visual_tokenizer:
            generated_frames = self.item_processor.decode_video(generated_frames, self.spatial_compression)
        generated_text = self.item_processor.tokenizer.decode(generation_result_processed)

        # Write frames into video using FFmpeg
        if len(generated_frames) == 1:
            output_img_path = output_video_path.replace(".mp4", ".jpg")
            generated_frames[0].save(output_img_path)
        elif len(generated_frames) > 1:
            print(f"Saving to local path {output_video_path}")

            # Create a temporary folder in the current working directory to save individual PNG frames
            temp_frame_folder = f'temp_frames_{random.randint(0, 999999)}' # To avoid conflict when we have multiple runs.
            os.makedirs(temp_frame_folder, exist_ok=True)

            # Save each frame as a PNG file
            for frame_index, frame in enumerate(generated_frames):
                frame_path = os.path.join(temp_frame_folder, f'frame_{frame_index:04d}.png')
                frame.save(frame_path, format='PNG')

            # Define FPS if not provided
            if fps is None:
                fps = 25  # Fallback to default if fps is not available
                print(f"FPS is not set, thus set to {fps} by default.")

            # Use FFmpeg to create video from PNG frames
            self.create_video_with_ffmpeg_from_pngs(temp_frame_folder, output_video_path, fps)

            # Clean up the temporary frame folder
            os.system(f'rm -rf {temp_frame_folder}')
        else:
            raise ValueError("Not enough frames in generated_frames to decode.")
        

        return generated_text

    

    def create_video_with_ffmpeg_from_pngs(self, folder_path: str, output_video_path: str, fps: int):
        # Use ffmpeg to create video from PNG frames
        cmd = f'ffmpeg -y -f image2 -framerate {fps} -i {folder_path}/frame_%04d.png -vcodec libx264 -crf 17 -pix_fmt yuv420p "{output_video_path}"'
        os.system(cmd)
        print(f"Video saved at: {output_video_path}")


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
        
        ### Original
        # cfg_processor = LLMImageStartTriggeredUnbatchedClassifierFreeGuidanceLogitsProcessor(
        #     guidance_scale=cfg,
        #     model=self.model,
        #     image_start_token_id=self.item_processor.token2id(self.item_processor.image_start_token),
        #     image_end_token_id=self.item_processor.token2id(self.item_processor.image_end_token),
        #     image_next_line_token_id=self.item_processor.token2id(self.item_processor.new_line_token),
        #     patch_size=32,
        # )

        # candidate_processor = MultiModalLogitsProcessor(
        #     image_start_token_id=self.item_processor.token2id(self.item_processor.image_start_token),
        #     image_end_token_id=self.item_processor.token2id(self.item_processor.image_end_token),
        #     image_next_line_token_id=self.item_processor.token2id(self.item_processor.new_line_token),
        #     patch_size=32,
        #     voc_size=self.model.config.vocab_size,
        # )

        # topk_processor = InterleavedTopKLogitsWarper(
        #     image_top_k=image_top_k,
        #     text_top_k=text_top_k,
        #     image_start_token_id=self.item_processor.token2id(self.item_processor.image_start_token),
        #     image_end_token_id=self.item_processor.token2id(self.item_processor.image_end_token),
        # )

        ### New
        cfg_processor = LLMVideoStartTriggeredUnbatchedClassifierFreeGuidanceLogitsProcessor(
            guidance_scale=cfg,
            model=self.model,
            image_start_token_id=self.item_processor.token2id(self.item_processor.image_start_token),
            image_end_token_id=self.item_processor.token2id(self.item_processor.image_end_token),
            image_next_line_token_id=self.item_processor.token2id(self.item_processor.new_line_token),
            video_start_token_id=self.item_processor.token2id(self.item_processor.video_start_token),
            video_end_token_id=self.item_processor.token2id(self.item_processor.video_end_token),
            patch_size=32,
        )

        candidate_processor = VideoLogitsProcessor(
            image_start_token_id=self.item_processor.token2id(self.item_processor.image_start_token),
            image_end_token_id=self.item_processor.token2id(self.item_processor.image_end_token),
            image_next_line_token_id=self.item_processor.token2id(self.item_processor.new_line_token),
            video_start_token_id=self.item_processor.token2id(self.item_processor.video_start_token),
            video_end_token_id=self.item_processor.token2id(self.item_processor.video_end_token),
            patch_size=32,
            voc_size=self.model.config.vocab_size,
            visual_tokenizer=self.visual_tokenizer,
        )

        topk_processor = InterleavedTopKLogitsWarper(
            image_top_k=image_top_k,
            text_top_k=text_top_k,
            image_start_token_id=self.item_processor.token2id(self.item_processor.video_start_token),
            image_end_token_id=self.item_processor.token2id(self.item_processor.video_end_token),
        )

        logits_processor.append(cfg_processor)
        logits_processor.append(candidate_processor)
        logits_processor.append(topk_processor)

        return logits_processor
    
    def uncond_prompt_from_prompt(self, prompt, attention_mask, video_start_token_id, mask_token_id, only_prompt=False):
        # Extract uncond_prompt by dropping tokens before the video_start_token
        video_start_index = (prompt == video_start_token_id).nonzero(as_tuple=True)[1][0]

        # Extract uncond_prompt by slicing from the video_start_token onwards
        uncond_prompt = prompt[:, video_start_index:]
        
        if only_prompt:
            return uncond_prompt
        else:
            # Extract uncond_prompt_mask by identifying the masked tokens, where the tokens are still not generated.
            uncond_prompt_mask = uncond_prompt == mask_token_id

            # Obtain attention masks for uncondition generation
            uncond_attention_mask = attention_mask[:,:,video_start_index:,video_start_index:]

            return uncond_prompt, uncond_prompt_mask, uncond_attention_mask
    
    ### Random history mask
    # def mask_history_pattern(self, history_length, device, ratio=0.3):
    #     # Calculate the number of elements to set to mask_token
    #     num_to_mask = int(ratio * history_length)

    #     # Randomly select 60% of the indices to modify
    #     mask_indices = torch.randperm(history_length, device=device)[:num_to_mask]

    #     # Print for alert
    #     print(f"Mask history ratio: {ratio}")

    #     return mask_indices
    
    ### Periodic history mask
    def mask_history_pattern(self, 
                             history_length,
                             device,
                             ratio: float = 0.3,
                             *,
                             random_offset: bool = False) -> torch.LongTensor:
        """
        Return indices that are (almost) equally spaced.

        Args
        ----
        history_length : total number of tokens in the history
        device         : CUDA / CPU device where the tensor should live
        ratio          : fraction of tokens to mask
        random_offset  : if True, start at a random position inside the first interval
                        to avoid always masking the same indices

        Returns
        -------
        mask_idx : 1-D LongTensor containing `num_to_mask` indices
        """
        num_to_mask = max(1, int(round(ratio * history_length)))

        # ideal distance between two masked tokens (may be fractional)
        step = history_length / num_to_mask

        # optional shift so you don’t always mask index 0
        if random_offset:
            offset = torch.randint(0, int(step), (1,), device=device)
        else:
            offset = torch.zeros(1, dtype=torch.long, device=device)

        # linspace gives evenly-spaced *floats* – round to nearest int
        mask_idx = torch.round(offset + torch.arange(num_to_mask, device=device) * step).long()\
                    .clamp_(0, history_length - 1)\
                    .unique()         # just in case rounding produced duplicates

        # If rounding caused us to lose some positions, pad the tail
        while mask_idx.numel() < num_to_mask:
            extra = torch.tensor([history_length - 1 - len(mask_idx)],
                                device=device)
            mask_idx = torch.cat([mask_idx, extra])

        return mask_idx
    
    def mask_partial_histroy(self, history, mask_token_id, mask_pattern = None):
        history_clone = history.clone()
        history_content = history_clone[:,3:-1] # eliminate image_start_token, h_grid_token, w_grid_token and image_end_token
        if mask_pattern is None:
            mask_pattern = self.mask_history_pattern(history_content.shape[-1], device=history.device) # .unsqueeze(dim=0)
        history_content[:,mask_pattern] = mask_token_id
        history_clone[:,3:-1] = history_content
        return history_clone, mask_pattern

    
    def generate_mmrope_position_ids(self, prompt):
        cache_position = torch.arange(0, prompt.shape[1], device=prompt.device)
        position_ids = cache_position.unsqueeze(0)
        position_ids_3d, image_content_flag = self.model.model.generate_mm_position_ids(video_sequence = prompt)
        position_ids_text = position_ids.expand_as(position_ids_3d[0]) # expand the batch_size dimension.
        position_ids_3d = self.model.model.set_3d_global_offset(position_ids_3d, position_ids, image_content_flag)
        position_ids = (position_ids_text, ) + position_ids_3d + (image_content_flag, )

        return position_ids
    
    def crop_mmrope_position_ids(self, position_ids, crop_start):
        position_ids_new = [pi[:,crop_start:] for pi in position_ids]
        return tuple(position_ids_new)
        

if __name__ == "__main__":
    # parser = FlexARInferenceSolver.get_args_parser()
    # args = parser.parse_args()
    # solver = FlexARInferenceSolver(**vars(args))

    parser = FlexARVidInferenceSolver.get_args_parser()
    args = parser.parse_args()
    solver = FlexARVidInferenceSolver(**vars(args))