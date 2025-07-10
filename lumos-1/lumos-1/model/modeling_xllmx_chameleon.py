import functools
import logging
import math
from typing import List
import numpy as np

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from .chameleon import ChameleonForConditionalGeneration, ChameleonMMRoPEForConditionalGeneration
from .configuration_xllmx_chameleon import ChameleonXLLMXConfig
from model.utils import create_in_media_segment_mask, mask_or_random_replace_tokens_video, \
                    extract_image_video_masks, create_attention_mask_t2v, create_in_media_segment_mask_list, \
                    define_loss, FocalLoss, CrossEntropyFrameDecay, chunked_logsumexp_stable

logger = logging.getLogger(__name__)

default_linear_init = functools.partial(nn.init.kaiming_uniform_, a=math.sqrt(5))


__all__ = ["ChameleonXLLMXForConditionalGeneration"]



### V1: The Chameleon function that supports attention_mask specification
# class ChameleonXLLMXForConditionalGeneration(ChameleonForConditionalGeneration):
class ChameleonXLLMXForConditionalGeneration(ChameleonMMRoPEForConditionalGeneration): # V2: Enable MM-RoPE
    config_class = ChameleonXLLMXConfig

    def __init__(self, config):
        super().__init__(config)
    
    def forward(self, input_ids=None, labels=None, training=True, **kwargs):
        """
        Main forward method that chooses which specific forward pass to use.
        
        Args:
            input_ids (List[List[int]]): Input token IDs.
            labels (List[List[int]]): Label token IDs.
            training (bool): Whether it is training or not.
            forward_type (str): Type of forward function to use. Default is 'default'.
            **kwargs: Additional arguments passed to the forward function.

        Returns:
            Varies based on the forward function used.
        """

        if kwargs.get('cfg').MaskedAR:
            forward_type = 'forward_mask'
        else:
            forward_type = 'forward_default'
        
        forward_method = getattr(self, forward_type)
        return forward_method(input_ids, labels, training, **kwargs)
        

    def forward_default(self, input_ids=None, labels=None, training=True, **kwargs):

        max_tokens = max([len(_) for _ in input_ids])
        max_tokens = min(max_tokens, self.config.max_position_embeddings)
        input_ids = [_[:max_tokens] for _ in input_ids]
        labels = [_[:max_tokens] for _ in labels]

        input_ids = [example + [0] * (max_tokens - len(example)) for example in input_ids]

        # input_ids = [[10000 if num > 65536 else num for num in sublist] for sublist in input_ids]  ## for debugging
        input_ids = torch.tensor(input_ids, dtype=torch.int64, device=self.device)

        labels = [label + [-100] * (max_tokens - len(label)) for label in labels]
        # labels = [[10000 if num > 65536 else num for num in sublist] for sublist in labels]  ## for debugging
        labels = torch.tensor(labels, dtype=torch.int64, device=self.device)

        # explicit use_cache=False for the following
        # https://github.com/Lightning-AI/pytorch-lightning/issues/19267
        forward_kwargs = {}
        result = ChameleonForConditionalGeneration.forward(
            self, input_ids=input_ids, labels=labels, use_cache=False, **forward_kwargs
        )

        c_loss = result[0]
        
        additional_loss_dict = {}
        if self.config.z_loss_weight > 0:
            logits: torch.Tensor = result[1]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            valid_mask = shift_labels >= 0
            z_loss = torch.logsumexp(shift_logits, dim=-1).pow(2)[valid_mask].mean()
            additional_loss_dict["z_loss"] = (z_loss, self.config.z_loss_weight)
        return c_loss, additional_loss_dict
    

    ### V2: Separate loss calculation to a function for clarity
    def forward_mask(self, input_ids=None, labels=None, training=True, **kwargs):
        
        max_tokens = max([len(_) for _ in input_ids])
        max_tokens = min(max_tokens, self.config.max_position_embeddings)
        input_ids = [_[:max_tokens] for _ in input_ids]
        labels = [_[:max_tokens] for _ in labels]

        input_ids = [example + [0] * (max_tokens - len(example)) for example in input_ids]
        input_ids = torch.tensor(input_ids, dtype=torch.int64, device=self.device)

        labels = [label + [-100] * (max_tokens - len(label)) for label in labels]
        labels = torch.tensor(labels, dtype=torch.int64, device=self.device)

        ### Mask for creating MaskGIT-like input and models
        input_ids, labels, mar_loss_weight, mask_prob = mask_or_random_replace_tokens_video(
            input_ids=input_ids, labels=labels, mask_id=kwargs['special_token']['mask_token'], 
            config=kwargs['cfg'], mask_schedule=kwargs['mask_schedule'], special_token=kwargs['special_token'], is_train=True
        )

        ### Attention masks for bi-directional transformer
        attention_mask = create_attention_mask_t2v(
            sequence=input_ids, special_token=kwargs['special_token'], pad_id=0, 
            rm_pad_in_image=False, return_inverse_mask=True
        )
        
        # NOTE 1 explicit use_cache=False for the following
        # https://github.com/Lightning-AI/pytorch-lightning/issues/19267
        # NOTE 2 We do not use "labels" here because we will calculate loss out of the forward function.
        forward_kwargs = {'attention_mask': attention_mask}
        result = ChameleonForConditionalGeneration.forward(
            self, input_ids=input_ids, use_cache=False, **forward_kwargs
        )
        logits = result[0]

        ### Calculate closs using the new helper function if labels are provided
        closs, additional_loss_dict = None, {}
        if labels is not None:
            kwargs['mar_loss_weight'] = mar_loss_weight
            closs, additional_loss_dict = self.calculate_closs_ntp_mar(logits, labels, **kwargs)

        return closs, additional_loss_dict


    ### V3: Improve the memory usage.
    def calculate_closs_ntp_mar(self, logits, labels, **kwargs):
        """
        Calculate the combined loss (closs) from logits and labels.

        Args:
            logits (torch.Tensor): The predicted logits of shape [N, L, vocab_size].
            labels (torch.Tensor): The true labels of shape [N, L].
            kwargs (dict): Additional keyword arguments, including special tokens.

        Returns:
            torch.Tensor: The average loss (closs).
            dict: Additional loss dictionary.
        """
        batch_size, seq_len = labels.shape
        closs = None
        if labels is not None:
            # NOTE we do not need to shift the token list since masked AR do not use next token prediction.
            logits = logits.contiguous()
            labels = labels.contiguous()
            labels = labels.to(logits.device)

            ### Extract video_mask, image_mask and image_content_token_mask
            video_mask, image_mask, image_content_token_mask, has_complete_video = extract_image_video_masks(kwargs['special_token'], labels)

            ### Calculate next-token prediction loss
            if kwargs['cfg'].compute_ntp_loss:
                assert False, "Please test CEChunked in the next line: loss_fct_ce = define_loss('CE')."
                loss_fct_ce = define_loss('CE')
                ntp_loss =   torch.empty([0], dtype=logits.dtype, device=logits.device)
                ntp_labels = torch.empty([0], dtype=labels.dtype, device=logits.device)
                ntp_logits = torch.empty([0,  logits.shape[-1]], dtype=logits.dtype, device=logits.device)

                if has_complete_video.any():
                    video_mask_complete = video_mask[has_complete_video]
                    image_mask_complete = image_mask[has_complete_video]
                    assert video_mask_complete.any() and image_mask_complete.any(), "Data might not contain valid videos or video frames."
                    print(f"video_mask_complete {video_mask_complete.shape}")
                    print(f"image_mask_complete {image_mask_complete.shape}")
                    print(f"has_complete_video {has_complete_video.shape}")

                    # TODO Handle scenarios when we have multiple videos by using torch.where().
                    video_start_positions = torch.argmax(video_mask_complete.int(), dim=1) # Locate the index of video_start_token
                    image_start_positions = torch.argmax(image_mask_complete.int(), dim=1) # Locate the index of image_start_token

                    # NOTE Since we use CFG in training, then ntp_start=0. lo[ns - 1 : ne - 1] will fail if ns-1=-1.
                    # We MUST set ntp_start to be 1 in CFG training.
                    ntp_start = video_start_positions
                    ntp_start = torch.clamp(ntp_start, min=1) 
                    ntp_end = image_start_positions + 3
                    
                    # # V1: costs a lot of memory
                    # logits_complete = logits[has_complete_video]
                    # labels_complete = labels[has_complete_video]
                    # ntp_logits = torch.cat([lo[ns - 1 : ne - 1] for ns, ne, lo in zip(ntp_start, ntp_end, logits_complete)])
                    # ntp_labels = torch.cat([la[ns : ne] for ns, ne, la in zip(ntp_start, ntp_end, labels_complete)])
                    # ntp_loss = loss_fct_ce(ntp_logits, ntp_labels) 
                    
                    # V2: Improve memory usage.
                    for ns, ne, lo, la, hcv in zip(ntp_start, ntp_end, logits, labels, has_complete_video):
                        if hcv:
                            ntp_logits = torch.cat([ntp_logits, lo[ns - 1 : ne - 1]])
                            ntp_labels = torch.cat([ntp_labels, la[ns : ne]])
                    ntp_loss = loss_fct_ce(ntp_logits, ntp_labels)


            ### Calculate Masked AR loss            
            # The following is the improved version for saving memory.
            loss_fct_mar = define_loss(
                loss_type = kwargs['cfg'].train_loss, start_coef = kwargs['cfg'].decay_start_coef,
                special_token = kwargs['special_token'], device = logits.device,
                frame_num = (labels[0]==kwargs['special_token']['image_start_token']).sum()
            )
            if kwargs['cfg'].train_loss == "CEDecay":
                mar_loss = loss_fct_mar(logits, labels)
            elif kwargs['cfg'].train_loss in ["CE", "Focal", "CEChunked"]:
                if kwargs['cfg'].train_with_vis_tok:
                    vis_tok_start = kwargs['cfg'].vis_tok_start # 65536 for Cosmos
                    # Eliminate non-visual tokens: [bs * seq_len, 129536] -> [bs * visual_seq_len, 129536]
                    visual_logits = logits.flatten(0,1)[labels.flatten(0,1) >= vis_tok_start] # [bs * visual_seq_len, 129536]
                    # Eliminate non-visual logits (extract visual logits): [bs * visual_seq_len, 129536] -> [bs * visual_seq_len, 64000]
                    visual_logits = visual_logits[..., vis_tok_start:]
                    # Eliminate non-visual labels (extract visual labels and offset it)
                    visual_labels = labels.flatten(0,1)[labels.flatten(0,1) >= vis_tok_start] - vis_tok_start
                    # Loss calculation
                    mar_loss = loss_fct_mar(visual_logits, visual_labels)
                else:
                    # V1: Use all logits
                    mar_loss = loss_fct_mar(logits.flatten(0,1), labels.flatten(0,1))[image_content_token_mask.flatten()]
                    # print((labels[image_content_token_mask]).max(), (labels[image_content_token_mask]).min(), (labels[image_content_token_mask]==-100).sum())
            else:
                # assert False, "Not implemented."
                raise NotImplementedError(f"{kwargs['cfg'].train_loss} is not supported yet.")
            

            if 'mar_loss_weight' in kwargs and kwargs['mar_loss_weight'] is not None:
                # TODO This needs to be tested
                assert False, "mar_loss_weight needs to be tested."
                loss_weight = torch.cat(loss_weight) # .view(-1)
                # mar_loss = ((mar_loss * loss_weight).sum(dim=-1) / loss_weight.sum(dim=-1)).mean()
                mar_loss = mar_loss * loss_weight # this might work

            ### Cat two losses and average them
            all_closs = torch.cat([ntp_loss, mar_loss]) if kwargs['cfg'].compute_ntp_loss else mar_loss
            all_closs = all_closs[all_closs != 0] # NOTE This is to exclude cases when the label is -100 and cross_entropy ouputs 0 for these labels.
            closs = all_closs.mean()
      
        additional_loss_dict = {}
        if self.config.z_loss_weight > 0:

            ### V4: Improve memory usage by chunking. NOTE: tested. Results are identical with the original version.
            valid_mask = visual_labels >= 0 if kwargs['cfg'].train_with_vis_tok else labels >= 0
            # chunked stable log-sum-exp => [B, T]
            lse = chunked_logsumexp_stable(visual_logits, chunk_size=2000) if kwargs['cfg'].train_with_vis_tok else chunked_logsumexp_stable(logits, chunk_size=4096)
            # square, then filter by valid_mask => [num_valid_positions]
            z_loss = lse.pow(2)[valid_mask].mean()
            additional_loss_dict["z_loss"] = (z_loss, self.config.z_loss_weight)


        if kwargs['cfg'].frame_closs_recorder:

            ### V2
            additional_loss_dict = self.calculate_frame_wise_closs(
                mar_loss, labels, batch_size, kwargs['special_token'], vis_tok_start, 
                kwargs['cfg'].mask_type, additional_loss_dict
            )
            

        return closs, additional_loss_dict
    

    ### V2: Frame-wise loss calculation, fixing the bug for no considering the first frame.
    def calculate_frame_wise_closs(
        self, 
        mar_loss: torch.Tensor, 
        labels: torch.Tensor, 
        batch_size: int, 
        special_token: dict, 
        vis_tok_start: int, 
        mask_type: str, 
        additional_loss_dict: dict
    ) -> dict:
        """
        Calculates the frame-wise conditional loss for each frame in the batch.
        This is a rough estimation of the frame loss since frame boundaries 
        are not explicitly calculated. Loss is distributed among frames based 
        on visual tokens present in the labels.

        Args:
            mar_loss (torch.Tensor): The flattened loss for all tokens in the batch.
            labels (torch.Tensor): The tensor containing token labels (shape: [batch_size, sequence_length]).
            batch_size (int): The number of sequences in the batch.
            special_token (dict): Dictionary containing special tokens (e.g., `image_start_token`, `image_end_token`).
            vis_tok_start (int): The starting token index for visual tokens.
            mask_type (str): Specifies the masking type, either 'random' or 'tube'.
            additional_loss_dict (dict): Dictionary to store additional loss values.

        Returns:
            dict: Updated dictionary with frame-wise conditional loss values.
        """
        assert mask_type in ["random", "tube", "DiffForcing"], "mask_type must be 'random' or 'tube' to make this estimation roughly accurate."
        # NOTE: The frame loss is not accurate when using "DiffForcing" because the frame boundaries are not explicitly calculated.
        #       But since every frame has different masked token, it might not be that important to calculate loss for every frame?

        # Extract special tokens
        image_start_token = special_token['image_start_token']
        image_end_token = special_token['image_end_token']

        # Reshape labels to (batch_size, sequence_length)
        labels = labels.view(batch_size, -1)

        # Calculate the number of visual tokens for each sequence
        vis_tok_nums = [(l >= vis_tok_start).sum().item() for l in labels]

        # Cumulative sum of visual token counts to determine boundaries
        cul_vis_tok_nums = np.cumsum(vis_tok_nums) 
        cul_vis_tok_nums = np.append(np.array([0]), cul_vis_tok_nums) # Add the start token to obtain: [0, xx, 2*xx, 3*xx, ...]

        # Rough estimation of the number of frames (based on `image_start_token`)
        frame_num = (labels[0] == image_start_token).sum().item()

        # Initialize a list to store losses for each frame
        frame_losses = [[] for _ in range(frame_num)]

        # Split the loss for each sequence into frame-specific chunks
        # print(cul_vis_tok_nums)
        for i in range(batch_size):
            batch_i_loss = mar_loss[cul_vis_tok_nums[i] : cul_vis_tok_nums[i + 1]]
            batch_i_frame_losses = batch_i_loss.chunk(frame_num)

            # Append frame losses for the current batch
            for frame_idx, batch_i_frame_loss in enumerate(batch_i_frame_losses):
                frame_losses[frame_idx].append(batch_i_frame_loss)

        # Compute the mean loss for each frame across batches
        frame_losses = [torch.cat(frame_loss).mean() for frame_loss in frame_losses]

        # Add frame-wise loss to the additional loss dictionary
        for frame_idx, frame_loss in enumerate(frame_losses):
            loss_name = f'tmp_closs_CE_frame{frame_idx}'
            additional_loss_dict[loss_name] = (frame_loss, 0.)

        return additional_loss_dict


    def get_fsdp_wrap_module_list(self) -> List:
        modules = [*list(self.model.layers), self.lm_head, self.model.embed_tokens]
        if hasattr(self.model, "vqmodel"):  # may be deleted
            modules.append(self.model.vqmodel)
        return modules

    def get_checkpointing_wrap_module_list(self) -> List:
        modules = [
            *list(self.model.layers),
        ]
        return modules



class PartialLinear(nn.Module):
    def __init__(
        self, 
        original_linear: nn.Linear, 
        subset_start: int, 
        subset_end: int
    ):
        """
        Initializes the PartialLinear layer by referencing a subset of weights from the original Linear layer.

        Args:
            original_linear (nn.Linear): The original linear layer (lm_head) to reference weights from.
            subset_start (int): The starting index of the vocabulary subset.
            subset_end (int): The ending index (exclusive) of the vocabulary subset.
        """
        super(PartialLinear, self).__init__()
        assert subset_start >= 0 and subset_end <= original_linear.out_features, "Subset indices out of range."
        assert subset_start < subset_end, "subset_start must be less than subset_end."

        self.original_linear = original_linear
        self.subset_start = subset_start
        self.subset_end = subset_end
        self.subset_size = subset_end - subset_start

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the PartialLinear layer.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, in_features].

        Returns:
            torch.Tensor: 
                - During training: Logits tensor of shape [batch_size, seq_len, subset_size].
                - During inference: Logits tensor of shape [batch_size, seq_len, original_vocab_size].
        """
        if self.training:
            # Compute only subset of logits
            weight_subset = self.original_linear.weight[self.subset_start:self.subset_end, :]  # [subset_size, in_features]
            logits_subset = torch.matmul(x, weight_subset.t())  # [batch_size, seq_len, subset_size]
            if self.original_linear.bias is not None:
                bias_subset = self.original_linear.bias[self.subset_start:self.subset_end]  # [subset_size]
                logits_subset += bias_subset  # Broadcasting adds bias
            return logits_subset
        else:
            # Compute full logits during inference
            return self.original_linear(x)
