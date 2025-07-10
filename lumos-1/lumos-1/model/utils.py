####################################################################################################
#      Partially copy paste from https://github.com/showlab/Show-o/blob/main/training/utils.py
#      and https://github.com/showlab/Show-o/blob/main/training/prompting_utils.py
####################################################################################################

import math
import random
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig, OmegaConf
from typing import Any, List, Tuple, Union
import os
import json


#####################################################################################
#                                 config utils                                      #
#####################################################################################
def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)

    return conf


def flatten_omega_conf(cfg: Any, resolve: bool = False) -> List[Tuple[str, Any]]:
    ret = []

    def handle_dict(key: Any, value: Any, resolve: bool) -> List[Tuple[str, Any]]:
        return [(f"{key}.{k1}", v1) for k1, v1 in flatten_omega_conf(value, resolve=resolve)]

    def handle_list(key: Any, value: Any, resolve: bool) -> List[Tuple[str, Any]]:
        return [(f"{key}.{idx}", v1) for idx, v1 in flatten_omega_conf(value, resolve=resolve)]

    if isinstance(cfg, DictConfig):
        for k, v in cfg.items_ex(resolve=resolve):
            if isinstance(v, DictConfig):
                ret.extend(handle_dict(k, v, resolve=resolve))
            elif isinstance(v, ListConfig):
                ret.extend(handle_list(k, v, resolve=resolve))
            else:
                ret.append((str(k), v))
    elif isinstance(cfg, ListConfig):
        for idx, v in enumerate(cfg._iter_ex(resolve=resolve)):
            if isinstance(v, DictConfig):
                ret.extend(handle_dict(idx, v, resolve=resolve))
            elif isinstance(v, ListConfig):
                ret.extend(handle_list(idx, v, resolve=resolve))
            else:
                ret.append((str(idx), v))
    else:
        assert False

    return ret


#####################################################################################
#                                  training utils                                   #
#####################################################################################
def soft_target_cross_entropy(logits, targets, soft_targets):
    # ignore the first token from logits and targets (class id token)
    logits = logits[:, 1:]
    targets = targets[:, 1:]

    logits = logits[..., : soft_targets.shape[-1]]

    log_probs = F.log_softmax(logits, dim=-1)
    padding_mask = targets.eq(-100)

    loss = torch.sum(-soft_targets * log_probs, dim=-1)
    loss.masked_fill_(padding_mask, 0.0)

    # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
    num_active_elements = padding_mask.numel() - padding_mask.long().sum()
    loss = loss.sum() / num_active_elements
    return loss


def get_loss_weight(t, mask, min_val=0.3):
    return 1 - (1 - mask) * ((1 - t) * (1 - min_val))[:, None]


def mask_or_random_replace_tokens(image_tokens, mask_id, config, mask_schedule, is_train=True):
    """
    Mask or randomly replace tokens in the input image token sequences.

    Args:
        image_tokens: Tensor of image tokens, shape (batch_size, seq_len).
        mask_id: The ID used for masking tokens.
        config: Configuration object containing training settings.
        mask_schedule: Function that defines the masking schedule.
        is_train: Boolean indicating if the function is used for training.

    Returns:
        input_ids: Modified input tokens with masking or random replacement.
        labels: Labels for computing loss.
        loss_weight: Optional loss weight tensor.
        mask_prob: The probability of masking each token.
    """
    batch_size, seq_len = image_tokens.shape

    # If not training, use predefined mask ratios for evaluation if available
    if not is_train and config.training.get("eval_mask_ratios", None):
        mask_prob = random.choices(config.training.eval_mask_ratios, k=batch_size)
        mask_prob = torch.tensor(mask_prob, device=image_tokens.device)
    else:
        # Sample a random timestep for each image
        timesteps = torch.rand(batch_size, device=image_tokens.device)
        # Sample a random mask probability for each image using timestep and cosine schedule
        mask_prob = mask_schedule(timesteps)
        # Clip the mask probability to ensure it's at least the minimum masking rate
        mask_prob = mask_prob.clip(config.training.min_masking_rate)

    # creat a random mask for each image
    num_token_masked = (seq_len * mask_prob).round().clamp(min=1)

    # Determine if we should mask contiguous regions of tokens
    mask_contiguous_region_prob = config.training.get("mask_contiguous_region_prob", None)

    if mask_contiguous_region_prob is None:
        mask_contiguous_region = False
    else:
        mask_contiguous_region = random.random() < mask_contiguous_region_prob

    if not mask_contiguous_region:
        batch_randperm = torch.rand(batch_size, seq_len, device=image_tokens.device).argsort(dim=-1)
        mask = batch_randperm < num_token_masked.unsqueeze(-1)
    else:
        # Mask contiguous regions by reshaping the tokens into a 2D resolution (e.g., square grid)
        resolution = int(seq_len ** 0.5)
        mask = torch.zeros((batch_size, resolution, resolution), device=image_tokens.device)

        # TODO - would be nice to vectorize
        for batch_idx, num_token_masked_ in enumerate(num_token_masked):
            num_token_masked_ = int(num_token_masked_.item())

            # NOTE: a bit handwavy with the bounds but gets a rectangle of ~num_token_masked_
            num_token_masked_height = random.randint(
                math.ceil(num_token_masked_ / resolution), min(resolution, num_token_masked_)
            )
            num_token_masked_height = min(num_token_masked_height, resolution)

            num_token_masked_width = math.ceil(num_token_masked_ / num_token_masked_height)
            num_token_masked_width = min(num_token_masked_width, resolution)

            start_idx_height = random.randint(0, resolution - num_token_masked_height)
            start_idx_width = random.randint(0, resolution - num_token_masked_width)

            mask[
            batch_idx,
            start_idx_height: start_idx_height + num_token_masked_height,
            start_idx_width: start_idx_width + num_token_masked_width,
            ] = 1

        mask = mask.reshape(batch_size, seq_len)
        mask = mask.to(torch.bool)

    # mask images and create input and labels
    if config.training.get("noise_type", "mask"):
        # Replace masked tokens with the mask ID
        input_ids = torch.where(mask, mask_id, image_tokens)
    elif config.training.get("noise_type", "random_replace"):
        # Replace masked tokens with randomly sampled tokens from the vocabulary
        random_tokens = torch.randint_like(
            image_tokens, low=0, high=config.model.codebook_size, device=image_tokens.device
        )
        input_ids = torch.where(mask, random_tokens, image_tokens)
    else:
        raise ValueError(f"noise_type {config.training.noise_type} not supported")

    # Prepare labels for training
    if (
            config.training.get("predict_all_tokens", False)
            or config.training.get("noise_type", "mask") == "random_replace"
    ):  
        # If predicting all tokens or using random replacement, use all original tokens as labels
        labels = image_tokens
        # Compute the loss weight based on mask probability
        loss_weight = get_loss_weight(mask_prob, mask.long())
    else:
        # Only predict masked tokens, use -100 to ignore other tokens in the loss computation
        labels = torch.where(mask, image_tokens, -100)
        loss_weight = None

    return input_ids, labels, loss_weight, mask_prob



def mask_or_random_replace_tokens_video(input_ids, labels, mask_id, config, mask_schedule, special_token, is_train=True):
    """
    Mask or randomly replace tokens in the input image token sequences.

    Args:
        input_ids: a tensor containing the list of token lists, where each list represents tokens for a specific example.
        labels: a tensor containing the list of token labels, with tokens unnecessary for training set to -100.
        mask_id: The ID used for masking tokens.
        config: Configuration object containing training settings.
        mask_schedule: Function that defines the masking schedule.
        is_train: Boolean indicating if the function is used for training.

    Returns:
        input_ids: Modified input tokens with masking or random replacement.
        labels: Labels for computing loss.
        loss_weight: Optional loss weight tensor.
        mask_prob: The probability of masking each token.
    """
    # batch_size, seq_len = input_ids.shape
    batch_size, _ = input_ids.shape
    _, image_mask, image_content_token_mask, _ = extract_image_video_masks(special_token, labels)
    _, _, full_image_content_token_mask, _     = extract_image_video_masks(special_token, input_ids)
    seq_len_list = [i.sum() for i in image_content_token_mask]

    # If not training, use predefined mask ratios for evaluation if available
    if not is_train and config.get("eval_mask_ratios", None):
        assert False, "Not implemented"
        # TODO Hangjie: This needs to be modified if we want to make it work in our framework.
        mask_prob = random.choices(config.eval_mask_ratios, k=batch_size)
        mask_prob = torch.tensor(mask_prob, device=input_ids.device)
    else:
        # Sample a random timestep for each image
        timesteps = torch.rand(batch_size, device=input_ids.device)
        # Sample a random mask probability for each image using timestep and cosine schedule
        mask_prob = mask_schedule(timesteps)
        # Clip the mask probability to ensure it's at least the minimum masking rate
        mask_prob = mask_prob.clip(config.min_masking_rate)
        # print('mask_prob ', mask_prob)

    # creat a random mask for each image
    num_token_masked_list = [(seq_len * mask_prob[s_idx]).round().clamp(min=1) for s_idx, seq_len in enumerate(seq_len_list)]

    # Determine if we should mask contiguous regions of tokens
    mask_contiguous_region_prob = config.mask_contiguous_region_prob

    if mask_contiguous_region_prob is None:
        mask_contiguous_region = False
    else:
        mask_contiguous_region = random.random() < mask_contiguous_region_prob

    if not mask_contiguous_region:
        if config.mask_type == 'tubeDecay':
            image_len = [first_media_segment_number(m) for m in image_content_token_mask]
            image_content_len = [m.sum() for m in image_content_token_mask]
            image_count = [torch.ceil(m.sum()/il).int().item() for m, il in zip(image_content_token_mask, image_len)] # consider imcomplete ones
            mask_list = []
            for idx, (il, icl, ic, mp) in enumerate(zip(image_len, image_content_len, image_count, mask_prob)):
                image_num_token_masked_start = (il * mp).round().clamp(min=1)
                image_num_token_masked_end = (il * torch.tensor(1., dtype=mp.dtype)).round().clamp(min=1)
                mask_count_list = torch.linspace(image_num_token_masked_start, image_num_token_masked_end, ic)
                batch_randperm = torch.rand(1, il, device=input_ids.device).argsort(dim=-1)
                video_mask = torch.cat([batch_randperm < m for m in mask_count_list], dim=1)[:,:icl]
                mask_list.append(video_mask)

        elif config.mask_type == 'tube': ### Generate boolean masks with repetitive patterns for erery image
            ## V1: T2V with tube mask
            image_len = [first_media_segment_number(m) for m in image_content_token_mask]
            image_content_len = [m.sum() for m in image_content_token_mask]
            # Image count considering video prediction task, meaning that some images are given
            image_count = [torch.ceil(m.sum()/il).int().item() for m, il in zip(image_content_token_mask, image_len)] # consider imcomplete ones
            mask_list = []
            for idx, (il, icl, ic, mp) in enumerate(zip(image_len, image_content_len, image_count, mask_prob)):
                image_num_token_masked = (il * mp).round().clamp(min=1)
                batch_randperm = torch.rand(1, il, device=input_ids.device).argsort(dim=-1)
                mask_list.append((batch_randperm < image_num_token_masked).repeat(1, ic)[:,:icl])

        elif config.mask_type == 'DiffForcing':

            ## V1: T2V with diffusion forcing mask
            image_len = [first_media_segment_number(m) for m in image_content_token_mask]
            image_content_len = [m.sum() for m in image_content_token_mask]
            # Image count considering video prediction task, meaning that some images are given
            image_count = [torch.ceil(m.sum()/il).int().item() for m, il in zip(image_content_token_mask, image_len)] # consider imcomplete ones

            mask_list = []
            for idx, (il, icl, ic) in enumerate(zip(image_len, image_content_len, image_count)):
                # Sample a random timestep for each image
                timesteps = torch.rand((ic), device=input_ids.device)
                # Sample a random mask probability for each image using timestep and cosine schedule
                mask_prob = mask_schedule(timesteps)
                # Clip the mask probability to ensure it's at least the minimum masking rate
                mask_prob = mask_prob.clip(config.min_masking_rate)

                one_video_mask = []
                for mp in mask_prob:
                    image_num_token_masked = (il * mp).round().clamp(min=1)
                    batch_randperm = torch.rand(1, il, device=input_ids.device).argsort(dim=-1)
                    one_video_mask.append((batch_randperm < image_num_token_masked))
                
                one_video_mask = torch.cat(one_video_mask, dim=-1)[:,:icl]
                mask_list.append(one_video_mask)
            # print("Diffusion Masking.")

        elif config.mask_type == 'random':  ### Generate boolean masks without repetitive patterns for erery image
            mask_list = []
            for seq_len, num_token_masked in zip(seq_len_list, num_token_masked_list):
                batch_randperm = torch.rand(1, seq_len, device=input_ids.device).argsort(dim=-1)
                mask_list.append(batch_randperm < num_token_masked.unsqueeze(-1))

        else:
            assert False, f"Mask type {config.mask_type} is not implemented."
    else:
        assert False, "Not implemented"
        # TODO Hangjie: currently not support, must modified code to support this function.
        # Mask contiguous regions by reshaping the tokens into a 2D resolution (e.g., square grid)
        resolution = int(seq_len ** 0.5)
        mask = torch.zeros((batch_size, resolution, resolution), device=input_ids.device)

        # TODO - would be nice to vectorize
        for batch_idx, num_token_masked_ in enumerate(num_token_masked):
            num_token_masked_ = int(num_token_masked_.item())

            # NOTE: a bit handwavy with the bounds but gets a rectangle of ~num_token_masked_
            num_token_masked_height = random.randint(
                math.ceil(num_token_masked_ / resolution), min(resolution, num_token_masked_)
            )
            num_token_masked_height = min(num_token_masked_height, resolution)

            num_token_masked_width = math.ceil(num_token_masked_ / num_token_masked_height)
            num_token_masked_width = min(num_token_masked_width, resolution)

            start_idx_height = random.randint(0, resolution - num_token_masked_height)
            start_idx_width = random.randint(0, resolution - num_token_masked_width)

            mask[
            batch_idx,
            start_idx_height: start_idx_height + num_token_masked_height,
            start_idx_width: start_idx_width + num_token_masked_width,
            ] = 1

        mask = mask.reshape(batch_size, seq_len)
        mask = mask.to(torch.bool)

    # mask images and create input and labels
    if config.noise_type == "mask":
        # Replace masked tokens with the mask ID
        for idx, (mask, content_token_mask) in enumerate(zip(mask_list, image_content_token_mask)):
            content = input_ids[idx, content_token_mask]
            masked_content = torch.where(mask, mask_id, content)
            input_ids[idx, content_token_mask] = masked_content
    elif config.noise_type == "random_replace":
        assert False, "Not implemented"
        # TODO Hangjie: currently not support, must modified code to support this function.
        # Replace masked tokens with randomly sampled tokens from the vocabulary
        random_tokens = torch.randint_like(
            image_tokens, low=0, high=config.model.codebook_size, device=input_ids.device
        )
        input_ids = torch.where(mask, random_tokens, image_tokens)
    else:
        raise ValueError(f"noise_type {config.training.noise_type} not supported")


    # Prepare labels for training
    if (
            config.predict_all_tokens
            or config.noise_type == "random_replace"
    ):  
        # If predicting all tokens or using random replacement, use all original labels
        labels = labels

        # Compute the loss weight based on mask probability
        loss_weight = []
        for mask in mask_list:
            loss_weight.append(get_loss_weight(mask_prob, mask.long()))
    else:
        # Only predict masked tokens, use -100 to ignore other tokens in the loss computation
        for idx, (mask, content_token_mask) in enumerate(zip(mask_list, image_content_token_mask)):
            content_label = labels[idx, content_token_mask]
            masked_content_label = torch.where(mask, content_label, -100)
            labels[idx, content_token_mask] = masked_content_label

        loss_weight = None
    

    return input_ids, labels, loss_weight, mask_prob



def first_media_segment_number(mask):
    """
    Calculate the length of the first contiguous segment of True values in a binary mask.
    
    The function iterates through the given mask (a binary list or array-like structure of True and False values),
    and calculates the length of the first contiguous segment where the values are True. 
    Once the first segment of True ends (a False is encountered after a contiguous segment of True), 
    the function stops counting and returns the length of this first segment.

    Parameters:
    mask : array-like
        A binary mask represented as an array-like structure (such as a list or NumPy array) 
        containing boolean values (True/False). It is expected to be iterable.

    Returns:
    int
        The length of the first contiguous segment of True values in the mask. 
        If no True values exist, the function will return 0.
    """

    # Convert the mask to a list for efficient iteration
    mask_list = mask.tolist()
    first_segment_length = 0
    in_segment = False
    
    for value in mask_list:
        if value:  # if we encounter a True value
            first_segment_length += 1
            in_segment = True
        elif in_segment:  # if False is encountered after finding the first segment
            break  # Exit the loop as we have found the first segment
    
    return first_segment_length



### V2：Extract masks for video and video frames while considering the cases when partial videos are input to the model.
def extract_image_video_masks(special_token, labels, image_content_mask=True):
    """
    Extract masks for video segments, image segments, and image content tokens.

    Args:
        special_token (dict): Dictionary containing special token IDs, including:
            - 'video_start_token': Token ID indicating the start of a video segment.
            - 'video_end_token': Token ID indicating the end of a video segment.
            - 'image_start_token': Token ID indicating the start of an image segment.
            - 'image_end_token': Token ID indicating the end of an image segment.
            - 'sep_token': Token ID indicating a separator between tokens.
        labels (torch.Tensor): Tensor of shape [batch_size, sequence_length] representing the label tokens.'
        content_mask (bool): A boolean value indicating whether we need to return image_content_token_mask

    Returns:
        tuple: A tuple containing:
            - video_mask (torch.Tensor): Boolean mask indicating video segments.
            - image_mask (torch.Tensor): Boolean mask indicating image segments within video frames.
            - image_content_token_mask (torch.Tensor): Boolean mask indicating image content tokens, excluding special tokens.
    """

    ### Define special tokens
    # Extract the IDs for the different special tokens used in the sequence
    video_start_token = special_token['video_start_token']
    video_end_token = special_token['video_end_token']
    image_start_token = special_token['image_start_token']
    image_end_token = special_token['image_end_token']
    sep_token = special_token['sep_token']

    ### Create masks for video and image segments
    # Create a mask for video segments, identifying regions between the video start and end tokens
    video_mask, has_complete_video = create_in_media_segment_mask(sequence=labels, soi_id=video_start_token, eoi_id=video_end_token)

    # Create a mask for image segments, identifying regions between the image start and end tokens
    image_mask, _                  = create_in_media_segment_mask(sequence=labels, soi_id=image_start_token, eoi_id=image_end_token)

    # Refine the image mask to only include images that are within the video frames
    # (i.e., ensure that images are always a part of a video segment)
    # This line is compatible with video prediction since it might not have video start token.
    image_mask = (has_complete_video[:,None] * (image_mask & video_mask)) + ((~has_complete_video[:,None]) * image_mask)
    # image_mask = image_mask & video_mask if has_complete_video else image_mask

    image_content_token_mask = None
    if image_content_mask:
        ### Create masks for special tokens within image segments
        # Identify the positions of specific special tokens within the image segments
        sep_token_mask = (labels == sep_token) & image_mask  # Mask for separator tokens within the image segments
        image_start_token_mask = (labels == image_start_token) & image_mask  # Mask for image start tokens within image segments
        image_end_token_mask = (labels == image_end_token) & image_mask  # Mask for image end tokens within image segments

        # Create the height token mask for the image segments
        # - Prepend a column of False and then drop the last column to shift the mask to the right by 1 position.
        # - This allows you to mark the position immediately after the `image_start_token` as `h_token`.
        image_h_token_mask = torch.cat(
            [torch.zeros((image_mask.size(0), 1), dtype=torch.bool, device=labels.device), image_start_token_mask],
            dim=1
        )[:, :-1]

        # Create the width token mask for the image segments
        # - Prepend two columns of False and then drop the last two columns to shift the mask by 2 positions.
        # - This marks the position immediately after `h_token` as `w_token`.
        image_w_token_mask = torch.cat(
            [torch.zeros((image_mask.size(0), 2), dtype=torch.bool, device=labels.device), image_start_token_mask],
            dim=1
        )[:, :-2]

        ### Create the image content token mask
        # Start with the image mask and exclude all special tokens to get a mask for image content tokens
        # - Clone the image mask to avoid modifying it directly.
        # - Set positions corresponding to special tokens (e.g., sep_token, image_start/end tokens, h/w tokens) to False.
        image_content_token_mask = image_mask.clone()
        image_content_token_mask[sep_token_mask | image_start_token_mask | image_end_token_mask | image_h_token_mask | image_w_token_mask] = False
        # NOTE NOTE NOTE Currently, we do not add new_line_token here, 
        # since in "config.mask_type == 'tube'", we use first_media_segment_number() function, which by default assumes that image_content is a continous sequence.

    # Return all three masks: video_mask, image_mask, and image_content_token_mask
    return video_mask, image_mask, image_content_token_mask, has_complete_video
            


def create_in_media_segment_mask(sequence, soi_id, eoi_id):
    """
    Create a mask indicating the segments of the sequence that belong to media regions.

    Args:
        sequence (torch.Tensor): Input tensor of shape [N, L], where N is the batch size and L is the sequence length.
        soi_id (int): The token ID that indicates the start of a media segment.
        eoi_id (int): The token ID that indicates the end of a media segment.

    Returns:
        torch.Tensor: A boolean mask of shape [N, L] where True indicates tokens that are part of a media segment.
    """

    # sequence is expected to be of shape [N, L]
    N, L = sequence.shape

    # Create masks for start and end of media tokens
    is_start_media = sequence == soi_id   # True where the token is the start of a media segment
    is_end_media = sequence == eoi_id     # True where the token is the end of a media segment
    has_complete_media = is_start_media.any(-1) & is_end_media.any(-1)

    # Create cumulative sum masks to identify regions of media tokens
    cumulative_start = torch.cumsum(is_start_media, dim=1)  # Cumulative sum for start tokens
    cumulative_end = torch.cumsum(is_end_media, dim=1)      # Cumulative sum for end tokens

    # Identify segments of the sequence that contain media tokens
    # This generates a mask including start_token and end_token.
    in_media_segment = (cumulative_start > cumulative_end) | is_start_media | is_end_media

    return in_media_segment, has_complete_media


def create_in_media_segment_mask_list(sequence, soi_id, eoi_id):
    """
    Create masks indicating the segments of the sequence that belong to each media region.
    NOTE: This is different from create_in_media_segment_mask(), which creates a mask for all the media in a sequence,
          while this function aims to create separate masks for all media (e.g., 5 masks for 5 media sub-sequence).

    Args:
        sequence (torch.Tensor): Input tensor of shape [N, L], where N is the batch size and L is the sequence length.
        soi_id (int): The token ID that indicates the start of a media segment.
        eoi_id (int): The token ID that indicates the end of a media segment.

    Returns:
        list of torch.Tensor: A list of boolean masks, each of shape [N, L], where True indicates tokens that are part of a media segment.
        torch.Tensor: A boolean tensor of shape [N] indicating whether each sequence in the batch contains complete media segments.
    """

    # sequence is expected to be of shape [N, L]
    N, L = sequence.shape

    # Create masks for start and end of media tokens
    is_start_media = sequence == soi_id   # True where the token is the start of a media segment
    is_end_media = sequence == eoi_id     # True where the token is the end of a media segment
    has_complete_media = is_start_media.any(-1) & is_end_media.any(-1)

    # Create cumulative sum masks to identify regions of media tokens
    cumulative_start = torch.cumsum(is_start_media, dim=1)  # Cumulative sum for start tokens
    cumulative_end = torch.cumsum(is_end_media, dim=1)      # Cumulative sum for end tokens

    # Identify the number of media segments (assuming equal for all batches)
    num_media_segments = is_start_media.sum(dim=1).max().item()
    assert (is_start_media.sum(dim=1) == num_media_segments).all() # Assert all batches have the same number of frames. 

    # Create masks for each media segment
    segment_masks = []
    for i in range(1, num_media_segments + 1):
        # Create masks for the i-th media segment including both start and end tokens
        in_segment = (cumulative_start >= i) & (cumulative_end < i)
        in_segment |= (cumulative_start == i) & is_end_media  # Include the end token for the current segment
        segment_masks.append(in_segment)

    return segment_masks, has_complete_media







#####################################################################################
#                                    Loss utils                                     #
#####################################################################################
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='none', ignore_index=-100):
        """
        Focal Loss for addressing class imbalance in classification tasks.

        Parameters:
        - gamma (float, default=2.0): Focusing parameter that helps down-weight easy examples and focus more on hard examples.
        - alpha (Tensor or None, default=None): Class balancing factor. If None, uniform weighting is applied. If provided as a tensor, it is used as a per-class weight vector.
        - reduction (str, default='none'): Specifies the reduction method. Options are:
            'none' - no reduction, returns individual losses.
            'mean' - returns the mean of all losses.
            'sum' - returns the sum of all losses.
        - ignore_index (int, default=-100): The label value to be ignored when calculating the loss. Any sample with this label will not contribute to the loss.

        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # Now alpha can be a tensor (per-class weights)
        self.reduction = reduction
        self.ignore_index = ignore_index  # Set the ignore index

    
    def forward(self, logits, labels):
        """
        Forward pass for computing the Focal Loss.

        Parameters:
        - logits (Tensor, shape [B, C]): The raw output logits from the model, where:
            - B: Batch size.
            - C: Number of classes.
        - labels (Tensor, shape [B]): The ground truth labels for each sample in the batch, where each value is the class index (0 to C-1).
          Labels that are equal to ignore_index are excluded from the loss computation.

        Returns:
        - Tensor: The computed loss value (either per-sample or aggregated based on the reduction method).
        
        The loss is calculated as:
            - Cross-entropy loss weighted by the factor (1 - p_t) ** gamma, where p_t is the predicted probability of the true class.
            - Optionally, class weights are applied using the alpha parameter.
            - The loss for samples with labels equal to ignore_index is excluded.
        """
        # Create a mask for labels that are -100 (these labels should be ignored)
        mask = (labels != self.ignore_index)

        # Compute the probabilities via softmax
        probs = F.softmax(logits, dim=-1)

        # We mask out the invalid labels before gathering the probabilities
        # Create a tensor of valid indices
        valid_labels = torch.where(mask, labels, torch.tensor(0, dtype=labels.dtype, device=labels.device))

        # Get the probability of the correct class for each sample, while ignoring -100 labels
        p_t = probs.gather(1, valid_labels.view(-1, 1))  # shape [B, 1]
        p_t = p_t.view(-1)  # Flatten to shape [B]

        # Handle alpha tensor: if alpha is None, default to a uniform weight
        if self.alpha is None:
            self.alpha = torch.ones(logits.size(-1), device=logits.device)  # Uniform weight for all classes

        # If alpha is a tensor (per-class weights), gather it based on the labels
        alpha_t = self.alpha.gather(0, valid_labels.view(-1))  # shape [B]

        # Compute the focal loss components
        loss = -alpha_t * (1 - p_t) ** self.gamma * torch.log(p_t)
        # print((1 - p_t) ** self.gamma)
        # import ipdb
        # ipdb.set_trace()
        
        # Mask out the loss values for ignored labels (labels == ignore_index)
        loss = loss * mask.float()  # Ensure the mask is in float to be used for multiplication

        # Explicitly set loss to zero for labels == ignore_index
        loss = torch.where(mask, loss, torch.zeros_like(loss))

        # Handle NaNs and Infs (numerical stability)
        # loss = loss[torch.isfinite(loss)]  # Remove NaNs or Infs
        loss[~torch.isfinite(loss)] = 0      # Remove loss of NaNs or Infs with 0, which will be ignored during the loss average.

        # Apply reduction if needed
        if self.reduction == 'none':
            return loss         # Element-wise loss
        elif self.reduction == 'mean':
            return loss.mean()  # Mean loss
        elif self.reduction == 'sum':
            return loss.sum()   # Sum of losses

        return loss             # Default case



class CrossEntropyFrameDecay(nn.Module):
    def __init__(self, frame_num, special_token, device, start_coef, end_coef=1, reduction='none'):
        '''
        We'd better set end_coef to 1.
        '''
        super(CrossEntropyFrameDecay, self).__init__()
        self.loss_fct = CrossEntropyLoss(reduction=reduction)
        self.frame_coef = torch.linspace(start_coef, end_coef, frame_num).to(device)
        self.special_token = special_token
    
    def forward(self, logits, labels):
        ### Define special tokens
        # Extract the IDs for the different special tokens used in the sequence
        video_start_token = self.special_token['video_start_token']
        video_end_token = self.special_token['video_end_token']
        image_start_token = self.special_token['image_start_token']
        image_end_token = self.special_token['image_end_token']
        sep_token = self.special_token['sep_token']

        video_mask, has_complete_video = create_in_media_segment_mask(
            sequence=labels, 
            soi_id=video_start_token, 
            eoi_id=video_end_token
        )
        
        image_mask_list, has_complete_image = create_in_media_segment_mask_list(
            sequence=labels, 
            soi_id=image_start_token,
            eoi_id=image_end_token,
        )

        all_loss = []
        for frame_idx, (frame_mask, coef) in enumerate(zip(image_mask_list, self.frame_coef)):
            # loss_name = f'tmp_closs_{frame_wise_loss_type}_frame{frame_idx}'
            frame_logits = logits[frame_mask]
            frame_labels = labels[frame_mask]
            frame_loss = self.loss_fct(frame_logits, frame_labels) * coef
            all_loss.append(frame_loss)

            # frame_loss = frame_loss[frame_loss!=0].mean()
            # additional_loss_dict[loss_name] =  (frame_loss, 0.)
        
        return torch.cat(all_loss)



def chunked_logsumexp_stable(logits: torch.Tensor, chunk_size: int = 8192) -> torch.Tensor:
    """
    Computes logsumexp(logits, dim=-1) in a numerically stable way,
    chunking over the vocab dimension to reduce memory usage.

    Args:
        logits: A [batch_size, seq_len, vocab_size] tensor.
        chunk_size: Number of vocab elements to process at once.

    Returns:
        A [batch_size, seq_len] tensor of log-sum-exp over dim=-1.
    """
    B, T, V = logits.shape

    # 1. Find max along the vocab dimension (for stable exponent).
    #    Shape: [batch_size, seq_len, 1]
    max_vals = logits.max(dim=-1, keepdim=True).values

    # 2. Prepare an accumulator for the exponent sum, shape: [B, T].
    sumexp = torch.zeros(B, T, dtype=logits.dtype, device=logits.device)

    # 3. Loop over chunks of the vocab dimension.
    #    exponentiate only a fraction of the logits at a time.
    for start in range(0, V, chunk_size):
        end = min(start + chunk_size, V)
        # Slice the [B, T, chunk_size].
        chunk_logits = logits[..., start:end]
        # Subtract max for numerical stability.
        exp_chunk = (chunk_logits - max_vals).exp()  # [B, T, chunk]
        # Sum over the chunk dimension => [B, T].
        sumexp += exp_chunk.sum(dim=-1)

    # 4. Final log-sum-exp:
    #    logsumexp(x) = max(x) + log( sum( exp(x - max(x)) ) )
    lse = max_vals.squeeze(dim=-1) + sumexp.log()
    return lse




class CEWithChunkedOutputLoss(torch.nn.Module):
    """
    Cross-entropy with chunked outputs that saves memory by only upcasting one chunk at a time.

    Whenever the model is trained with bf16, before running CE, we have to upcast
    it to fp32 for better accuracy and stability. When upcasting happens, the memory usage doubles.
    Models like llama3 have large vocabulary size and, therefore, have a large output
    tensor of shape ``(bsz * num_tokens, vocab_size)``. Chunking reduces memory usage.

    For more details, please refer to: https://github.com/pytorch/torchtune/pull/1390
    """
    def __init__(self, chunk_size: int = 200, ignore_index: int = -100, reduction: str = "mean"):
        """
        Args:
            chunk_size (int): The size of every chunk.
            ignore_index (int): Index that will be ignored during loss computation.
            reduction (str): Specifies the reduction to apply to the output:
                'none' | 'mean' | 'sum'. Defaults to 'mean'.
        """
        super().__init__()
        self.chunk_size = chunk_size
        self.ignore_index = ignore_index
        self.reduction = reduction

    def compute_cross_entropy(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for a chunk.
        """
        return F.cross_entropy(
            logits.float(), labels, ignore_index=self.ignore_index, reduction="none"
        )

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): Logits tensor of shape ``(batch_size * num_tokens, vocab_size)``.
            labels (torch.Tensor): Ground truth labels of shape ``(batch_size * num_tokens)``.

        Returns:
            torch.Tensor: Cross entropy loss.
        """
        # Split logits and labels into chunks along the sequence dimension
        num_output_chunks = math.ceil(labels.shape[0] / self.chunk_size)
        logits_chunks = logits.chunk(num_output_chunks, dim=0)
        labels_chunks = labels.chunk(num_output_chunks, dim=0)

        total_elements = (labels != self.ignore_index).sum()
        # print(sum((chunk != self.ignore_index).sum() for chunk in labels_chunks), total_elements)

        # Compute the loss for each chunk
        chunk_losses = []
        for logits_chunk, labels_chunk in zip(logits_chunks, labels_chunks):
            chunk_losses.append(self.compute_cross_entropy(logits_chunk, labels_chunk))

        # Concatenate chunk losses
        total_loss = torch.cat(chunk_losses)

        # Apply reduction
        if self.reduction == "mean":
            return total_loss.sum() / total_elements
        elif self.reduction == "sum":
            return total_loss.sum()
        elif self.reduction == "none":
            # Return the loss per element
            return total_loss
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")



def define_loss(loss_type, start_coef=None, special_token=None, device=None, frame_num=None):
    if loss_type == 'Focal':
        loss_fct = FocalLoss(gamma=2.0, reduction='none', ignore_index=-100) 
    elif loss_type == 'CE':
        loss_fct = CrossEntropyLoss(reduction='none')
    elif loss_type == 'CEChunked':
        loss_fct = CEWithChunkedOutputLoss(reduction='none')
    elif loss_type == 'CEDecay':
        assert (start_coef is not None) and (special_token is not None) and (device is not None) and (frame_num is not None)
        loss_fct = CrossEntropyFrameDecay(
            frame_num=frame_num, special_token=special_token, device=device,
            start_coef=start_coef, end_coef=1, reduction='none'
        )
    else:
        assert False, f'{loss_type} loss is still not supported.'
    
    return loss_fct




#####################################################################################
#                              Masking Generation utils                             #
#####################################################################################

### This is the original one, copied from https://github.com/showlab/Show-o/blob/main/training/prompting_utils.py
def create_attention_mask_predict_next(sequence, pad_id=128256, soi_id=128257, eoi_id=128258, rm_pad_in_image=False,
                                       return_inverse_mask=True):
    """
    Create an attention mask for predicting the next token in a sequence.

    Args:
        sequence (torch.Tensor): Input tensor of shape [N, L], where N is the batch size and L is the sequence length.
        pad_id (int): The token ID used for padding (default: 128256).
        soi_id (int): The token ID that indicates the start of an image (default: 128257).
        eoi_id (int): The token ID that indicates the end of an image (default: 128258).
        rm_pad_in_image (bool): Whether to remove padding tokens from the attention mask in image segments (default: False).
        return_inverse_mask (bool): If True, returns an inverted mask where valid tokens are 1.0 and padding tokens are minimal value (default: True).

    Returns:
        torch.Tensor: An attention mask with an extra dimension for batch processing. If return_inverse_mask is True, it returns the inverted mask; otherwise, it returns the original mask.
    """
    # sequence is expected to be of shape [N, L]
    N, L = sequence.shape

    # Create a boolean mask to identify padding tokens
    is_padding = sequence == pad_id  # True for padding tokens
    
    # Create masks for start and end of image tokens
    is_start_image = sequence == soi_id   # True where the token is the start of an image
    is_end_image = sequence == eoi_id     # True where the token is the end of an image

    # Create cumulative sum masks to identify regions of image tokens
    cumulative_start = torch.cumsum(is_start_image, dim=1)  # Cumulative sum for start tokens
    cumulative_end = torch.cumsum(is_end_image, dim=1)      # Cumulative sum for end tokens

    # Identify segments of the sequence that contain image tokens
    # Doing this will generate a mask inclduing start_token and end_token.
    in_image_segment = (cumulative_start > cumulative_end) | is_start_image | is_end_image

    # Identify text tokens as those not in image segments
    is_text = ~(in_image_segment)
    
    # Create a causal mask to ensure that the attention is only from past tokens
    causal_mask = torch.tril(torch.ones((L, L), dtype=torch.bool)).to(sequence.device)

    # Create a mask for text tokens combined with the causal mask
    mask_text = is_text[:, :, None] * causal_mask[None, :, :] # [N, L, 1] * [1, L, L]

    # Combine text and image segment masks
    is_text_image = is_text | in_image_segment

    # Create a bidirectional mask for text and image segments
    mask_text_image_bi = is_text_image[:, :, None] * is_text_image[:, None, :] # [N, L, 1] * [N, 1, L]

    if rm_pad_in_image:
        # For each sequence in the batch, find where the start of the image token is
        sid_img = torch.where(sequence == soi_id)[1]   # Indices of start image tokens
        
        for i in range(mask_text_image_bi.shape[0]): # Loop through each sequence
            # Find the index of the last padding token
            pad_end_idx = torch.where(sequence[i] == pad_id)
            if len(pad_end_idx[0]) != 0:
                pad_end_idx = pad_end_idx[0][-1]
                # Set all tokens after the last padding index to not attend to previous tokens
                mask_text[i][pad_end_idx + 1:, :pad_end_idx + 1] = 0
            
            # Create a mask to ignore padding tokens within the image segments
            id_padding = torch.where(is_padding[i] == True)
            mask_text_image_bi[i][sid_img[i]:, id_padding[0]] = 0
    
    # Update the text mask to include the text-image bidirectional mask in the image segments
    mask_text[in_image_segment] = mask_text_image_bi[in_image_segment]
    # No token attends to padding tokens and padding tokens do not attend to any token
    if return_inverse_mask:
        inverted_mask = 1.0 - mask_text.type(sequence.dtype)
        inverted_mask = inverted_mask.masked_fill(
            inverted_mask.to(torch.bool), torch.iinfo(sequence.dtype).min
        )
        return inverted_mask.unsqueeze(1)
    else:
        return mask_text.unsqueeze(1)



def create_attention_mask_t2v(sequence, special_token, pad_id=0, rm_pad_in_image=False,
                              return_inverse_mask=True):
    """
    Create an attention mask for text-to-video (t2v) in a sequence.

    Args:
        sequence (torch.Tensor): Input tensor of shape [N, L], where N is the batch size and L is the sequence length.
        special_token (dict): Dictionary containing the special tokens, such as 'image_start_token' and 'image_end_token'.
        pad_id (int, optional): The token ID used for padding (default: 0).
        rm_pad_in_image (bool, optional): Whether to remove padding tokens from the attention mask in image segments (default: False).
        return_inverse_mask (bool, optional): If True, returns an inverted mask where valid tokens are 1.0 and padding tokens are minimal value (default: True).

    Returns:
        torch.Tensor: An attention mask of shape [N, 1, L, L] for batch processing. If return_inverse_mask is True,
                      it returns the inverted mask; otherwise, it returns the original mask.
    """

    # sequence is expected to be of shape [N, L]
    N, L = sequence.shape

    # Create a boolean mask to identify padding tokens
    is_padding = sequence == pad_id  # True for padding tokens

    # Identify segments of the sequence that contain image tokens
    # Doing this will generate a mask inclduing start_token and end_token.
    _, in_image_segment, _, _ = extract_image_video_masks(special_token=special_token, labels=sequence, image_content_mask=False)
    
    # Identify text tokens as those not in image segments
    is_text = ~(in_image_segment)

    # Combine text and image segment masks
    is_text_image = is_text | in_image_segment

    # Create a bidirectional mask for text and image segments
    # NOTE Test function passed
    # mask_text_image_bi = is_text_image[:, :, None] * is_text_image[:, None, :] # [N, L, 1] * [N, 1, L] # TODO Can be deleted
    mask_text_image_bi = create_bidirectional_mask_video(sequence, special_token, is_text_image, in_image_segment)



    # Remove padding tokens from the attention mask if specified
    if rm_pad_in_image:
        assert False, "This needs to be tested."
        # Ensure padding tokens do not attend to other tokens and are not attended by other tokens
        mask_text_image_bi = mask_text_image_bi & (~is_padding[:, :, None]) & (~is_padding[:, None, :])

    
    # # Update the text mask to include the text-image bidirectional mask in the image segments
    # mask_text[in_image_segment] = mask_text_image_bi[in_image_segment]  # TODO Can be deleted

    mask_text = mask_text_image_bi
    # No token attends to padding tokens and padding tokens do not attend to any token
    if return_inverse_mask:
        inverted_mask = 1.0 - mask_text.type(sequence.dtype)
        inverted_mask = inverted_mask.masked_fill(
            inverted_mask.to(torch.bool), torch.iinfo(sequence.dtype).min
        )
        return inverted_mask.unsqueeze(1)
    else:
        return mask_text.unsqueeze(1)


def create_bidirectional_mask_video(sequence, special_token, is_text_image, in_image_segment):
    """
    Masks being causal between frames while bi-directional within frames.

    Args:
        sequence (torch.Tensor): Input tensor of shape [N, L], where N is the batch size and L is the sequence length.


    Returns:
        torch.Tensor: An attention mask with an extra dimension for batch processing. If return_inverse_mask is True, it returns the inverted mask; otherwise, it returns the original mask.
    """

    # sequence is expected to be of shape [N, L]
    N, L = sequence.shape

    ### Step 1: Create the Causal Mask Across Frames
    causal_mask = torch.tril(torch.ones((L, L), dtype=torch.bool, device=sequence.device))

    ### Step 2: Create the Bidirectional Mask for Intra-frame Attention
    # Use the `in_image_segment` mask to determine which tokens are part of the same frame.
    frame_mask = torch.zeros((N, L, L), dtype=torch.bool, device=sequence.device)

    # Calculate the cumulative sum of `is_start_image` to track frame numbers
    is_start_image = sequence == special_token['image_start_token']
    frame_id = torch.cumsum(is_start_image, dim=1)  # Different frame_ids for different frames

    # Create bidirectional intra-frame mask based on frame ID matches
    frame_match = (frame_id[:, :, None] == frame_id[:, None, :])  # [N, L, L] - True if tokens are in the same frame
    bidirectional_frame_mask = in_image_segment[:, :, None] & in_image_segment[:, None, :] & frame_match

    # Set bidirectional attention within each frame
    frame_mask[bidirectional_frame_mask] = True

    ### Step 3: Combine Causal and Bidirectional Masks
    # Start with the causal mask and then add bidirectional intra-frame attention
    mask_text_image_bi = causal_mask[None, :, :].expand(N, L, L)  # Start with causal mask for all tokens
    mask_text_image_bi[frame_mask] = True  # Allow full attention within frames

    # Apply the general attention constraints (e.g., text and image segments)
    mask_text_image_bi = mask_text_image_bi & (is_text_image[:, :, None] & is_text_image[:, None, :])

    return mask_text_image_bi





#####################################################################################
#                                  inference utils                                  #
#####################################################################################

def calculate_resolution_from_token_list(sequence, soi_id, eoi_id, new_line_id, spatial_compress_ratio):
    """
    Calculate resolution form a token list.

    Args:
        sequence (torch.Tensor): Input tensor of shape [N, L], where N is the batch size and L is the sequence length.
        soi_id (int): The token ID that indicates the start of a media segment.
        eoi_id (int): The token ID that indicates the end of a media segment.
        new_line_id (int): The token ID that indicates a new line in the media segment.
        spatial_compress_ratio (int): The size of each patch, used to calculate the final resolution.

    Returns:
        list: A list containing the width and height of the media in pixels, calculated based on patch size.
    """
    ### NOTE Currently, we only support N==1.

    # sequence is expected to be of shape [N, L]
    N, L = sequence.shape
    assert N == 1, "N >= 1 is not supported, which needs further test."

    # Create masks for start and end of media tokens
    is_start_media = sequence[0] == soi_id   # True where the token is the start of a media segment
    is_end_media = sequence[0] == eoi_id     # True where the token is the end of a media segment

    # Use argmax to find the first token index for is_start_media
    media_token_first_start = torch.argmax(is_start_media.int()).item()

    # Use argmax to find the first token index for is_end_media
    media_token_first_end = torch.argmax(is_end_media.int()).item()

    # Create a mask for new line tokens within the media segment
    is_new_line = sequence[0][media_token_first_start : (media_token_first_end + 1)] == new_line_id
    new_line_indices = torch.where(is_new_line)[0]

    # Calculate width and height based on new line indices and patch size
    if len(new_line_indices) >= 2:
        w = (new_line_indices[1] - new_line_indices[0] - 1).item()
    else:
        # w = (media_token_first_end - media_token_first_start + 1).item()
        assert False, "This media item does not have enough lines."

    h = is_new_line.sum().item()

    return [w * spatial_compress_ratio, h * spatial_compress_ratio]



### V2: Mask append multiple frames.
def mask_append(
    sequence,               # Tensor: Input sequence of tokens to which a masked frame will be appended.
    resolution,             # Tuple[int, int]: Original resolution (width, height) of the input image.
    resolution_token,       # Tensor: Token that represents the compressed resolution of the image.
    spatial_compress_ratio, # int: Ratio used to compress the resolution of the image spatially.
    mask_token_id,          # int: Token ID representing a masked element.
    new_line_id,            # int: Token ID representing a new line (used when appending rows).
    image_start_token_id,   # int: Token ID representing the start of an image in the sequence.
    image_end_token_id,     # int: Token ID representing the start of an image in the sequence.
    num_frames = 1,         # int: Number of masked frames which will be appended.
):
    """
    Appends a compressed and masked version of a frame to the input sequence of tokens.

    Args:
        sequence (Tensor): The input sequence to which a masked version of the frame will be appended.
        resolution (Tuple[int, int]): Original width and height of the frame to be appended.
        resolution_token (Tensor): Token representing the compressed width and height.
        spatial_compress_ratio (int): Factor by which the spatial dimensions of the frame are compressed.
        mask_token_id (int): Token ID representing the mask token in the sequence.
        new_line_id (int): Token ID representing the new line character for row-wise representation.
        image_start_token_id (int): Token ID representing the start of an image section in the sequence.
        image_end_token_id (int): Token ID representing the start of an image section in the sequence.
        num_frames (int): Number of masked frames which will be appended, default: 1.

    Returns:
        sequence (Tensor): Updated sequence with the masked frame appended.
        mask_sequence (Tensor): A boolean tensor representing which tokens are masked in the updated sequence.
    """

    ### NOTE Currently, we only support N==1.
    w, h = resolution[0]//spatial_compress_ratio, resolution[1]//spatial_compress_ratio

    # Create a full-frame mask of tokens represented in a 2D grid of size (height, width)
    full_frame_mask_toks = torch.ones((h, w), dtype=sequence.dtype, device=sequence.device) * mask_token_id

    # Add new line tokens at the end of each row of the frame tokens
    full_frame_mask_toks = torch.cat(
        (
            full_frame_mask_toks,
            torch.ones((h, 1), dtype=sequence.dtype, device=sequence.device)
                * new_line_id,
        ),
        dim=1,
    ).flatten()

    # Prepare the list of frame tokens, including special tokens for the grid size
    # The frame tokens consist of:
    # - `image_start_token_id`: Token to indicate the start of an image section.
    # - `resolution_token`: Token indicating the width and height after compression.
    # - `full_frame_mask_toks`: Tokens representing the frame itself, along with new line tokens.
    # - `image_end_token_id`: Token to indicate the end of an image section.
    frame_tokens = torch.cat(
        [
            torch.tensor([image_start_token_id], device=sequence.device, dtype=sequence.dtype),  # Start token for the frame
            resolution_token,        # Token for compressed width and height
            full_frame_mask_toks,    # Mask tokens representing the entire frame
            torch.tensor([image_end_token_id], device=sequence.device, dtype=sequence.dtype),   # End token for the frame
        ]
    ).unsqueeze(dim=0)  # Add a new dimension for batch processing

    sequence = torch.cat([sequence,] + [frame_tokens,] * num_frames, dim=1)

    # Create a boolean mask for the tokens in the updated sequence that matches `mask_token_id`
    mask_sequence = sequence == mask_token_id

    return sequence, mask_sequence



#####################################################################################
#                             Model loading utils                                   #
#####################################################################################
def read_vocab_size(load_path):
    """
    Read the vocabulary size from the config file located in the specified path.

    Args:
        load_path (str): Path to the directory containing the config.json file.

    Returns:
        int: The vocab_size from the config file.

    Raises:
        FileNotFoundError: If the config.json file is not found in the specified directory.
        KeyError: If the vocab_size key is not present in the config.json file.
    """
    # Construct the full path to the config.json file
    config_path = os.path.join(load_path, "config.json")
    
    # Check if the config file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    # Read the config.json file
    with open(config_path, "r") as f:
        config = json.load(f)

    # Check if the vocab_size key exists in the config file
    if "vocab_size" in config:
        return config["vocab_size"]
    else:
        raise KeyError("'vocab_size' key is missing in the config file.")






#####################################################################################
#                                     misc                                          #
#####################################################################################
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

from torchvision import transforms
def image_transform(image, resolution=256, normalize=True):
    image = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC)(image)
    image = transforms.CenterCrop((resolution, resolution))(image)
    image = transforms.ToTensor()(image)
    if normalize:
        image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(image)
    return image


if __name__=="__main__":
    ### Test create_bidirectional_mask_video: case 1
    # sequence = torch.tensor([[1,2,3,100,101,102,200,1,2,3,100,101,102,200,1,2]])
    # special_token = {'image_start_token': 100}
    # is_text_image = torch.ones(sequence.shape, dtype=torch.bool)
    # in_image_segment = torch.tensor([[False,False,False,True,True,True,True,False,False,False,True,True,True,True,False,False]])

    # # print(create_bidirectional_mask_video(sequence, special_token, is_text_image, in_image_segment))
    # # tensor([[[ True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
    # #         [ True,  True, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
    # #         [ True,  True,  True, False, False, False, False, False, False, False, False, False, False, False, False, False],
    # #         [ True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False],
    # #         [ True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False],
    # #         [ True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False],
    # #         [ True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False],
    # #         [ True,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False, False],
    # #         [ True,  True,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False],
    # #         [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False],
    # #         [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True, True,  True,  True,  True, False, False],
    # #         [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True, True,  True,  True,  True, False, False],
    # #         [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True, True,  True,  True,  True, False, False],
    # #         [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True, True,  True,  True,  True, False, False],
    # #         [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True, True,  True,  True,  True,  True, False],
    # #         [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,True,  True,  True,  True,  True,  True]]])
    
    ### Test create_bidirectional_mask_video: case 2
    # sequence = torch.tensor([[1,2,3,100,101,200,100,101,200,1,2]])
    # special_token = {'image_start_token': 100}
    # is_text_image = torch.ones(sequence.shape, dtype=torch.bool)
    # in_image_segment = torch.tensor([[False,False,False,True,True,True,True,True,True,False,False]])

    # # print(create_bidirectional_mask_video(sequence, special_token, is_text_image, in_image_segment))
    # # tensor([[[ True, False, False, False, False, False, False, False, False, False, False],
    # #         [ True,  True, False, False, False, False, False, False, False, False, False],
    # #         [ True,  True,  True, False, False, False, False, False, False, False, False],
    # #         [ True,  True,  True,  True,  True,  True, False, False, False, False, False],
    # #         [ True,  True,  True,  True,  True,  True, False, False, False, False, False],
    # #         [ True,  True,  True,  True,  True,  True, False, False, False, False, False],
    # #         [ True,  True,  True,  True,  True,  True,  True,  True,  True, False, False],
    # #         [ True,  True,  True,  True,  True,  True,  True,  True,  True, False, False],
    # #         [ True,  True,  True,  True,  True,  True,  True,  True,  True, False, False],
    # #         [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True, False],
    # #         [ True,  True,  True,  True,  True,  True,  True,  True,  True,  True, True]]])


    ### Test focal loss
    # Example usage:
    # Suppose you have 3 classes, and class 2 is the minority class
    alpha = None
    # alpha = torch.tensor([0.1, 0.3, 0.6])  # Larger weight for class 2 (minority class)
    mar_logits = torch.tensor([[0.1, 0.3, 0.6], [0.1, 0.3, 0.6], [0.1, 0.3, 0.6], [0.1, 0.3, 0.6], [0.1, 0.3, 0.6]])
    mar_labels = torch.tensor([-100, 1, -100, 0, 2])
    # mar_labels = torch.tensor([0, 1, 2])
    loss_fct = FocalLoss(gamma=2.0, alpha=alpha, reduction='none', ignore_index=-100)
    mar_loss = loss_fct(mar_logits, mar_labels)
    print(mar_loss)
