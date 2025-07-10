# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Chameleon License found in the
# LICENSE file in the root directory of this source tree.

from functools import cached_property

import torch


class VocabInfo:
    def __init__(self, vocab_map: dict[str, int]):
        self.name2val = vocab_map

        self.bos_id = vocab_map.get("<s>")
        self.eos_id = vocab_map.get("</s>")
        self.boi_id = vocab_map.get("<racm3:break>")
        self.eoi_id = vocab_map.get("<eoss>")
        self.pad_id = vocab_map.get("<pad>")
        self.eot_id = vocab_map.get("<reserved08706>")

    @property
    def begin_sequence(self) -> int:
        return self.bos_id

    @property
    def end_sequence(self) -> int:
        return self.eos_id

    @property
    def begin_image(self) -> int:
        return self.boi_id

    @property
    def end_image(self) -> int:
        return self.eoi_id

    @property
    def padding(self) -> int:
        return self.pad_id

    @property
    def end_turn(self) -> int:
        return self.eot_id

    @cached_property
    def val2name(self) -> dict[int, str]:
        return {v: k for k, v in self.name2val.items()}

    @cached_property
    def all_tokens(self) -> list[int]:
        return sorted(self.name2val.values())

    @cached_property
    def image_tokens(self) -> list[int]:
        return sorted([val for name, val in self.name2val.items() if name.startswith("IMGIMG")])
    
    @cached_property
    def video_tokens(self) -> list[int]:
        return sorted([val for name, val in self.name2val.items() if name.startswith("VIDVID")])

    @cached_property
    def special_tokens(self) -> list[int]:
        return sorted([val for name, val in self.name2val.items() if name.startswith("<") and name != "<"])

    @cached_property
    def text_tokens(self) -> list[int]:
        if len(self.video_tokens) > 0:
            return sorted(set(self.all_tokens) - set(self.image_tokens) - set(self.video_tokens) - set(self.special_tokens))
        else:
            return sorted(set(self.all_tokens) - set(self.image_tokens) - set(self.special_tokens))


class VocabTranslation:
    def __init__(self, vocab_info: VocabInfo, device: str | None = None):
        """
        Initializes the VocabTranslation class with vocab information and an optional device.

        Args:
            vocab_info (VocabInfo): An object containing vocabulary information, including token-to-name and image tokens.
            device (str, optional): The device to store tensors on (e.g., 'cpu' or 'cuda'). Defaults to None.
        """
        self._vocab = vocab_info
        self._device = device

    @cached_property # @cached_property: This is a Python decorator that ensures the result of the method is computed once and cached for subsequent calls. So, the first time bpe2img is accessed, it will compute the value, but after that, it will return the cached value.
    def bpe2img(self) -> dict[int, int]:
        """
        Maps BPE tokens to image tokens by converting BPE token names to a corresponding image token number.

        This function uses a dictionary to map BPE tokens to image token numbers, replacing characters 'A' through 'J'
        with digits 0 through 9 in the BPE token names.

        Returns:
            dict[int, int]: A dictionary mapping BPE token IDs to corresponding image token IDs.
        """

        # This line creates a dictionary that maps characters 'A' to 'J' (the first 10 uppercase letters) to the corresponding digits '0' to '9'.
        img_tkn_chr_mapping = {chr(ord("A") + i): str(i) for i in range(10)}
        # output: {'A': '0', 'B': '1', 'C': '2', 'D': '3', 'E': '4', 'F': '5', 'G': '6', 'H': '7', 'I': '8', 'J': '9'}

        def remap(old_name: str) -> str:
            return "".join(img_tkn_chr_mapping.get(c, c) for c in old_name[len("IMGIMG") : -1])
        
        # return a dict like {4:0, 5:1, 6:2, 7:3, ..., 8191: 8187, 8192: 8188, 8193: 8189, 8194: 8190, 8195: 8191}
        return {tok: int(remap(self._vocab.val2name[tok])) for tok in self._vocab.image_tokens}

    @cached_property
    def img2bpe(self) -> dict[int, int]:
        """
        Maps image tokens to BPE tokens by reversing the mapping from bpe2img.

        Returns:
            dict[int, int]: A dictionary mapping image token IDs back to their corresponding BPE token IDs.
        """
        # reverse the key and value order.
        # return a dict like: {0:4, 1:5, 2:6, 3:7, ..., 8187: 8191, 8188: 8192, 8189: 8193, 8190: 8194, 8191: 8195}
        return {v: k for k, v in self.bpe2img.items()}

    @cached_property
    def bpe2img_search_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Creates and returns sorted tensors for efficient token lookup during conversion.

        This function sorts the BPE tokens and their corresponding image tokens into tensors for fast access during
        the conversion process.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing two tensors:
                - sorted BPE tokens tensor
                - sorted image tokens tensor
        """
        sorted_bpe = torch.tensor(sorted(self.bpe2img.keys()), device=self._device)
        sorted_img = torch.tensor(sorted(self.bpe2img.values()), device=self._device)
        return sorted_bpe, sorted_img

    @cached_property
    def img2bpe_mapping_tensor(self) -> torch.LongTensor:
        """
        Creates a tensor that directly maps image token IDs to their corresponding BPE token IDs.

        This tensor is used for efficient lookups during the conversion from image tokens to BPE tokens.

        Returns:
            torch.LongTensor: A tensor where the index corresponds to the image token and the value is the BPE token ID.
        """
        mapping = torch.zeros(
            max(self.img2bpe.keys()) + 1,
            dtype=torch.int,
            device=self._device,
        )
        for k, v in self.img2bpe.items():
            mapping[k] = v
        return mapping

    def convert_bpe2img(self, bpe_batch: torch.Tensor) -> torch.Tensor:
        """
        Converts a batch of BPE token IDs to corresponding image token IDs.

        This function uses the search tensors for efficient conversion of BPE tokens to image tokens.

        Args:
            bpe_batch (torch.Tensor): A tensor of BPE token IDs of shape [N, L], where N is the batch size and L is the sequence length.

        Returns:
            torch.Tensor: A tensor of converted image token IDs of shape [N, L].
        """
        bpe_tok, img_tok = self.bpe2img_search_tensors
        return img_tok[torch.searchsorted(bpe_tok, bpe_batch)]

    def convert_img2bp2(self, img_batch: torch.Tensor) -> torch.Tensor:
        """
        Converts a batch of image token IDs to corresponding BPE token IDs.

        This function uses the precomputed mapping tensor for efficient conversion from image tokens to BPE tokens.

        Args:
            img_batch (torch.Tensor): A tensor of image token IDs of shape [N, L], where N is the batch size and L is the sequence length.

        Returns:
            torch.Tensor: A tensor of converted BPE token IDs of shape [N, L].
        """
        return self.img2bpe_mapping_tensor[img_batch]
    
    ##########################################################################
    #                        Video related functions                         #
    ##########################################################################

    @cached_property
    def bpe2vid(self) -> dict[int, int]:
        """
        Maps BPE tokens to video tokens by converting BPE token names to a corresponding video token number.

        This function uses a dictionary to map BPE tokens to video token numbers, replacing characters 'A' through 'J'
        with digits 0 through 9 in the BPE token names.

        Returns:
            dict[int, int]: A dictionary mapping BPE token IDs to corresponding video token IDs.
        """

        # This line creates a dictionary that maps characters 'A' to 'J' (the first 10 uppercase letters) to the corresponding digits '0' to '9'.
        vid_tkn_chr_mapping = {chr(ord("A") + i): str(i) for i in range(10)}
        # output: {'A': '0', 'B': '1', 'C': '2', 'D': '3', 'E': '4', 'F': '5', 'G': '6', 'H': '7', 'I': '8', 'J': '9'}

        def remap(old_name: str) -> str:
            return "".join(vid_tkn_chr_mapping.get(c, c) for c in old_name[len("VIDVID") : -1])
        
        # return a dict like {65536:0, 5:1, 6:2, 7:3, ..., 8191: 8187, 8192: 8188, 8193: 8189, 8194: 8190, 8195: 8191}
        return {tok: int(remap(self._vocab.val2name[tok])) for tok in self._vocab.video_tokens}
    
    @cached_property
    def vid2bpe(self) -> dict[int, int]:
        """
        Maps video tokens to BPE tokens by reversing the mapping from bpe2vid.

        Returns:
            dict[int, int]: A dictionary mapping video token IDs back to their corresponding BPE token IDs.
        """
        # reverse the key and value order.
        # return a dict like: {0:4, 1:5, 2:6, 3:7, ..., 8187: 8191, 8188: 8192, 8189: 8193, 8190: 8194, 8191: 8195}
        return {v: k for k, v in self.bpe2vid.items()}
    
    @cached_property
    def bpe2vid_search_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Creates and returns sorted tensors for efficient token lookup during conversion.

        This function sorts the BPE tokens and their corresponding video tokens into tensors for fast access during
        the conversion process.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing two tensors:
                - sorted BPE tokens tensor
                - sorted video tokens tensor
        """
        sorted_bpe = torch.tensor(sorted(self.bpe2vid.keys()), device=self._device)
        sorted_vid = torch.tensor(sorted(self.bpe2vid.values()), device=self._device)
        return sorted_bpe, sorted_vid

    @cached_property
    def vid2bpe_mapping_tensor(self) -> torch.LongTensor:
        """
        Creates a tensor that directly maps video token IDs to their corresponding BPE token IDs.

        This tensor is used for efficient lookups during the conversion from video tokens to BPE tokens.

        Returns:
            torch.LongTensor: A tensor where the index corresponds to the video token and the value is the BPE token ID.
        """
        mapping = torch.zeros(
            max(self.vid2bpe.keys()) + 1,
            dtype=torch.int,
            device=self._device,
        )
        for k, v in self.vid2bpe.items():
            mapping[k] = v
        return mapping

    def convert_bpe2vid(self, bpe_batch: torch.Tensor) -> torch.Tensor:
        """
        Converts a batch of BPE token IDs to corresponding video token IDs.

        This function uses the search tensors for efficient conversion of BPE tokens to video tokens.

        Args:
            bpe_batch (torch.Tensor): A tensor of BPE token IDs of shape [N, L], where N is the batch size and L is the sequence length.

        Returns:
            torch.Tensor: A tensor of converted video token IDs of shape [N, L].
        """
        bpe_tok, vid_tok = self.bpe2vid_search_tensors
        return vid_tok[torch.searchsorted(bpe_tok, bpe_batch)]

    def convert_vid2bp2(self, vid_batch: torch.Tensor) -> torch.Tensor:
        """
        Converts a batch of video token IDs to corresponding BPE token IDs.

        This function uses the precomputed mapping tensor for efficient conversion from video tokens to BPE tokens.

        Args:
            vid_batch (torch.Tensor): A tensor of video token IDs of shape [N, L], where N is the batch size and L is the sequence length.

        Returns:
            torch.Tensor: A tensor of converted BPE token IDs of shape [N, L].
        """
        return self.vid2bpe_mapping_tensor[vid_batch]
