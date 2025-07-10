# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A library for Causal Video Tokenizer inference."""

import numpy as np
import torch
from typing import Any, Union

from tqdm import tqdm

import PIL
from PIL import Image
from model.cosmos_tokenizer.utils import (
    load_model,
    load_encoder_model,
    load_decoder_model,
    numpy2tensor,
    pad_video_batch,
    tensor2numpy,
    unpad_video_batch,
)
from model.cosmos_tokenizer.networks import TokenizerConfigs

#################################################################
#                Original Causal Video Tokenizer                #            
#################################################################
class CausalVideoTokenizer(torch.nn.Module):
    def __init__(
        self,
        checkpoint: str = None,
        checkpoint_enc: str = None,
        checkpoint_dec: str = None,
        tokenizer_config: dict[str, Any] = None,
        device: str = "cuda",
        dtype: Union[str, torch.dtype] = "bfloat16"
    ) -> None:
        super().__init__()
        self._device = device
        self._dtype = dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype) # getattr(torch, dtype)
        self._full_model = (
            load_model(checkpoint, tokenizer_config, device).to(self._dtype)
            if checkpoint is not None
            else None
        )
        self._enc_model = (
            load_encoder_model(checkpoint_enc, tokenizer_config, device).to(self._dtype)
            if checkpoint_enc is not None
            else None
        )
        self._dec_model = (
            load_decoder_model(checkpoint_dec, tokenizer_config, device).to(self._dtype)
            if checkpoint_dec is not None
            else None
        )

    @torch.no_grad()
    def autoencode(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Reconstrcuts a batch of video tensors after embedding into a latent.

        Args:
            video: The input video Bx3xTxHxW layout, range [-1..1].
        Returns:
            The reconstructed video, layout Bx3xTxHxW, range [-1..1].
        """
        if self._full_model is not None:
            output_tensor = self._full_model(input_tensor)
            output_tensor = (
                output_tensor[0] if isinstance(output_tensor, tuple) else output_tensor
            )
        else:
            output_latent = self.encode(input_tensor)[0]
            output_tensor = self.decode(output_latent)
        return output_tensor

    @torch.no_grad()
    def encode(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor]:
        """Encodes a numpy video into a CausalVideo latent or code.

        Args:
            input_tensor: The input tensor Bx3xTxHxW layout, range [-1..1].
        Returns:
            For causal continuous video (CV) tokenizer, the tuple contains:
                - The latent embedding, Bx16x(t)x(h)x(w), where the compression
                rate is (T/t x H/h x W/w), and channel dimension of 16.
            For causal discrete video (DV) tokenizer, the tuple contains:
              1) The indices, Bx(t)x(h)x(w), from a codebook of size 64K, which
                is formed by FSQ levels of (8,8,8,5,5,5).
              2) The discrete code, Bx6x(t)x(h)x(w), where the compression rate
                is again (T/t x H/h x W/w), and channel dimension of 6.
        """
        assert input_tensor.ndim == 5, "input video should be of 5D."

        output_latent = self._enc_model(input_tensor)
        if isinstance(output_latent, torch.Tensor):
            return output_latent
        return output_latent[:-1]

    @torch.no_grad()
    def decode(self, input_latent: torch.Tensor) -> torch.Tensor:
        """Encodes a numpy video into a CausalVideo latent.

        Args:
            input_latent: The continuous latent Bx16xtxhxw for CV,
                        or the discrete indices Bxtxhxw for DV.
        Returns:
            The reconstructed tensor, layout [B,3,1+(T-1)*8,H*16,W*16] in range [-1..1].
        """
        assert (
            input_latent.ndim >= 4
        ), "input latent should be of 5D for continuous and 4D for discrete."
        return self._dec_model(input_latent)

    def forward(
        self,
        video: np.ndarray,
        temporal_window: int = 17,
    ) -> np.ndarray:
        """Reconstructs video using a pre-trained CausalTokenizer autoencoder.
        Given a video of arbitrary length, the forward invokes the CausalVideoTokenizer
        in a sliding manner with a `temporal_window` size.

        Args:
            video: The input video BxTxHxWx3 layout, range [0..255].
            temporal_window: The length of the temporal window to process, default=25.
        Returns:
            The reconstructed video in range [0..255], layout BxTxHxWx3.
        """
        assert video.ndim == 5, "input video should be of 5D."
        num_frames = video.shape[1]  # can be of any length.
        output_video_list = []
        for idx in tqdm(range(0, (num_frames - 1) // temporal_window + 1)):
            # Input video for the current window.
            start, end = idx * temporal_window, (idx + 1) * temporal_window
            input_video = video[:, start:end, ...]

            # Spatio-temporally pad input_video so it's evenly divisible.
            padded_input_video, crop_region = pad_video_batch(input_video)
            input_tensor = numpy2tensor(
                padded_input_video, dtype=self._dtype, device=self._device
            )
            output_tensor = self.autoencode(input_tensor)
            padded_output_video = tensor2numpy(output_tensor)
            output_video = unpad_video_batch(padded_output_video, crop_region)

            output_video_list.append(output_video)
        return np.concatenate(output_video_list, axis=1)



#################################################################
#        Extended Causal Video Tokenizer for Easy Usage         #            
#################################################################
class ExtendedCausalVideoTokenizer(CausalVideoTokenizer):
    def __init__(
        self,
        checkpoint: str = None,
        checkpoint_enc: str = None,
        checkpoint_dec: str = None,
        tokenizer_config: dict[str, Any] = None,
        device: str = "cuda",
        dtype: str = "bfloat16",
    ) -> None:
        # Initialize the parent class
        super().__init__(checkpoint, checkpoint_enc, checkpoint_dec, tokenizer_config, device, dtype)
    
    def vid_tokens_from_pil_list(self, img_list: list[PIL.Image]) -> list[list[int]]:
        processed_imgs = []
        for img in img_list:
            img = self._whiten_transparency(img)
            np_img = np.array(img) / 255.0  # Normalize to [0, 1]
            np_img = np_img * 2 - 1  # Scale to [-1, 1]
            img_tensor = torch.from_numpy(np_img).permute(2, 0, 1).to(self._device) # .to(self._vq_model.encoder.conv_in.weight)
            processed_imgs.append(img_tensor)
        
        processed_imgs = torch.stack(processed_imgs).to(self._dtype)
        imgs = processed_imgs[None, ...].transpose(1,2) # Add batch dimension
        toks_tensor = self.encode(imgs)[0]
        _, compressed_t, compressed_h, compressed_w = toks_tensor.shape
        toks_tensor = toks_tensor.reshape(compressed_t, compressed_h * compressed_w)
        toks_list = toks_tensor.int() # .tolist()

        return toks_list

    def pil_from_img_toks_list(self, tokens_list: list[torch.Tensor], h_latent_dim=32, w_latent_dim=32) -> list[PIL.Image]:
        pass
        assert False, "To be implemented."

    def img_tokens_from_pil(self, img: PIL.Image) -> list[int]:
        pass
        assert False, "To be implemented."
    
    def pil_from_img_toks(self, tokens: torch.Tensor, h_latent_dim=32, w_latent_dim=32) -> PIL.Image:
        pass
        assert False, "To be implemented."
    
    def _whiten_transparency(self, img: PIL.Image) -> PIL.Image:
        # Check if it's already in RGB format.
        if img.mode == "RGB":
            return img

        vals_rgba = np.array(img.convert("RGBA"))

        # If there is no transparency layer, simple convert and return.
        if not (vals_rgba[:, :, 3] < 255).any():
            return img.convert("RGB")

        # There is a transparency layer, blend it with a white background.

        # Calculate the alpha proportion for blending.
        alpha = vals_rgba[:, :, 3] / 255.0
        # Blend with white background.
        vals_rgb = (1 - alpha[:, :, np.newaxis]) * 255 + alpha[:, :, np.newaxis] * vals_rgba[:, :, :3]
        return PIL.Image.fromarray(vals_rgb.astype("uint8"), "RGB")
    
    def _pil_from_chw_tensor(self, chw_tensor: torch.Tensor) -> PIL.Image:
        # Ensure detachment and move tensor to CPU.
        detached_chw_tensor = chw_tensor.detach().cpu()

        # Normalize tensor to [0, 1] range from [-1, 1] range.
        normalized_chw_tensor = (torch.clamp(detached_chw_tensor, -1.0, 1.0) + 1.0) / 2.0

        # Permute CHW tensor to HWC format and convert to NumPy array.
        hwc_array = normalized_chw_tensor.permute(1, 2, 0).numpy()

        # Convert to an 8-bit unsigned integer format.
        image_array_uint8 = (hwc_array * 255).astype(np.uint8)

        # Convert NumPy array to PIL Image.
        pil_image = Image.fromarray(image_array_uint8)

        # Convert image to RGB if it is not already.
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        return pil_image







# ### ！！！ NOTE NOTE NOTE 这个应该要删掉，就是Chameleon原始的image tokenizer
# class ImageTokenizer:
#     def __init__(
#         self,
#         cfg_path: str,
#         ckpt_path: str,
#         device: str | torch.device | None = None,
#     ):
#         with open(cfg_path) as f:
#             config = yaml.safe_load(f)

#         params = config["model"]["params"]
#         if "lossconfig" in params:
#             del params["lossconfig"]
#         params["ckpt_path"] = ckpt_path

#         self._vq_model = VQModel(**params)
#         self._vq_model.eval()

#         if device is None:
#             devices = {p.device for p in self._vq_model.parameters()}
#             assert len(devices) == 1
#             device = devices.pop()
#         else:
#             self._vq_model.to(device)
#         self._device = device

#         dtypes = {p.dtype for p in self._vq_model.parameters()}
#         assert len(dtypes) == 1
#         self._dtype = dtypes.pop()

#     def _whiten_transparency(self, img: PIL.Image) -> PIL.Image:
#         # Check if it's already in RGB format.
#         if img.mode == "RGB":
#             return img

#         vals_rgba = np.array(img.convert("RGBA"))

#         # If there is no transparency layer, simple convert and return.
#         if not (vals_rgba[:, :, 3] < 255).any():
#             return img.convert("RGB")

#         # There is a transparency layer, blend it with a white background.

#         # Calculate the alpha proportion for blending.
#         alpha = vals_rgba[:, :, 3] / 255.0
#         # Blend with white background.
#         vals_rgb = (1 - alpha[:, :, np.newaxis]) * 255 + alpha[:, :, np.newaxis] * vals_rgba[:, :, :3]
#         return PIL.Image.fromarray(vals_rgb.astype("uint8"), "RGB")

#     # def _vqgan_input_from(self, img: PIL.Image, target_image_size=512) -> torch.Tensor:
#     #     # Resize with aspect ratio preservation.
#     #     s = min(img.size)
#     #     scale = target_image_size / s
#     #     new_size = (round(scale * img.size[0]), round(scale * img.size[1]))
#     #     img = img.resize(new_size, PIL.Image.LANCZOS)
#     #
#     #     # Center crop.
#     #     x0 = (img.width - target_image_size) // 2
#     #     y0 = (img.height - target_image_size) // 2
#     #     img = img.crop((x0, y0, x0 + target_image_size, y0 + target_image_size))
#     #
#     #     # Convert to tensor.
#     #     np_img = np.array(img) / 255.0  # Normalize to [0, 1]
#     #     np_img = np_img * 2 - 1  # Scale to [-1, 1]
#     #     tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).float()  # (Channels, Height, Width) format.
#     #
#     #     # Add batch dimension.
#     #     return tensor_img.unsqueeze(0)

#     def img_tokens_from_pil(self, img: PIL.Image) -> list[int]:
#         img = self._whiten_transparency(img)
#         # Convert to tensor.
#         np_img = np.array(img) / 255.0  # Normalize to [0, 1]
#         np_img = np_img * 2 - 1  # Scale to [-1, 1]
#         img = torch.from_numpy(np_img).permute(2, 0, 1).to(self._vq_model.encoder.conv_in.weight)
#         img = img.unsqueeze(0)

#         _, _, [_, _, img_toks] = self._vq_model.encode(img)
#         return img_toks
    
#     def img_tokens_from_pil_list(self, img_list: list[PIL.Image], batch_size: int = 32, keep_batch: bool = False) -> list[list[int]]:
#         # Batch processing for multiple images with specified batch size.
#         all_tokens_list = []
#         for i in range(0, len(img_list), batch_size):
#             batch_imgs = img_list[i:i + batch_size]

#             processed_imgs = []
#             for img in batch_imgs:
#                 img = self._whiten_transparency(img)
#                 np_img = np.array(img) / 255.0  # Normalize to [0, 1]
#                 np_img = np_img * 2 - 1  # Scale to [-1, 1]
#                 img_tensor = torch.from_numpy(np_img).permute(2, 0, 1).to(self._vq_model.encoder.conv_in.weight)
#                 processed_imgs.append(img_tensor)

#             # Stack images into a batch.
#             img_batch = torch.stack(processed_imgs)

#             _, _, [_, _, img_toks_batch] = self._vq_model.encode(img_batch)
            
#             if keep_batch:
#                 img_toks_batch = img_toks_batch.reshape(len(batch_imgs), -1)

#             all_tokens_list.extend(img_toks_batch)

#         return all_tokens_list

#     def _pil_from_chw_tensor(self, chw_tensor: torch.Tensor) -> PIL.Image:
#         # Ensure detachment and move tensor to CPU.
#         detached_chw_tensor = chw_tensor.detach().cpu()

#         # Normalize tensor to [0, 1] range from [-1, 1] range.
#         normalized_chw_tensor = (torch.clamp(detached_chw_tensor, -1.0, 1.0) + 1.0) / 2.0

#         # Permute CHW tensor to HWC format and convert to NumPy array.
#         hwc_array = normalized_chw_tensor.permute(1, 2, 0).numpy()

#         # Convert to an 8-bit unsigned integer format.
#         image_array_uint8 = (hwc_array * 255).astype(np.uint8)

#         # Convert NumPy array to PIL Image.
#         pil_image = Image.fromarray(image_array_uint8)

#         # Convert image to RGB if it is not already.
#         if pil_image.mode != "RGB":
#             pil_image = pil_image.convert("RGB")

#         return pil_image

#     def pil_from_img_toks(self, tokens: torch.Tensor, h_latent_dim=32, w_latent_dim=32) -> PIL.Image:
#         emb_dim = self._vq_model.quantize.embedding.weight.shape[-1]
#         codebook_entry = self._vq_model.quantize.get_codebook_entry(tokens, (1, h_latent_dim, w_latent_dim, emb_dim))
#         pixels = self._vq_model.decode(codebook_entry)
#         return self._pil_from_chw_tensor(pixels[0])
    
#     def pil_from_img_toks_list(self, tokens_list: list[torch.Tensor], h_latent_dim=32, w_latent_dim=32) -> list[PIL.Image]:
#         # Batch processing for multiple token lists.
#         batch_tokens = torch.stack(tokens_list)
#         emb_dim = self._vq_model.quantize.embedding.weight.shape[-1]
#         codebook_entry = self._vq_model.quantize.get_codebook_entry(batch_tokens, (len(tokens_list), h_latent_dim, w_latent_dim, emb_dim))
#         pixels_batch = self._vq_model.decode(codebook_entry)

#         pil_images = [self._pil_from_chw_tensor(pixels) for pixels in pixels_batch]
#         return pil_images

#     def latent_embedding_from_pil(self, img: PIL.Image):
#         img = self._whiten_transparency(img)

#         # Convert to tensor.
#         np_img = np.array(img) / 255.0  # Normalize to [0, 1]
#         np_img = np_img * 2 - 1  # Scale to [-1, 1]
#         img = torch.from_numpy(np_img).permute(2, 0, 1)  # (Channels, Height, Width) format.
#         img = img.unsqueeze(0).to(self._vq_model.encoder.conv_in.weight)
#         latent_embedding, _, _ = self._vq_model.encode(img)
#         return latent_embedding


if __name__=="__main__":
    cosmos_type = "Cosmos-Tokenizer-DV4x8x8"
    tokenizer_type = "DV" if "DV" in cosmos_type else "CV"
    spatial_compression = 8 # TODO This should be read from the string cosmos_type.
    temporal_compression = 4
    tokenizer_config = TokenizerConfigs[tokenizer_type].value
    tokenizer_config.update(dict(spatial_compression=spatial_compression))
    tokenizer_config.update(dict(temporal_compression=temporal_compression))
    tokenizer = ExtendedCausalVideoTokenizer(
        checkpoint = None,
        checkpoint_enc = 'ckpts/cosmos/Cosmos-Tokenizer-DV4x8x8/encoder.jit',
        checkpoint_dec = 'ckpts/cosmos/Cosmos-Tokenizer-DV4x8x8/decoder.jit',
        tokenizer_config = tokenizer_config,
        device = 'cuda',
        dtype = "bfloat16",
    )
