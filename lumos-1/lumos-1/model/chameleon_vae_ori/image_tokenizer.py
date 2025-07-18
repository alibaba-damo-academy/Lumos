# Copyright (c) Meta Platforms, Inc. and affiliates
#
# This source code is licensed under the Chameleon License found in the
# LICENSE file in the root directory of this source tree.

import PIL
from PIL import Image
import numpy as np
import torch
import yaml

from .vqgan import VQModel


class ImageTokenizer:
    def __init__(
        self,
        cfg_path: str,
        ckpt_path: str,
        device: str | torch.device | None = None,
    ):
        with open(cfg_path) as f:
            config = yaml.safe_load(f)

        params = config["model"]["params"]
        if "lossconfig" in params:
            del params["lossconfig"]
        params["ckpt_path"] = ckpt_path

        self._vq_model = VQModel(**params)
        self._vq_model.eval()

        if device is None:
            devices = {p.device for p in self._vq_model.parameters()}
            assert len(devices) == 1
            device = devices.pop()
        else:
            self._vq_model.to(device)
        self._device = device

        dtypes = {p.dtype for p in self._vq_model.parameters()}
        assert len(dtypes) == 1
        self._dtype = dtypes.pop()

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

    # def _vqgan_input_from(self, img: PIL.Image, target_image_size=512) -> torch.Tensor:
    #     # Resize with aspect ratio preservation.
    #     s = min(img.size)
    #     scale = target_image_size / s
    #     new_size = (round(scale * img.size[0]), round(scale * img.size[1]))
    #     img = img.resize(new_size, PIL.Image.LANCZOS)
    #
    #     # Center crop.
    #     x0 = (img.width - target_image_size) // 2
    #     y0 = (img.height - target_image_size) // 2
    #     img = img.crop((x0, y0, x0 + target_image_size, y0 + target_image_size))
    #
    #     # Convert to tensor.
    #     np_img = np.array(img) / 255.0  # Normalize to [0, 1]
    #     np_img = np_img * 2 - 1  # Scale to [-1, 1]
    #     tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).float()  # (Channels, Height, Width) format.
    #
    #     # Add batch dimension.
    #     return tensor_img.unsqueeze(0)

    def img_tokens_from_pil(self, img: PIL.Image) -> list[int]:
        img = self._whiten_transparency(img)
        # Convert to tensor.
        np_img = np.array(img) / 255.0  # Normalize to [0, 1]
        np_img = np_img * 2 - 1  # Scale to [-1, 1]
        img = torch.from_numpy(np_img).permute(2, 0, 1).to(self._vq_model.encoder.conv_in.weight)
        img = img.unsqueeze(0)

        _, _, [_, _, img_toks] = self._vq_model.encode(img)
        return img_toks
    
    def img_tokens_from_pil_list(self, img_list: list[PIL.Image], batch_size: int = 32, keep_batch: bool = False) -> list[list[int]]:
        # Batch processing for multiple images with specified batch size.
        all_tokens_list = []
        for i in range(0, len(img_list), batch_size):
            batch_imgs = img_list[i:i + batch_size]

            processed_imgs = []
            for img in batch_imgs:
                img = self._whiten_transparency(img)
                np_img = np.array(img) / 255.0  # Normalize to [0, 1]
                np_img = np_img * 2 - 1  # Scale to [-1, 1]
                img_tensor = torch.from_numpy(np_img).permute(2, 0, 1).to(self._vq_model.encoder.conv_in.weight)
                processed_imgs.append(img_tensor)

            # Stack images into a batch.
            img_batch = torch.stack(processed_imgs)

            _, _, [_, _, img_toks_batch] = self._vq_model.encode(img_batch)
            
            if keep_batch:
                img_toks_batch = img_toks_batch.reshape(len(batch_imgs), -1)

            all_tokens_list.extend(img_toks_batch)

        return all_tokens_list

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

    def pil_from_img_toks(self, tokens: torch.Tensor, h_latent_dim=32, w_latent_dim=32) -> PIL.Image:
        emb_dim = self._vq_model.quantize.embedding.weight.shape[-1]
        codebook_entry = self._vq_model.quantize.get_codebook_entry(tokens, (1, h_latent_dim, w_latent_dim, emb_dim))
        pixels = self._vq_model.decode(codebook_entry)
        return self._pil_from_chw_tensor(pixels[0])
    
    def pil_from_img_toks_list(self, tokens_list: list[torch.Tensor], h_latent_dim=32, w_latent_dim=32) -> list[PIL.Image]:
        # Batch processing for multiple token lists.
        batch_tokens = torch.stack(tokens_list)
        emb_dim = self._vq_model.quantize.embedding.weight.shape[-1]
        codebook_entry = self._vq_model.quantize.get_codebook_entry(batch_tokens, (len(tokens_list), h_latent_dim, w_latent_dim, emb_dim))
        pixels_batch = self._vq_model.decode(codebook_entry)

        pil_images = [self._pil_from_chw_tensor(pixels) for pixels in pixels_batch]
        return pil_images

    def latent_embedding_from_pil(self, img: PIL.Image):
        img = self._whiten_transparency(img)

        # Convert to tensor.
        np_img = np.array(img) / 255.0  # Normalize to [0, 1]
        np_img = np_img * 2 - 1  # Scale to [-1, 1]
        img = torch.from_numpy(np_img).permute(2, 0, 1)  # (Channels, Height, Width) format.
        img = img.unsqueeze(0).to(self._vq_model.encoder.conv_in.weight)
        latent_embedding, _, _ = self._vq_model.encode(img)
        return latent_embedding
