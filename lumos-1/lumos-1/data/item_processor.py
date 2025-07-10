import json
import logging
import random
from typing import Dict, List, Union
import traceback

from PIL import Image
import contextlib
import os
import tempfile
import cv2
import torch

from data.convertsation import Conversation
import model.chameleon_vae_ori as chameleon_vae_ori
from xllmx.data.data_reader import read_general
from xllmx.data.item_processor import MMConvItemProcessor

from model.cosmos_tokenizer.networks import TokenizerConfigs
from model.cosmos_tokenizer.utils import (
    get_filepaths,
    get_output_filepath,
    read_video,
    resize_video,
    write_video,
    tensor2numpy
)
from model.cosmos_tokenizer.video_lib import ExtendedCausalVideoTokenizer


logger = logging.getLogger(__name__)

#################################################################################
#        We modified center_crop to avoid randomness in video processing        #
#################################################################################
def center_crop(pil_image, crop_size, use_random_center_crop=True):
    while pil_image.size[0] >= 2 * crop_size[0] and pil_image.size[1] >= 2 * crop_size[1]:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = max(crop_size[0] / pil_image.size[0], crop_size[1] / pil_image.size[1])
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    if use_random_center_crop:
        crop_left = random.randint(0, pil_image.size[0] - crop_size[0])
        crop_upper = random.randint(0, pil_image.size[1] - crop_size[1])
    else:
        crop_left = (pil_image.size[0] - crop_size[0]) // 2
        crop_upper = (pil_image.size[1] - crop_size[1]) // 2

    crop_right = crop_left + crop_size[0]
    crop_lower = crop_upper + crop_size[1]
    return pil_image.crop(box=(crop_left, crop_upper, crop_right, crop_lower))

##############################################################################
#     We modified var_center_crop to avoid randomness in video processing    #
##############################################################################
def var_center_crop(pil_image, crop_size_list, use_random_center_crop=True, random_top_k=1):
    '''
    The function var_center_crop dynamically picks a crop size from a list of candidate crop sizes based on how well their aspect ratios match the original image.
    It calculates a "remainder percentage" for each crop size (indicating how similar its aspect ratio is to the image).
    The top k crop sizes (based on this measure) are sorted, and one is chosen randomly.
    The image is then center-cropped using the selected crop size.
    '''
    w, h = pil_image.size
    rem_percent = [min(cw / w, ch / h) / max(cw / w, ch / h) for cw, ch in crop_size_list]
    crop_size = random.choice(
        sorted(((x, y) for x, y in zip(rem_percent, crop_size_list)), reverse=True)[:random_top_k]
    )[1]
    return center_crop(pil_image, crop_size, use_random_center_crop)


def generate_crop_size_list(num_patches, patch_size, max_ratio=4.0):
    assert max_ratio >= 1.0
    crop_size_list = []
    wp, hp = num_patches, 1
    while wp > 0:
        if max(wp, hp) / min(wp, hp) <= max_ratio:
            crop_size_list.append((wp * patch_size, hp * patch_size))
        if (hp + 1) * wp <= num_patches:
            hp += 1
        else:
            wp -= 1
    return crop_size_list


class FlexARItemProcessor(MMConvItemProcessor):
    image_start_token = "<racm3:break>" # 8197 # fixed tokens for start and end, so can hardcode
    image_end_token = "<eoss>" # 8196 
    full_sub_sep_token = "<reserved08796>" # 8800
    sub_sub_sep_token = "<reserved08797>" # 8801
    sub_skip_token = "<reserved08798>" # 8802
    new_line_token = "<reserved08799>" # 8803
    # grid size token: "<reserved08800>" # 8804     size = 0 * self.patch_size (32) = 0
    # grid size token: "<reserved08801>" # 8805     size = 1 * self.patch_size (32) = 32
    # ......

    def __init__(
        self,
        tokenizer="Alpha-VLLM/Lumina-mGPT-7B-768",
        conv_template=Conversation,
        target_size=512,
    ):

        super().__init__(
            {
                "<|image|>": self.process_image,
            },
            ["<|image|>"],
            tokenizer,
            conv_template,
        )

        self.patch_size = 32
        self.crop_size_list = generate_crop_size_list((target_size // self.patch_size) ** 2, self.patch_size)
        logger.info("List of crop sizes:")
        # target_size = 512, self.crop_size_list = [(1024, 256), (992, 256), (960, 256), (928, 256), (896, 256), (896, 288), (864, 288), (832, 288), (800, 288), (800, 320), (768, 320), (736, 320), (736, 352), (704, 352), (672, 352), (672, 384), (640, 384), (608, 384), (608, 416), (576, 416), (576, 448), (544, 448), (544, 480), (512, 480), (512, 512), (480, 512), (480, 544), (448, 544), (448, 576), (416, 576), (416, 608), (384, 608), (384, 640), (384, 672), (352, 672), (352, 704), (352, 736), (320, 736), (320, 768), (320, 800), (288, 800), (288, 832), (288, 864), (288, 896), (256, 896), (256, 928), (256, 960), (256, 992), (256, 1024)]
        # target_size = 768, self.crop_size_list = [(1536, 384), (1504, 384), (1472, 384), (1440, 384), (1408, 384), (1408, 416), (1376, 416), (1344, 416), (1312, 416), (1312, 448), (1280, 448), (1248, 448), (1216, 448), (1216, 480), (1184, 480), (1152, 480), (1152, 512), (1120, 512), (1088, 512), (1056, 512), (1056, 544), (1024, 544), (1024, 576), (992, 576), (960, 576), (960, 608), (928, 608), (896, 608), (896, 640), (864, 640), (864, 672), (832, 672), (832, 704), (800, 704), (800, 736), (768, 736), (768, 768), (736, 768), (736, 800), (704, 800), (704, 832), (672, 832), (672, 864), (640, 864), (640, 896), (608, 896), (608, 928), (608, 960), (576, 960), (576, 992), (576, 1024), (544, 1024), (544, 1056), (512, 1056), (512, 1088), (512, 1120), (512, 1152), (480, 1152), (480, 1184), (480, 1216), (448, 1216), (448, 1248), (448, 1280), (448, 1312), (416, 1312), (416, 1344), (416, 1376), (416, 1408), (384, 1408), (384, 1440), (384, 1472), (384, 1504), (384, 1536)]

        for i in range(0, len(self.crop_size_list), 6):
            logger.info(" " + "".join([f"{f'{w} x {h}':14s}" for w, h in self.crop_size_list[i : i + 6]]))

        #  todo
        #  currently still use the original image tokenizer provided by Meta rather than transformers
        #  because the transformers implementation does not contain the vae decoder
        #  print(f'List files: {os.listdir("./ckpts/chameleon/tokenizer/text_tokenizer.json")}')
        self.chameleon_ori_vocab = chameleon_vae_ori.VocabInfo(
            json.load(open("./ckpts/chameleon/tokenizer/text_tokenizer.json", encoding="utf8"))["model"]["vocab"]
        )
        self.chameleon_ori_translation = chameleon_vae_ori.VocabTranslation(self.chameleon_ori_vocab, device="cuda")
        self.chameleon_ori_image_tokenizer = chameleon_vae_ori.ImageTokenizer(
            cfg_path="./ckpts/chameleon/tokenizer/vqgan.yaml",
            ckpt_path="./ckpts/chameleon/tokenizer/vqgan.ckpt",
            device="cuda",
        )

    @staticmethod
    def get_n_grids_token(n_grids):
        return f"<reserved{8800 + n_grids:05d}>"

    def token2id(self, token: str) -> int:
        return self.tokenizer.tokenizer.vocab[token]

    @torch.no_grad()
    def process_image(self, image) -> Dict:
        if isinstance(image, Image.Image):
            pass
        else:
            image = Image.open(read_general(image))

        image = var_center_crop(image, crop_size_list=self.crop_size_list)

        w_grids, h_grids = image.size[0] // self.patch_size, image.size[1] // self.patch_size

        image_toks = self.chameleon_ori_translation.convert_img2bp2(
            self.chameleon_ori_image_tokenizer.img_tokens_from_pil(image)
        ).view(-1)
        # self.chameleon_ori_image_tokenizer.img_tokens_from_pil(image): This method processes the image and produces a latent token representation of it, likely using some form of vector quantization from a trained VQGAN model (which is why the vqgan.yaml and vqgan.ckpt files are loaded during initialization).
        # self.chameleon_ori_translation.convert_img2bp2(...): Transforms these latent image tokens into a specific format used by the model (probably converting between different encoding formats).


        full_image_toks = image_toks.reshape(image.size[1] // 16, image.size[0] // 16)
        # reshaped back into a 2D grid with dimensions proportional to the image size, divided by 16. This prepares the tokens for further processing, including adding special tokens to structure the output.
        new_line_id = self.token2id(self.new_line_token)

        full_image_toks = torch.cat(
            (
                full_image_toks,
                torch.ones(image.size[1] // 16, 1, device=full_image_toks.device, dtype=full_image_toks.dtype)
                    * new_line_id,
            ),
            dim=1,
        ).flatten()
        # add a new line token (new_line_id) at the end of each row of image tokens to ensure a structured tokenization format that reflects the grid structure of the image.

        result_toks = [
            self.token2id(self.image_start_token),
            self.token2id(self.get_n_grids_token(h_grids)),   # you can basically know the size of the output image,
            self.token2id(self.get_n_grids_token(w_grids)),   # for example, <reserved{8807}><reserved{8808}> means that img_size = 224x256. 
            *full_image_toks.tolist(),
            self.token2id(self.image_end_token),
        ]

        return {"input_ids": result_toks, "labels": result_toks, "size_wh": (frame.size[0], frame.size[1])}

    def process_item(self, item, training_mode=False, out_flatten=True):
        if not out_flatten:
            return super().process_item(item, training_mode=training_mode)

        if training_mode:
            # breakpoint()
            print('FlexARItemProcess')
            tokens, labels = super().process_item(item, training_mode=training_mode)
            # For image generation:
            # Tokens: [0 (<s>), tokens for sentences, {dict for image}, 8710]
            #       {dict for image} explanation:                            
            #       'input_ids': [8197, ...... , 8196]
            #                    It usually contains several thousand image tokens, like img token range [4, 8195] and special tokens
            #       'labels': it is identical to 'input_ids' since we need to train the whole images
            #       'type': '<|image|>'
            #       'to_predict': True if we need to predict this image
            # Labels: [-100, ...., -100, 65536 (image), 8710], -100 means training

            input_tokens_item = []
            modified_labels_item = []
            for i, (token_or_media, ori_label) in enumerate(zip(tokens, labels)):
                '''
                    Reorganize 'tokens' and 'labels' into a full usable sequence without dicts for media. 
                '''
                if isinstance(token_or_media, int):
                    token = token_or_media
                    input_tokens_item.append(token)
                    modified_labels_item.append(ori_label)
                else:  # If it's a media token (e.g., a dict for image tokens)
                    input_tokens_item += token_or_media["input_ids"]
                    if ori_label <= 0:  # in the prompt part
                        modified_labels_item += [-100] * len(token_or_media["input_ids"])
                        # masked with -100 to avoid computing a loss for those tokens.
                    else:
                        modified_labels_item += token_or_media["labels"]

            return input_tokens_item, modified_labels_item
        else:
            tokens = super().process_item(item, training_mode=training_mode)
            input_tokens_item = []
            for i, token_or_media in enumerate(tokens):
                if isinstance(token_or_media, int):
                    input_tokens_item.append(token_or_media)
                else:
                    input_tokens_item += token_or_media["input_ids"]

            return input_tokens_item

    def decode_image(self, tokens: List[int]) -> Image.Image:
        if tokens[0] == self.token2id(self.image_start_token):
            tokens = tokens[1:]
        if tokens[-1] == self.token2id(self.image_end_token):
            tokens = tokens[:-1]

        h_grids, w_grids = tokens[0] - 8804, tokens[1] - 8804
        tokens = tokens[2:]
        h, w = h_grids * self.patch_size, w_grids * self.patch_size
        h_latent_dim, w_latent_dim = h_grids * 2, w_grids * 2

        for i in range(len(tokens)):
            if (i + 1) % (w_latent_dim + 1) != 0:
                tokens[i] = self.chameleon_ori_translation.bpe2img[tokens[i]]

        assert len(tokens) == h_latent_dim * (w_latent_dim + 1)
        tokens = torch.tensor(tokens, dtype=torch.int64).cuda()

        tokens = tokens.view(h_latent_dim, w_latent_dim + 1)[:, :-1].flatten() # This drops the new_line token "<reserved08799>"

        return self.chameleon_ori_image_tokenizer.pil_from_img_toks(tokens, h_latent_dim, w_latent_dim)



class FlexARItemProcessor2(MMConvItemProcessor):
    image_start_token = "<racm3:break>" # 8197 # fixed tokens for start and end, so can hardcode
    image_end_token = "<eoss>" # 8196 
    full_sub_sep_token = "<reserved08796>" # 8800
    sub_sub_sep_token = "<reserved08797>" # 8801
    sub_skip_token = "<reserved08798>" # 8802
    new_line_token = "<reserved08799>" # 8803
    # grid size token: "<reserved08800>" # 8804     size = 0 * self.patch_size (32) = 0
    # grid size token: "<reserved08801>" # 8805     size = 1 * self.patch_size (32) = 32
    # ......
    sep_token = "<reserved08706>" # 8710

    video_start_token = "<reserved09000>"
    video_end_token = "<reserved09001>"
    # video_new_frame_token = "reserved09002"
    mask_token = "<reserved08999>"

    def __init__(
        self,
        tokenizer="Alpha-VLLM/Lumina-mGPT-7B-768",
        conv_template=Conversation,
        target_size=512,
        target_task_vid=["predict_next_frame"], # TODO: "predict_whole_video".
        target_fps=8,
        duration=4,
        video_frame_batch_size=32,
        inference_mode=False,
        visual_tokenizer="Chameleon",
        cosmos_dtype=None,
    ):  
        if inference_mode:
            transform = {"<|image|>": self.process_image, "<|partial_video|>": self.process_partial_video}
            media_symbols = ["<|image|>", "<|partial_video|>"] 
        else:
            transform = {"<|image|>": self.process_image, "<|video|>": self.process_video,}
            media_symbols = ["<|image|>", "<|video|>"]

        super().__init__(
            transform,
            media_symbols,
            tokenizer,
            conv_template,
        )

        self.patch_size = 32
        self.target_fps = target_fps
        self.duration = duration
        self.video_frame_batch_size = video_frame_batch_size
        self.crop_size_list = generate_crop_size_list((target_size // self.patch_size) ** 2, self.patch_size)
        logger.info("List of crop sizes:")
        # target_size = 512, self.crop_size_list = [(1024, 256), (992, 256), (960, 256), (928, 256), (896, 256), (896, 288), (864, 288), (832, 288), (800, 288), (800, 320), (768, 320), (736, 320), (736, 352), (704, 352), (672, 352), (672, 384), (640, 384), (608, 384), (608, 416), (576, 416), (576, 448), (544, 448), (544, 480), (512, 480), (512, 512), (480, 512), (480, 544), (448, 544), (448, 576), (416, 576), (416, 608), (384, 608), (384, 640), (384, 672), (352, 672), (352, 704), (352, 736), (320, 736), (320, 768), (320, 800), (288, 800), (288, 832), (288, 864), (288, 896), (256, 896), (256, 928), (256, 960), (256, 992), (256, 1024)]
        # target_size = 768, self.crop_size_list = [(1536, 384), (1504, 384), (1472, 384), (1440, 384), (1408, 384), (1408, 416), (1376, 416), (1344, 416), (1312, 416), (1312, 448), (1280, 448), (1248, 448), (1216, 448), (1216, 480), (1184, 480), (1152, 480), (1152, 512), (1120, 512), (1088, 512), (1056, 512), (1056, 544), (1024, 544), (1024, 576), (992, 576), (960, 576), (960, 608), (928, 608), (896, 608), (896, 640), (864, 640), (864, 672), (832, 672), (832, 704), (800, 704), (800, 736), (768, 736), (768, 768), (736, 768), (736, 800), (704, 800), (704, 832), (672, 832), (672, 864), (640, 864), (640, 896), (608, 896), (608, 928), (608, 960), (576, 960), (576, 992), (576, 1024), (544, 1024), (544, 1056), (512, 1056), (512, 1088), (512, 1120), (512, 1152), (480, 1152), (480, 1184), (480, 1216), (448, 1216), (448, 1248), (448, 1280), (448, 1312), (416, 1312), (416, 1344), (416, 1376), (416, 1408), (384, 1408), (384, 1440), (384, 1472), (384, 1504), (384, 1536)]

        for i in range(0, len(self.crop_size_list), 6):
            logger.info(" " + "".join([f"{f'{w} x {h}':14s}" for w, h in self.crop_size_list[i : i + 6]]))

        #  TODO
        #  currently still use the original image tokenizer provided by Meta rather than transformers
        #  because the transformers implementation does not contain the vae decoder
        self.cosmos_dtype = cosmos_dtype
        self.visual_tokenizer = visual_tokenizer
        if visual_tokenizer == "Chameleon":
            self.chameleon_ori_vocab = chameleon_vae_ori.VocabInfo(
                json.load(open("./ckpts/chameleon/tokenizer/text_tokenizer.json", encoding="utf8"))["model"]["vocab"]
            )
            self.chameleon_ori_translation = chameleon_vae_ori.VocabTranslation(self.chameleon_ori_vocab, device="cuda")
            self.chameleon_ori_image_tokenizer = chameleon_vae_ori.ImageTokenizer(
                cfg_path="./ckpts/chameleon/tokenizer/vqgan.yaml",
                ckpt_path="./ckpts/chameleon/tokenizer/vqgan.ckpt",
                device="cuda",
            )
            self.spatial_compression = 16
        elif visual_tokenizer in ["Cosmos-Tokenizer-DV4x8x8"]:
            self.chameleon_vid_vocab = chameleon_vae_ori.VocabInfo(
                json.load(open("./ckpts/cosmos/tokenizer/text_tokenizer.json", encoding="utf8"))["model"]["vocab"]
            )
            self.chameleon_ori_translation = chameleon_vae_ori.VocabTranslation(self.chameleon_vid_vocab, device="cuda")
            
            tokenizer_type = "DV" if "DV" in visual_tokenizer else "CV"
            temporal_compression, spatial_compression = visual_tokenizer.split('DV')[1].split('x')[:2]
            self.temporal_compression, self.spatial_compression = int(temporal_compression), int(spatial_compression)
            tokenizer_config = TokenizerConfigs[tokenizer_type].value
            tokenizer_config.update(dict(spatial_compression=self.spatial_compression))
            tokenizer_config.update(dict(temporal_compression=self.temporal_compression))
            self.cosmos_visual_tokenizer = ExtendedCausalVideoTokenizer(
                checkpoint = None,
                checkpoint_enc = 'ckpts/cosmos/Cosmos-Tokenizer-DV4x8x8/encoder.jit',
                checkpoint_dec = 'ckpts/cosmos/Cosmos-Tokenizer-DV4x8x8/decoder.jit',
                tokenizer_config = tokenizer_config,
                device = 'cuda',
                dtype = self.cosmos_dtype, # "bfloat16",
            )
        else:
            assert False, f"Visual_tokenizer {visual_tokenizer} is not supported."


    @staticmethod
    def get_n_grids_token(n_grids):
        return f"<reserved{8800 + n_grids:05d}>"

    def token2id(self, token: str) -> int:
        return self.tokenizer.tokenizer.vocab[token]

    @torch.no_grad()
    def process_image(self, image) -> Dict:
        if isinstance(image, Image.Image):
            pass
        else:
            image = Image.open(read_general(image))

        image = var_center_crop(image, crop_size_list=self.crop_size_list)

        w_grids, h_grids = image.size[0] // self.patch_size, image.size[1] // self.patch_size

        image_toks = self.chameleon_ori_translation.convert_img2bp2(
            self.chameleon_ori_image_tokenizer.img_tokens_from_pil(image)
        ).view(-1)
        # self.chameleon_ori_image_tokenizer.img_tokens_from_pil(image): This method processes the image and produces a latent token representation of it, likely using some form of vector quantization from a trained VQGAN model (which is why the vqgan.yaml and vqgan.ckpt files are loaded during initialization).
        # self.chameleon_ori_translation.convert_img2bp2(...): Transforms these latent image tokens into a specific format used by the model (probably converting between different encoding formats).


        full_image_toks = image_toks.reshape(image.size[1] // self.spatial_compression, image.size[0] // self.spatial_compression)
        # reshaped back into a 2D grid with dimensions proportional to the image size, divided by 16. This prepares the tokens for further processing, including adding special tokens to structure the output.
        new_line_id = self.token2id(self.new_line_token)

        full_image_toks = torch.cat(
            (
                full_image_toks,
                torch.ones(image.size[1] // self.spatial_compression, 1, device=full_image_toks.device, dtype=full_image_toks.dtype)
                    * new_line_id,
            ),
            dim=1,
        ).flatten()
        # add a new line token (new_line_id) at the end of each row of image tokens to ensure a structured tokenization format that reflects the grid structure of the image.

        result_toks = [
            self.token2id(self.image_start_token),
            self.token2id(self.get_n_grids_token(h_grids)),   # you can basically know the size of the output image,
            self.token2id(self.get_n_grids_token(w_grids)),   # for example, <reserved{8807}><reserved{8808}> means that img_size = 224x256. 
            *full_image_toks.tolist(),
            self.token2id(self.image_end_token),
        ]

        return {"input_ids": result_toks, "labels": result_toks}


    ### process_video V2: Batch processing
    @torch.no_grad()
    def process_video(self, video_frames: Union[str, List]) -> Dict:    # , target_fps: int, duration: int
        """
        Processes a video (either a file path or a list of frames) and converts it into token sequences.
        Ensures the video meets FPS and duration requirements.

        :param video_frames: Either a string (video file path) or a list of PIL.Image objects or file paths.
        :param batch_size: Batch size for processing frames.
        :return: A dictionary containing input token IDs for the entire video and labels.
        """
        target_fps = self.target_fps
        duration = self.duration

        # Check if video_frames is a file path to a video
        if isinstance(video_frames, str):
            if target_fps == 1 and duration == 1:
                # import ipdb
                # ipdb.set_trace()
                video_frames = self.extract_image(video_frames)
            else:
                # Extract frames from the video file with target FPS and duration
                video_frames = self.extract_frames_from_video(video_frames, target_fps=target_fps, duration=duration)

        # If no valid frames are extracted (video too short, i.e., video_frames = []), return an empty dictionary
        if not video_frames:
            return {}

        # Process all frames using batch inference
        processed_frames = []
        for frame in video_frames:
            # Check if the frame is already a PIL.Image object, if not, open it
            if not isinstance(frame, Image.Image):
                frame = Image.open(read_general(frame))

            # Center crop and resize frame
            frame = var_center_crop(
                frame, crop_size_list=self.crop_size_list,
                use_random_center_crop=False, random_top_k=1
            )
            processed_frames.append(frame)
        
        ### Visual token extraction v1
        # # Tokenize the frames using batch inference
        # frame_tokens_list = self.chameleon_ori_image_tokenizer.img_tokens_from_pil_list(
        #     processed_frames, batch_size=self.video_frame_batch_size, keep_batch=False
        # ) # max: 8191, min: 6
        # frame_tokens_list = self.chameleon_ori_translation.convert_img2bp2(frame_tokens_list).view(-1).reshape(len(processed_frames), -1)

        ### Visual token extraction v2
        if self.visual_tokenizer == "Chameleon":
            # Tokenize the frames using batch inference
            frame_tokens_list = self.chameleon_ori_image_tokenizer.img_tokens_from_pil_list(
                processed_frames, batch_size=self.video_frame_batch_size, keep_batch=False
            ) # max: 8191, min: 6
            frame_tokens_list = self.chameleon_ori_translation.convert_img2bp2(frame_tokens_list).view(-1).reshape(len(processed_frames), -1)
        elif "Cosmos-Tokenizer" in self.visual_tokenizer:
            frame_tokens_list = self.cosmos_visual_tokenizer.vid_tokens_from_pil_list(processed_frames)
            frame_tokens_list = self.chameleon_ori_translation.convert_vid2bp2(frame_tokens_list).view(-1).reshape(len(frame_tokens_list), -1)
        else:
            assert False, f"Visual_tokenizer {visual_tokenizer} is not supported."

        all_frame_tokens = []  # This will store the tokens for all frames
        frame_w, frame_h = processed_frames[0].size[0], processed_frames[0].size[1]
        for frame_toks, frame in zip(frame_tokens_list, processed_frames):

            # Get the width and height grids (grid size for the frame)
            w_grids, h_grids = frame_w // self.patch_size, frame_h // self.patch_size

            # Reshape the tokens into a 2D grid (for handling in rows and adding new line tokens) 
            full_frame_toks = frame_toks.reshape(frame_h // self.spatial_compression, frame_w // self.spatial_compression)

            # Add new line tokens at the end of each row of the frame tokens
            new_line_id = self.token2id(self.new_line_token)
            full_frame_toks = torch.cat(
                (
                    full_frame_toks,
                    torch.ones(frame_h // self.spatial_compression, 1, device=full_frame_toks.device, dtype=full_frame_toks.dtype)
                        * new_line_id,
                ),
                dim=1,
            ).flatten()

            # Prepare the frame token list, including special tokens for the grid size
            frame_tokens = [
                self.token2id(self.image_start_token),           # Start token for the frame
                self.token2id(self.get_n_grids_token(h_grids)),  # Grid height (h_grids)
                self.token2id(self.get_n_grids_token(w_grids)),  # Grid width (w_grids)
                *full_frame_toks.tolist(),                       # Tokens for the frame itself
                self.token2id(self.image_end_token),             # End token for the frame
            ]

            # Append the tokens for this frame to the overall video token list
            all_frame_tokens.extend(frame_tokens)

        # Compose frame tokens into a full sequence
        video_tokens = [self.token2id(self.video_start_token)]
        video_tokens += [self.token2id(self.get_n_grids_token(duration)), 
                         self.token2id(self.get_n_grids_token(target_fps))] 
        # duration_token, fps_token
        # for frame_tokens in all_frame_tokens:
        #     video_tokens += frame_tokens
        video_tokens += all_frame_tokens
        video_tokens += [self.token2id(self.video_end_token)]

        # Once all frames are processed, return the token sequence
        return {"input_ids": video_tokens, "labels": video_tokens,
                "size_wh": (frame_w, frame_h)}


    ### process_partial_video V2: Batch processing and code resuing from self.process_video
    @torch.no_grad()
    def process_partial_video(self, video_frames: Union[str, List]) -> Dict:    # , target_fps: int, duration: int
        # Check if video_frames is a file path to a video
        if isinstance(video_frames, str):
            # Extract frames from the video file with target FPS and duration
            video_frames = self.extract_all_frames_from_partial_video(video_frames)

        return self.process_video(video_frames)


    ### New version, support local file and oss file processing
    def extract_frames_from_video(self, video_path: str, target_fps: int, duration: float) -> List[Image.Image]:
        """
        Extracts central frames from a video file and ensures it meets the FPS and duration requirements.
        Note that we extract central frames.
        
        :param video_path: Path to the video file. Can be a local path or OSS path.
        :param target_fps: Target frames per second (FPS).
        :param duration: Desired duration of the video in seconds.
        :return: List of PIL.Image objects representing central frames of the video, or raises an exception if the video is too short.
        """
        # Placeholder for frames to return
        frames = []

        # Extract the suffix from the video path (e.g., .mp4, .avi)
        _, suffix = os.path.splitext(video_path)

        # Context manager to handle video path (OSS or local)
        with {
            True: contextlib.nullcontext(video_path),  # Handle local path
        }[True] as video_data:

            cap = cv2.VideoCapture(video_path)  # Use the local path with VideoCapture

            # Get original FPS and total frames of the video
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_in_video = total_frames / original_fps  # Video's total duration in seconds

            # Calculate the required number of frames for the given duration at target_fps
            required_frames = round(target_fps * duration)

            # If the video is shorter than the required duration, raise an exception
            if duration_in_video < duration:
                cap.release()
                return []
                # raise Exception("Video duration is too short.")
            
            # Frame step to match the target_fps from original_fps
            frame_step = int(original_fps / target_fps)

            # Calculate the middle section start and end frame indices
            middle_frame_index = total_frames // 2
            start_frame = middle_frame_index - (required_frames * frame_step) // 2
            end_frame = start_frame + required_frames * frame_step

            # Ensure start_frame is not negative and end_frame is within total_frames
            start_frame = max(0, start_frame)
            end_frame = min(total_frames, end_frame)

            # Skip to the start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # Extract frames from the video according to the target_fps
            frame_count = start_frame
            while cap.isOpened() and len(frames) < required_frames and frame_count < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                # Only capture the frame if it's at the appropriate step to match the target_fps
                if frame_count % frame_step == 0:
                    # Convert frame from BGR (OpenCV format) to RGB (PIL format)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    frames.append(pil_image)

                frame_count += 1

            cap.release()

            # If we extracted fewer frames than needed, return an empty list
            if len(frames) < required_frames:
                return []
                # raise Exception("The number of frames is not enough.")

            print('required_frames:', required_frames)

            return frames[:required_frames]  # Return only the required number of frames
    
    def extract_image(self, image_path: str)-> List[Image.Image]:
        """
        Extracts the image given an image file.
        
        :param video_path: Path to the video file. Can be a local path or OSS path.
        :return: List of PIL.Image objects representing the image.
        """

        # Placeholder for frames to return
        frames = []

        # Extract the suffix from the video path (e.g., .mp4, .avi)
        _, suffix = os.path.splitext(image_path)

        # Handle local and OSS paths
        with {
            True: contextlib.nullcontext(image_path),  # Handle local path
        }[True] as image_data:

            # Read image from local path
            img = Image.open(image_path)

            img = img.convert("RGB")  # Ensure consistency in color format
            frames.append(img)

        return frames  # Return as a list for consistency with video frame extraction



    def extract_all_frames_from_partial_video(self, video_path: str) -> List[Image.Image]:
        """
        Extracts all frames from a video file.

        :param video_path: Path to the video file. Can be a local path or OSS path.
        :return: List of PIL.Image objects representing all frames of the video.
        """
        # Placeholder for frames to return
        frames = []

        # Extract the suffix from the video path (e.g., .mp4, .avi)
        _, suffix = os.path.splitext(video_path)

        # Context manager to handle video path (OSS or local)
        with {
            True: contextlib.nullcontext(video_path),  # Handle local path
        }[True] as video_data:

            cap = cv2.VideoCapture(video_path)  # Use the local path with VideoCapture

            # Check if the video was successfully opened
            if not cap.isOpened():
                print("Error: Could not open video.")
                return []

            # Extract all frames from the video
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert frame from BGR (OpenCV format) to RGB (PIL format)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)

            # Release the capture object
            cap.release()

        return frames



    def process_item(self, item, training_mode=False, out_flatten=True):
        if not out_flatten:
            return super().process_item(item, training_mode=training_mode)

        if training_mode:
            print('FlexARItemProcess')
            tokens, labels = super().process_item(item, training_mode=training_mode)
            # For image generation:
            # Tokens: [0 (<s>), tokens for sentences, {dict for image}, 8710]
            #       {dict for image} explanation:                            
            #       'input_ids': [8197, ...... , 8196]
            #                    It usually contains several thousand image tokens, like img token range [4, 8195] and special tokens
            #       'labels': it is identical to 'input_ids' since we need to train the whole images
            #       'type': '<|image|>'
            #       'to_predict': True if we need to predict this image}
            # Labels: [-100, ...., -100, 65536 (image), 8710], -100 means training

            input_tokens_item = []
            modified_labels_item = []
            for i, (token_or_media, ori_label) in enumerate(zip(tokens, labels)):
                '''
                    Reorganize 'tokens' and 'labels' into a full usable sequence without dicts for media. 
                '''
                if isinstance(token_or_media, int):
                    token = token_or_media
                    input_tokens_item.append(token)
                    modified_labels_item.append(ori_label)
                else:  # If it's a media token (e.g., a dict for image tokens)
                    # print(f"token_or_media {token_or_media}") 测试！！！
                    if "input_ids" in token_or_media and "labels" in token_or_media:
                        input_tokens_item += token_or_media["input_ids"]
                        if ori_label <= 0:  # in the prompt part
                            modified_labels_item += [-100] * len(token_or_media["input_ids"])
                            # masked with -100 to avoid computing a loss for those tokens.
                        else:
                            modified_labels_item += token_or_media["labels"]
                    else:
                        return [], []

            return input_tokens_item, modified_labels_item
        else:
            tokens = super().process_item(item, training_mode=training_mode)
            input_tokens_item = []
            for i, token_or_media in enumerate(tokens):
                if isinstance(token_or_media, int):
                    input_tokens_item.append(token_or_media)
                else:
                    input_tokens_item += token_or_media["input_ids"]

            return input_tokens_item

    def decode_image(self, tokens: List[int]) -> Image.Image:
        if tokens[0] == self.token2id(self.image_start_token):
            tokens = tokens[1:]
        if tokens[-1] == self.token2id(self.image_end_token):
            tokens = tokens[:-1]

        h_grids, w_grids = tokens[0] - 8804, tokens[1] - 8804
        tokens = tokens[2:]
        h, w = h_grids * self.patch_size, w_grids * self.patch_size
        h_latent_dim, w_latent_dim = h_grids * 2, w_grids * 2

        for i in range(len(tokens)):
            if (i + 1) % (w_latent_dim + 1) != 0:
                tokens[i] = self.chameleon_ori_translation.bpe2img[tokens[i]]

        assert len(tokens) == h_latent_dim * (w_latent_dim + 1)
        tokens = torch.tensor(tokens, dtype=torch.int64).cuda()

        tokens = tokens.view(h_latent_dim, w_latent_dim + 1)[:, :-1].flatten() # This drops the new_line token "<reserved08799>"

        return self.chameleon_ori_image_tokenizer.pil_from_img_toks(tokens, h_latent_dim, w_latent_dim)
    
    def decode_video(self, tokens: List[List[int]], spatial_compression: int) -> List[Image.Image]:
        image_start_token = self.token2id(self.image_start_token)
        image_end_token = self.token2id(self.image_end_token)
        h_grids, w_grids = tokens[0][1] - 8804, tokens[0][2] - 8804
        h, w = h_grids * self.patch_size, w_grids * self.patch_size
        h_latent_dim, w_latent_dim = h_grids * self.patch_size//spatial_compression, w_grids * self.patch_size//spatial_compression

        ### Create a look-up table for the dict, aiming for fast conversion.
        # Convert the dictionary to a PyTorch lookup table
        max_key = max(self.chameleon_ori_translation.bpe2vid.keys())  # Find the maximum key in the dictionary
        bpe2vid_lookup_table = torch.full((max_key + 1,), -1, dtype=torch.int32, device="cuda")  # Initialize with a default value
        for key, value in self.chameleon_ori_translation.bpe2vid.items():
            bpe2vid_lookup_table[key] = value
        
        ### Elimiate the image start and end tokens.
        latent_tokens = []
        for one_latent_token in tokens:
            if one_latent_token[0] == image_start_token:
                one_latent_token = one_latent_token[1:]
            if one_latent_token[-1] == image_end_token:
                one_latent_token = one_latent_token[:-1]
            latent_tokens.append(one_latent_token[2:])

        ### Eliminate the new_line_token
        latent_tokens = torch.tensor(latent_tokens, device="cuda")
        num_latent_tokens = latent_tokens.shape[0]
        latent_tokens = latent_tokens.reshape(num_latent_tokens, h_latent_dim, w_latent_dim + 1)[:,:,:-1].contiguous()
        
        ### Look-up table conversion
        flattened_latent_tokens = latent_tokens.view(-1)  # Flatten to 1D
        converted_flattened = bpe2vid_lookup_table[flattened_latent_tokens]  # Apply the lookup table
        latent_tokens = converted_flattened.view(1, num_latent_tokens, h_latent_dim, w_latent_dim)  # Reshape back to original shape
        
        ### Decode to rgb space
        video_rgb_tensor = self.cosmos_visual_tokenizer.decode(latent_tokens)[0].permute(1,0,2,3).to(torch.float32)

        return [self.cosmos_visual_tokenizer._pil_from_chw_tensor(i) for i in video_rgb_tensor]