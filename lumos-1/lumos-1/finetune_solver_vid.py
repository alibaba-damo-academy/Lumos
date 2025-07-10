import pickle
from typing import List, Tuple
import time
import random
import json
import os
import math
import sys
import contextlib
import datetime
from copy import deepcopy

from fairscale.nn.model_parallel import initialize as fs_init
from accelerate import init_empty_weights
import torch

from xllmx.data.dataset import FinetuneConversationDataset
from xllmx.data.sampler import FinetuneDistSampler
from model import ChameleonXLLMXConfig, ChameleonXLLMXForConditionalGeneration, get_mask_chedule
from xllmx.data.item_processor import ItemProcessorBase
from xllmx.solvers.finetune import FinetuneSolverBase
import xllmx.util.misc as misc
import xllmx.util.lr_sched as lr_sched
import xllmx.util as util
from model.utils import read_vocab_size


class ItemProcessor(ItemProcessorBase):
    def process_item(self, data_item: dict, training_mode=False) -> Tuple[List, List]:
        assert training_mode

        if "token" in data_item and "label" in data_item:
            data_item = data_item
        else:
            assert "file" in data_item
            with open(data_item["file"], "rb") as f:
                data_item = pickle.load(f)

        tokens = data_item["token"]
        labels = data_item["label"]

        assert len(tokens) == len(labels)

        return tokens, labels

    def predict_item_token_length(self, data_item: dict) -> int:
        if "token" in data_item:
            return len(data_item["token"])
        elif "len" in data_item:
            return data_item["len"]
        else:
            raise ValueError()


class Solver2(FinetuneSolverBase):
    def __init__(self, args):
        """
        Initializes the Solver2 class by calling the parent class constructor and setting up additional attributes.
        
        Parameters:
            args (argparse.Namespace): Command-line arguments parsed by the argument parser.
        """
        super().__init__(args)  # Call the superclass (FinetuneSolverBase) init method

        if args.no_resume_iterations:
            self.start_epoch = 0
            self.start_iter = 0
            self.logger.info(f"Setting resumed epochs and iters both to 0.")


        with open('data/num2tokens.json', 'r') as f:
            self.num2tokens = json.load(f)
        
        ### hardcoded video info
        self.data_video_fps = self.args.data_video_fps
        self.data_video_frames = self.args.data_video_frames
        self.data_video_duration = self.data_video_frames / self.data_video_fps # 4 by default
        
        ### hard-coded video_placeholder_token
        self.video_placeholder_token = {
            "<video_resolution_placeholder>": 65538,
            "<video_frames_placeholder>": 65539,
            "<video_fps_placeholder>": 65540
        }
        self.special_token = {
            "image_start_token": 8197,  # "<racm3:break>" # fixed tokens for start and end, so can hardcode
            "image_end_token": 8196,    # "<eoss>" 
            "new_line_token": 8803,     # "<reserved08799>"
            "sep_token": 8710,          # "<reserved08706>"
            "video_start_token": 9004,  # "<reserved09000>"
            "video_end_token": 9005,    # "<reserved09001>"
            "mask_token":9003,          # "<reserved08999>"
        }


        # self.video_placeholder_token = {65538: "<video_resolution_placeholder>",
        #                                 65539: "<video_frames_placeholder>",
        #                                 65540: "<video_fps_placeholder>"}
        # This might change if we have multiple data types like <|image|>, <|video|> and <|partial_video|>

        if self.args.run_eval:
            self.dataset_eval, self.sampler_eval, self.dataloader_eval = self.build_eval_data()
            assert len(self.args.eval_mode) > 0, "To run evaluation, you must indicate at least one eval_mode."

            n_update_per_eval = (len(self.dataloader_train) // (self.args.eval_in_epoch+1)) // self.args.accum_iter
            self.update_eval_list = [n_update_per_eval*(i+1) for i in range(self.args.eval_in_epoch)]
            self.logger.info(f"The update index list {self.update_eval_list} (accum_iter is {self.args.accum_iter})")
        
        # Support for mask AR
        self.mask_schedule = None
        if self.args.MaskedAR:
            # mask_args = self.args.mask_schedule.get("params", {})
            mask_args = {} # TODO add args if it is used
            self.mask_schedule = get_mask_chedule(self.args.mask_schedule, **mask_args)
        
        if "Cosmos-Tokenizer" in self.args.visual_tokenizer:
            self.temporal_compression, self.spatial_compression = [int(i) for i in self.args.visual_tokenizer.split('DV')[1].split('x')[:2]]
        
        # Support for image-video joint training
        if self.args.joint_img_video:
            # In this case, we cannot use gradient accumulation unless modification to the code is implemented.
            if self.args.accum_iter != 1:
                raise NotImplementedError(
                    f"Currently, this code base does not support the joint use of " \
                    f"image-video joint training and grad accumulation."
                )
            
            self.dataset_train_img, self.sampler_train_img, self.dataloader_train_img = self.build_image_data()

    @classmethod
    def get_args_parser(cls):
        parser = super().get_args_parser()
        # task-specific parameters
        parser.add_argument("--max_seq_len", default=4096, type=int, help="max token length")
        parser.add_argument("--mask_image_logits", default=True)
        parser.add_argument("--unmask_image_logits", action="store_false", dest="mask_image_logits")
        parser.add_argument("--dropout", type=float, default=0.0)
        parser.add_argument("--z_loss_weight", type=float, default=0.0)
        parser.add_argument("--model_size", type=str, default="7B", choices=["7B", "34B", "1B", "0.5B", "0.5B_XMMRoPE", "1B_MMRoPE", "3B_MMRoPE"])


        ### Support for video pre-training
        parser.add_argument("--pretrain_task", nargs='+', default=[], 
                                help="List of pretraining tasks")
        parser.add_argument("--video_fps", nargs='+', default=[4], type=int,
                                help="List of video fps choices")
        parser.add_argument("--video_duration", nargs='+', default=[4], type=int,
                                help="List of video duration choices")
        parser.add_argument("--cfg_mode", nargs='+', default=[], type=str,
                                help="List of cfg mode")
        parser.add_argument("--masking_mode", nargs='+', default=[], type=str,
                                help="List of masking mode")
        parser.add_argument("--text_to_video_prob", type=float, default=0.5)
        parser.add_argument("--eval_data_config", default=None, type=str, help="eval data config path")
        parser.add_argument("--run_eval", action="store_true", default=False)
        parser.add_argument("--eval_in_epoch", type=int, default=None)
        parser.add_argument("--eval_mode", nargs='+', default=[], type=str, help="List of eval mode")
        parser.add_argument("--data_video_fps", default=None, type=int, help="data_video_fps of the pre-tokenized data")
        parser.add_argument("--data_video_frames", default=None, type=int, help="data_video_frames of the pre-tokenized data")


        ### Support for mask AR
        parser.add_argument("--MaskedAR", action="store_true", default=False)
        parser.add_argument("--mask_schedule", type=str, default="cosine", choices=["cosine", "linear", "sigmoid"])
        parser.add_argument("--min_masking_rate", type=float, default=0.0)
        parser.add_argument("--mask_contiguous_region_prob", type=float, default=None)
        parser.add_argument("--noise_type", type=str, default="mask", choices=["mask", "random_replace"])
        parser.add_argument("--predict_all_tokens", action="store_true", default=False)
        parser.add_argument(
            "--frame_closs_recorder", action="store_true", default=False,
            help="Record frame-wise closs for MaskedAR."
        )
        # parser.add_argument("--mask_tube", action="store_true", default=False,
        #                         help="Spatio-temporal tube masking rather than random masking for all frames.")
        parser.add_argument("--mask_type", type=str, default="random", choices=["random", "tube", "tubeDecay", "DiffForcing"])
        parser.add_argument("--train_loss", type=str, default="CE", choices=["CE", "Focal", "CEDecay", "CEChunked"])
        parser.add_argument("--decay_start_coef", type=float, default=None)
        parser.add_argument("--no_ntp_loss", action="store_false", default=True, dest="compute_ntp_loss")
        parser.add_argument(
            "--train_with_slim_lm_head", action="store_true", default=False,
            help="Use a slim lm_head to save some memory by utilizing only the visual part of the lm_head. This is used in training only."
        )
        parser.add_argument(
            "--train_with_vis_tok", action="store_true", default=False,
            help="We only train with the visual tokens, which could save memory."
        )
        parser.add_argument("--vis_tok_start", type=int, default=65536)


        ### Support for COSMOS tokenizer
        parser.add_argument("--visual_tokenizer", type=str, default="Chameleon", choices=["Chameleon", "Cosmos-Tokenizer-DV4x8x8"])
        parser.add_argument(
            "--vocab_size", type=int, default=65536, choices=[65536, 129536],
            help="Different visual tokenizers have different vocab sizes."
        )
        # parser.add_argument("--attn_implementation", type=str, default="sdpa", choices=["sdpa", "flash_attention_2"])
        

        ### Support image-video joint training
        parser.add_argument("--joint_img_video", action="store_true", default=False)
        parser.add_argument(
            "--img_batch_size", default=8, type=int,
            help="Batch size per GPU (effective batch size is img_batch_size * accum_iter * # gpus",
        )
        parser.add_argument("--img_data_config", default=None, type=str, help="img data config path")
        parser.add_argument(
            "--video_iter_per_img_iter", default=4, type=int,
            help="Number of video iterations per image iteration",
        )
        parser.add_argument("--no_resume_metric_logger", action="store_true", default=False)
        parser.add_argument("--no_resume_iterations", action="store_true", default=False)
        


        return parser

    def _model_func(
        self,
        init_from: str,
    ) -> (ChameleonXLLMXForConditionalGeneration, None):
        """
            Instantiate and configure a model based on the given initialization source.
            The model is instantiated only on rank0, with other ranks receiving the model weights 
            during the Fully Sharded Data Parallel (FSDP) wrapping process.

            Args:
                init_from (str): The path or identifier for the pre-trained model to initialize from.

            Returns:
                tuple: A tuple containing:
                    - model (ChameleonXLLMXForConditionalGeneration): The model instantiated with the configuration.
                    - None: Placeholder for future use (currently returns None).
        """
        # Only instantiate the model on rank0
        # Other ranks will receive the model weights from rank0 during FSDP wrapping (through `sync_module_states`)
        # See https://github.com/pytorch/pytorch/issues/105840

        init_from_vocab_size = read_vocab_size(init_from)
        self.logger.info(f'The vocab size in {init_from} is {init_from_vocab_size}.')

        if self.dp_rank == 0:
            ### Instantiate the model using pre-trained weights on rank0
            if init_from_vocab_size == self.args.vocab_size:
                ### Checkpoints finetuned by us.
                model = ChameleonXLLMXForConditionalGeneration.from_pretrained(
                    init_from,
                    max_position_embeddings=self.args.max_seq_len,
                    mask_image_logits=self.args.mask_image_logits,
                    dropout=self.args.dropout,
                    z_loss_weight=self.args.z_loss_weight,
                    torch_dtype=torch.bfloat16,
                    device_map="cpu",
                    vocab_size=self.args.vocab_size,
                    # _attn_implementation="flash_attention_2",
                )
            elif init_from_vocab_size == 65536 and init_from_vocab_size < self.args.vocab_size:
                ### Chameleon pre-trained checkpoints
                model = ChameleonXLLMXForConditionalGeneration.from_pretrained(
                    init_from,
                    max_position_embeddings=self.args.max_seq_len,
                    mask_image_logits=self.args.mask_image_logits,
                    dropout=self.args.dropout,
                    z_loss_weight=self.args.z_loss_weight,
                    torch_dtype=torch.bfloat16,
                    device_map="cpu",
                    # _attn_implementation="flash_attention_2",
                )

                ### Replace embed_tokens.
                current_embed_tokens = model.get_input_embeddings()
                pad_token_id = model.config.pad_token_id if model.config.pad_token_id is not None else 0
                new_embeddings = torch.nn.Embedding(
                    self.args.vocab_size, 
                    current_embed_tokens.embedding_dim,
                    model.config.pad_token_id,
                )
                self.logger.info(f'model.config.pad_token_id {model.config.pad_token_id}.')
                # NOTE: Use padding_idx to ensure it doesn't contribute to the gradient
                # However, in this implementation, pad_token_id is None, maybe because the repo sets the label of the pad token to -100.
                
                # Copy the original embeddings for the first 65536 tokens
                new_embeddings.weight.data[:65536] = current_embed_tokens.weight.data
                # Update the model's input embeddings
                model.set_input_embeddings(new_embeddings)
                # Update the model's vocab_size
                model.vocab_size = self.args.vocab_size
                model.config.vocab_size = self.args.vocab_size  # Update the config object as well

                ### Replace lm_head.
                current_lm_head = model.get_output_embeddings()
                new_lm_head = torch.nn.Linear(current_lm_head.weight.shape[1], self.args.vocab_size, bias=False) # [129536, 4096]
                new_lm_head.weight.data[:65536] = current_lm_head.weight.data
                model.set_output_embeddings(new_lm_head)
                
            else:
                raise NotImplementedError(f"Unsupported configuration: init_from_vocab_size={init_from_vocab_size}, target_vocab_size={self.args.vocab_size}")

        else:
            # On other ranks, use an empty weight configuration to create the model (for FSDP wrapping)
            if init_from_vocab_size == self.args.vocab_size:
                ### Checkpoints finetuned by us.
                with init_empty_weights():
                    config = ChameleonXLLMXConfig.from_pretrained(
                        init_from,
                        max_position_embeddings=self.args.max_seq_len,
                        mask_image_logits=self.args.mask_image_logits,
                        dropout=self.args.dropout,
                        z_loss_weight=self.args.z_loss_weight,
                        torch_dtype=torch.bfloat16,
                        vocab_size=self.args.vocab_size,
                        # _attn_implementation="flash_attention_2",
                    )
                    model = ChameleonXLLMXForConditionalGeneration(config)
            
            elif init_from_vocab_size == 65536 and init_from_vocab_size < self.args.vocab_size:
                ### Chameleon pre-trained checkpoints
                with init_empty_weights():
                    config = ChameleonXLLMXConfig.from_pretrained(
                        init_from,
                        max_position_embeddings=self.args.max_seq_len,
                        mask_image_logits=self.args.mask_image_logits,
                        dropout=self.args.dropout,
                        z_loss_weight=self.args.z_loss_weight,
                        torch_dtype=torch.bfloat16,
                        # _attn_implementation="flash_attention_2",
                    )
                    model = ChameleonXLLMXForConditionalGeneration(config)

                    ### Replace embed_tokens.
                    current_embed_tokens = model.get_input_embeddings()
                    pad_token_id = model.config.pad_token_id if model.config.pad_token_id is not None else 0
                    new_embeddings = torch.nn.Embedding(
                        self.args.vocab_size, 
                        current_embed_tokens.embedding_dim,
                        padding_idx=pad_token_id,
                    )
                    # NOTE: Use padding_idx to ensure it doesn't contribute to the gradient
                    # However, in this implementation, pad_token_id is None, maybe because we set the label of the pad token to -100.
                    
                    # Update the model's input embeddings
                    model.set_input_embeddings(new_embeddings)
                    # Update the model's vocab_size
                    model.vocab_size = self.args.vocab_size
                    model.config.vocab_size = self.args.vocab_size  # Update the config object as well

                    ### Replace lm_head.
                    current_lm_head = model.get_output_embeddings()
                    new_lm_head = torch.nn.Linear(current_lm_head.weight.shape[1], self.args.vocab_size, bias=False) # [129536, 4096]
                    model.set_output_embeddings(new_lm_head)
            
            else:
                raise NotImplementedError(f"Unsupported configuration: init_from_vocab_size={init_from_vocab_size}, target_vocab_size={self.args.vocab_size}")
            

        del model.model.vqmodel

        return model, None

    def _item_processor_func(self) -> ItemProcessorBase:
        return ItemProcessor()

    def _make_and_save_starting_point(self, save_path: str) -> None:

        pretrained_name = {
            "0.5B": "/mnt/workspace/workgroup/hangjie.yhj/code/Lumina-mGPT-main/huggingface_weights/Chameleon_0.5B_mGPT",
            "0.5B_XMMRoPE": "/mnt/workspace/workgroup/hangjie.yhj/code/Lumina-mGPT-main/huggingface_weights/Chameleon_0.5B_mGPT_XMMRoPE",
            "1B": "/mnt/workspace/workgroup/hangjie.yhj/code/Lumina-mGPT-main/huggingface_weights/Chameleon_1B_mGPT",
            "1B_MMRoPE": "/mnt/workspace/workgroup/hangjie.yhj/code/Lumina-mGPT-main/huggingface_weights/Chameleon_1B_mGPT_MMRoPE",
            "3B_MMRoPE": "/mnt/workspace/workgroup/hangjie.yhj/code/Lumina-mGPT-main/huggingface_weights/Chameleon_3B_mGPT_MMRoPE",
            "7B": "Alpha-VLLM/Chameleon_7B_mGPT", # "/mnt/workspace/workgroup/hangjie.yhj/code/Lumina-mGPT-main/huggingface_weights/Chameleon_7B_mGPT",  # Any starting_point file_path should work.
            "34B": "Alpha-VLLM/Chameleon_34B_mGPT",
        }[self.args.model_size]

        
        if self.args.model_size in ["1B", "3B_MMRoPE", "7B", "1B_MMRoPE"]:
            model = ChameleonXLLMXForConditionalGeneration.from_pretrained(
                pretrained_name,
                max_position_embeddings=self.args.max_seq_len,
                mask_image_logits=self.args.mask_image_logits,
                dropout=self.args.dropout,
                z_loss_weight=self.args.z_loss_weight,
                torch_dtype=torch.bfloat16,
                device_map=None if "1B" in self.args.model_size or "3B" in self.args.model_size else "cpu",
                local_files_only=True, # This is for easy usage.
                ignore_mismatched_sizes=True if "1B" in self.args.model_size or "3B" in self.args.model_size else False,
            )
        else:
            config = ChameleonXLLMXConfig.from_pretrained(
                pretrained_name,
                max_position_embeddings=self.args.max_seq_len,
                mask_image_logits=self.args.mask_image_logits,
                dropout=self.args.dropout,
                z_loss_weight=self.args.z_loss_weight,
                torch_dtype=torch.bfloat16,
                device_map=None,
                local_files_only=True, # This is for easy usage.
            )
            model = ChameleonXLLMXForConditionalGeneration(config)

        image_tokens = model.model.vocabulary_mapping.image_tokens
        model.lm_head.weight.data[image_tokens] = torch.zeros_like(model.lm_head.weight.data[image_tokens])

        model.save_pretrained(save_path, max_shard_size="10GB")

    
    ### The v3 run function that supports image-video joint training resume.
    def run(self):
        ################################### Misc #########################################
        self.logger.info(f"Start training for {self.args.epochs} epochs")
        start_time = time.time()
        self.all_fps_duration = self.define_fps_duration_combinations()
        # self.logger.info(
        #     f"All possible RGB fps and duration combinations {self.all_fps_duration[0]}.\n"\
        #     f"All possible latent fps and duration combinations {self.all_fps_duration[1]}.\n"
        # )
        self.logger.info(f"All possible RGB and latent fps and duration combinations {self.all_fps_duration}.\n")

        if self.args.no_resume_metric_logger:
            self.metric_logger_to_resume = None
            self.logger.info("Setting metric_logger to None before training start...")

        ############################ Training function selection #########################
        function_name = "train_one_epoch_joint" if self.args.joint_img_video else "train_one_epoch"
        train_one_epoch_func = getattr(self, function_name)
        self.logger.info(f"Training function: {train_one_epoch_func}\n")

        ####################################### Training  ################################
        for epoch in range(self.start_epoch, self.args.epochs):
            if self.args.joint_img_video: 
                if self.start_iter > 0: # NOTE: This means that this is a resumed training for jointing training.
                    # Consecutive steps of each modality
                    num_image_iters = len(self.dataloader_train_img)  # the integer length
                    video_steps = self.args.video_iter_per_img_iter  # e.g. 4
                    image_steps = 1
                    cycle_steps = image_steps + video_steps           
                    start_iter_vid = self.start_iter // cycle_steps * video_steps
                    start_iter_img = (self.start_iter // cycle_steps * image_steps) % num_image_iters
                else:
                    start_iter_vid = self.start_iter
                    start_iter_img = self.start_iter
            
            if self.args.joint_img_video: 
                self.dataloader_train.sampler.set_epoch(epoch, start_iter_vid)  # todo rename set_epoch
                self.dataloader_train_img.sampler.set_epoch(epoch, start_iter_img)  # todo rename set_epoch
            else:
                self.dataloader_train.sampler.set_epoch(epoch, self.start_iter)  # todo rename set_epoch
            
            if self.args.run_eval:
                self.dataloader_eval.sampler.set_epoch(epoch, self.start_iter)  # todo rename set_epoch
                if epoch == 0: # Eval before start
                    for eval_one_mode in self.args.eval_mode:
                        eval_stats = self.eval_one_epoch(
                            epoch,
                            self.start_iter,
                            eval_mode=eval_one_mode,
                            log_writer=self.log_writer,
                            metric_logger=self.metric_logger_to_resume,
                        )

            train_stats = train_one_epoch_func(
                epoch,
                self.start_iter,
                log_writer=self.log_writer,
                metric_logger=self.metric_logger_to_resume,
            )

            if self.args.run_eval:
                for eval_one_mode in self.args.eval_mode:
                    eval_stats = self.eval_one_epoch(
                        epoch,
                        self.start_iter,
                        eval_mode=eval_one_mode,
                        log_writer=self.log_writer,
                        metric_logger=self.metric_logger_to_resume,
                    )

            if epoch % self.args.save_interval == 0 or epoch + 1 == self.args.epochs:
                util.ckpt.save(
                    self.args.output_dir,
                    self.global_rank == 0,
                    self.model,
                    self.optimizer,
                    self.tokenizer,
                    self.args,
                    epoch=epoch,
                    max_keep=self.args.ckpt_max_keep,
                )

            log_stats = {**{f"train_{k}": v for k, v in train_stats.items()}, "epoch": epoch}

            if self.global_rank == 0:
                if self.log_writer is not None:
                    self.log_writer.flush()
                with open(os.path.join(self.args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

            self.start_iter = 0
            self.metric_logger_to_resume = None

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.logger.info("Training time {}".format(total_time_str))
      

    ### V3: support maskAR based on V2
    def train_one_epoch(
        self,
        epoch: int,
        start_iter: int,
        log_writer=None,
        metric_logger=None,
    ):
        self.model.train(True)
        if metric_logger is None:
            metric_logger = misc.MetricLogger(delimiter="  ")
            metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))

        header = "Epoch: [{}]".format(epoch)
        print_freq = 10  # todo arg

        accum_iter = self.args.accum_iter
        accum_counter = 0

        self.optimizer.zero_grad()
        for data_iter_step, batch_data in enumerate(
            metric_logger.log_every(
                self.dataloader_train,
                print_freq,
                header,
                start_iter,
                self.args.batch_size * fs_init.get_data_parallel_world_size(),
            ),
            start=start_iter,
        ):  
            # ### For debugging
            # if data_iter_step>=10:
            #     break

            accum_counter = (accum_counter + 1) % accum_iter
            is_gradient_accumulation_boundary = accum_counter == 0

            examples, labels = batch_data
            if is_gradient_accumulation_boundary or data_iter_step == start_iter:
                lr_sched.adjust_learning_rate_epoch(
                    self.optimizer, data_iter_step / len(self.dataloader_train) + epoch, self.args
                )
            
            if "predict_video" in self.args.pretrain_task:
                '''
                    We need to replace the special tokens in the token list (i.e., <video_resolution_placeholder>, <video_frames_placeholder>, <video_fps_placeholder>)
                        with specific tokens.
                '''     

                fps_duration = random.choice(self.all_fps_duration)
                for ex, la in zip(examples, labels):
                    # if -100 in labels: 
                    #     # This means that we do not do the classifier-free guidance drop.
                    #     # If we do the drop, then 65539 and 65540 will be dropped.
                    ex = self.replace_with_multiple(
                        lst=ex, to_replace=self.video_placeholder_token["<video_frames_placeholder>"], # 65539 
                        new_tokens=self.num2tokens[str(fps_duration[0][0]*fps_duration[0][1])], labels=False
                    )
                    ex = self.replace_with_multiple(
                        lst=ex, to_replace=self.video_placeholder_token["<video_fps_placeholder>"], # 65540 
                        new_tokens=self.num2tokens[str(fps_duration[0][0])], labels=False
                    )
                    
                    la = self.replace_with_multiple(
                        lst=la, to_replace=self.video_placeholder_token["<video_frames_placeholder>"], # 65539, 
                        new_tokens=self.num2tokens[str(fps_duration[0][0]*fps_duration[0][1])], labels=True
                    )
                    la = self.replace_with_multiple(
                        lst=la, to_replace=self.video_placeholder_token["<video_fps_placeholder>"], # 65540 
                        new_tokens=self.num2tokens[str(fps_duration[0][0])], labels=True
                    )

                ### Masking for variable length training
                # Masks the initial non-visible frames, and drops frame tokens based on the given FPS and duration.
                examples, labels = self.mask_and_drop_video_random(
                    examples, 
                    labels, 
                    fps_duration[1],
                    masking_mode = self.args.masking_mode,
                    text_to_video_prob = self.args.text_to_video_prob,
                )

                ### Replace the video duration and fps tokens in the video token sequence (not text sequence).  
                examples, labels = self.update_video_duration_fps_token_in_video_seq(examples, labels, fps_duration[0])

                ### NOTE: to supoport for video classifier-free gudiance (CFG) training
                for el_idx, (ex, la) in enumerate(zip(examples, labels)):
                    if ex[-2] == la[-2] == 9005 and ex.count(9005) == 1 and ex.count(9004) == 1:  # video generation data
                        if random.random() < 0.1:
                        # if True: # for debugging
                            # print(f"Before cfg drop    ex {len(ex)} la {len(la)}")

                            # Randomly choose between dropping text only or dropping both text and masked video
                            cfg_mode = random.choice(self.args.cfg_mode)
                            # print(f"cfg_mode {cfg_mode}")

                            if cfg_mode == "text_and_video":
                                ### Classifier-free guidance with text and masked video
                                examples[el_idx] = labels[el_idx] = [_ for _ in la[:-1] if _ != -100]
                            
                            elif cfg_mode == "text_only":
                                ### Classifier-free guidance with only text
                                video_start_idx = ex.index(9004) # index only finds the first one.
                                examples[el_idx] = [e for e_idx, e in enumerate(ex) if e_idx>=video_start_idx]
                                labels[el_idx] = [l for l_idx, l in enumerate(la) if l_idx>=video_start_idx]
                            
                            else:
                                assert False, f"{cfg_mode} not implemented."
    

                # image_start_token = "<racm3:break>" # 8197 # fixed tokens for start and end, so can hardcode
                # image_end_token = "<eoss>" # 8196 
                # full_sub_sep_token = "<reserved08796>" # 8800
                # sub_sub_sep_token = "<reserved08797>" # 8801
                # sub_skip_token = "<reserved08798>" # 8802
                # new_line_token = "<reserved08799>" # 8803
                # # grid size token: "<reserved08800>" # 8804     size = 0 * self.patch_size (32) = 0
                # # grid size token: "<reserved08801>" # 8805     size = 1 * self.patch_size (32) = 32
                # video_start_token = "<reserved09000>" # 9004
                # video_end_token = "<reserved09001>" # 9005
                # sep_token = "<reserved08706>" # 8710

            additional_infer_dict = {'cfg':self.args , 'mask_schedule': self.mask_schedule, 'special_token':self.special_token}
            with {
                "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
                "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
                "fp32": contextlib.nullcontext(),
                "tf32": contextlib.nullcontext(),
            }[self.args.precision]:
                c_loss, additional_loss_dict = self.model(examples, labels, **additional_infer_dict)
            
            loss = c_loss
            for add_loss, weight in additional_loss_dict.values():
                loss = loss + add_loss * weight
            loss_value = loss.item()
            c_loss_value = c_loss.item()
            if not math.isfinite(loss_value):
                self.logger.error("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            effective_loss = loss / accum_iter

            with (
                self.model.no_sync()
                if self.args.data_parallel in ["sdp", "hsdp"] and not is_gradient_accumulation_boundary
                else contextlib.nullcontext()
            ):
                effective_loss.backward()

            if is_gradient_accumulation_boundary:
                grad_norm = self.model.clip_grad_norm_(max_norm=self.args.clip_grad)
                metric_logger.update(grad_norm=grad_norm)

                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                # self.logger.info("Param update.")

            torch.cuda.synchronize()

            metric_logger.update(closs=c_loss_value)
            metric_logger.update(**{key: val[0].item() for key, val in additional_loss_dict.items()})
            lr = self.optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)

            for metric_name, metric in metric_logger.meters.items():
                metric_value = metric.value
                metric_value = util.dist.all_reduce_mean(metric_value)
                if log_writer is not None:
                    log_writer.add_scalar(
                        metric_name, metric_value, data_iter_step + len(self.dataloader_train) * epoch
                    )

            # save within epoch
            n_update_per_save = self.args.save_iteration_interval // accum_iter
            if (
                is_gradient_accumulation_boundary and ((data_iter_step + 1) // accum_iter) % n_update_per_save == 0
            ) or (data_iter_step + 1 == accum_iter and epoch == 0):
                util.ckpt.save(
                    self.args.output_dir,
                    self.global_rank == 0,
                    self.model,
                    self.optimizer,
                    self.tokenizer,
                    self.args,
                    epoch=epoch,
                    iteration=data_iter_step,
                    additional_rank_specific={
                        "metric_logger": metric_logger,
                    },
                    max_keep=self.args.ckpt_max_keep,
                )
            


            # V2: Run eval in-between training loop
            if self.args.eval_in_epoch is not None:
                # self.logger.info(f"{(data_iter_step + 1) // accum_iter} {self.update_eval_list}")
                if (is_gradient_accumulation_boundary and \
                    ((data_iter_step + 1) // accum_iter) in self.update_eval_list):
                    if self.args.run_eval:
                        for eval_one_mode in self.args.eval_mode:
                            eval_stats = self.eval_one_epoch(
                                epoch,
                                self.start_iter,
                                eval_mode=eval_one_mode,
                                log_writer=self.log_writer,
                                metric_logger=self.metric_logger_to_resume,
                            )
                        self.model.train(True)

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        self.logger.info(f"Training averaged stats:\n{metric_logger}")
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    


    ### V5: support image-video joint resumed training based on V4
    def train_one_epoch_joint(
        self,
        epoch: int,
        start_iter: int,
        log_writer=None,
        metric_logger=None,
    ):
        self.model.train(True)
        if metric_logger is None:
            metric_logger = misc.MetricLogger(delimiter="  ")
            metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))

        header = "Epoch: [{}]".format(epoch)
        print_freq = 10  # todo arg

        accum_iter = self.args.accum_iter
        accum_counter = 0

        # Consecutive steps of each modality
        video_steps = self.args.video_iter_per_img_iter  # e.g. 4
        image_steps = 1
        cycle_steps = image_steps + video_steps

        # Create iterators for both dataloaders
        num_video_iters = len(self.dataloader_train)      # the integer length
        num_image_iters = len(self.dataloader_train_img)  # the integer length
        # set image start_iter
        if start_iter > 0: # NOTE: This means that this is a resumed training.
            start_iter_img = (start_iter // cycle_steps * image_steps) % num_image_iters
            start_iter_vid = start_iter // cycle_steps * video_steps
        else:
            start_iter_img = start_iter
            start_iter_vid = start_iter
        self.logger.info(f"start_iter_img, start_iter_vid: {start_iter_img}, {start_iter_vid}")

        video_iter = metric_logger.log_every(
            self.dataloader_train,
            print_freq,
            header,
            start_iter_vid,
            self.args.batch_size * fs_init.get_data_parallel_world_size(),
        )
        image_iter = metric_logger.log_every(
            self.dataloader_train_img,
            print_freq,
            header,
            start_iter_img,
            self.args.img_batch_size * fs_init.get_data_parallel_world_size(),
        )
        max_iterations = num_video_iters + math.ceil(num_video_iters/video_steps) # min or max(len(self.dataloader_train), len(self.dataloader_train_img))
        # max_iterations = num_video_iters
        # assert num_video_iters == min(num_video_iters, num_image_iters)
        self.logger.info(f"num_video_iters, num_image_iters: {num_video_iters}, {num_image_iters}")
        self.logger.info(f"Total iterations for image-video joint training: {max_iterations}.")
        self.logger.info(f"Image iterations, video iterations: {math.ceil(num_video_iters/video_steps)}, {num_video_iters}.")

        self.optimizer.zero_grad()
        data_iter_step = start_iter
        in_video = False
        in_image = False

        for data_iter_step in range(start_iter, max_iterations):
            if data_iter_step % cycle_steps < image_steps: # image*X, video*Y, image*X, video*Y, ...
            # if data_iter_step % (video_steps + image_steps) == 0: # image, video*X, image, video*X, ...
                try:
                    # self.logger.info("image_iter")  # To see whther we write the code corrctly between data types
                    # torch.cuda.synchronize()  # Ensure previous operations are complete
                    # start_time = time.time()
                    batch_data = next(image_iter)
                    # torch.cuda.synchronize()  # Synchronize again to ensure batch loading is finished
                    # end_time = time.time()
                    # self.logger.info(f"Time taken to fetch image batch: {end_time - start_time:.4f} seconds")
                except StopIteration:
                    image_iter = metric_logger.log_every(
                        self.dataloader_train_img,
                        print_freq,
                        header,
                        # start_iter,
                        0,            # start from the beginning
                        self.args.img_batch_size * fs_init.get_data_parallel_world_size(),
                    )
                    batch_data = next(image_iter)
                in_video, in_image = False, True

                # 需要有一个逻辑是，if image已经用完了，那么重新 iter(image数据集)，那么就开始复用了。默认过video数据集。
            else:
                try:
                    # torch.cuda.synchronize()  # Ensure previous operations are complete
                    # start_time = time.time()
                    # self.logger.info("video_iter")  # To see whther we write the code corrctly on alternating between data types
                    batch_data = next(video_iter)
                    # torch.cuda.synchronize()  # Synchronize again to ensure batch loading is finished
                    # end_time = time.time()
                    # self.logger.info(f"Time taken to fetch video batch: {end_time - start_time:.4f} seconds")
                except StopIteration: # Video data is used up, therefore stop the training of this epoch.
                    video_iter = metric_logger.log_every(
                        self.dataloader_train,
                        print_freq,
                        header,
                        # start_iter,
                        0,             # start from the beginning
                        self.args.batch_size * fs_init.get_data_parallel_world_size(),
                    )
                    batch_data = next(video_iter)
                in_video, in_image = True, False
                    
            
            accum_counter = (accum_counter + 1) % accum_iter
            is_gradient_accumulation_boundary = accum_counter == 0

            examples, labels = batch_data
            if is_gradient_accumulation_boundary or data_iter_step == start_iter:
                lr_sched.adjust_learning_rate_epoch(
                    self.optimizer, data_iter_step / len(self.dataloader_train) + epoch, self.args
                )
            
            if "predict_video" in self.args.pretrain_task:
                '''
                    We need to replace the special tokens in the token list (i.e., <video_resolution_placeholder>, <video_frames_placeholder>, <video_fps_placeholder>)
                        with specific tokens.
                '''     
                # fps_duration = random.choice(self.all_fps_duration) 
                fps_duration = ((1, 1), (1, 1.0)) if in_image else random.choice(self.all_fps_duration)
                # self.logger.info(f"fps_duration {fps_duration}")
                for ex, la in zip(examples, labels):
                    # if -100 in labels: 
                    #     # This means that we do not do the classifier-free guidance drop.
                    #     # If we do the drop, then 65539 and 65540 will be dropped.
                    ex = self.replace_with_multiple(
                        lst=ex, to_replace=self.video_placeholder_token["<video_frames_placeholder>"], # 65539 
                        new_tokens=self.num2tokens[str(fps_duration[0][0]*fps_duration[0][1])], labels=False
                    )
                    ex = self.replace_with_multiple(
                        lst=ex, to_replace=self.video_placeholder_token["<video_fps_placeholder>"], # 65540 
                        new_tokens=self.num2tokens[str(fps_duration[0][0])], labels=False
                    )
                    
                    la = self.replace_with_multiple(
                        lst=la, to_replace=self.video_placeholder_token["<video_frames_placeholder>"], # 65539, 
                        new_tokens=self.num2tokens[str(fps_duration[0][0]*fps_duration[0][1])], labels=True
                    )
                    la = self.replace_with_multiple(
                        lst=la, to_replace=self.video_placeholder_token["<video_fps_placeholder>"], # 65540 
                        new_tokens=self.num2tokens[str(fps_duration[0][0])], labels=True
                    )

                ### Masking for variable length training
                # Masks the initial non-visible frames, and drops frame tokens based on the given FPS and duration.
                examples, labels = self.mask_and_drop_video_random(
                    examples, 
                    labels, 
                    fps_duration[1],
                    masking_mode = self.args.masking_mode,
                    text_to_video_prob = self.args.text_to_video_prob,
                )

                ### Replace the video duration and fps tokens in the video token sequence (not text sequence).  
                examples, labels = self.update_video_duration_fps_token_in_video_seq(examples, labels, fps_duration[0])

                ### NOTE: to supoport for video classifier-free gudiance (CFG) training
                for el_idx, (ex, la) in enumerate(zip(examples, labels)):
                    if ex[-2] == la[-2] == 9005 and ex.count(9005) == 1 and ex.count(9004) == 1:  # video generation data
                        if random.random() < 0.1:
                        # if True: # for debugging
                            # print(f"Before cfg drop    ex {len(ex)} la {len(la)}")

                            # Randomly choose between dropping text only or dropping both text and masked video
                            cfg_mode = random.choice(self.args.cfg_mode)
                            # print(f"cfg_mode {cfg_mode}")

                            if cfg_mode == "text_and_video":
                                ### Classifier-free guidance with text and masked video
                                examples[el_idx] = labels[el_idx] = [_ for _ in la[:-1] if _ != -100]
                            
                            elif cfg_mode == "text_only":
                                ### Classifier-free guidance with only text
                                video_start_idx = ex.index(9004) # index only finds the first one.
                                examples[el_idx] = [e for e_idx, e in enumerate(ex) if e_idx>=video_start_idx]
                                labels[el_idx] = [l for l_idx, l in enumerate(la) if l_idx>=video_start_idx]
                            
                            else:
                                assert False, f"{cfg_mode} not implemented."
            
            # torch.cuda.synchronize()  # Ensure previous operations are complete
            # start_time = time.time()
            additional_infer_dict = {'cfg':self.args , 'mask_schedule': self.mask_schedule, 'special_token':self.special_token}
            with {
                "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
                "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
                "fp32": contextlib.nullcontext(),
                "tf32": contextlib.nullcontext(),
            }[self.args.precision]:
                c_loss, additional_loss_dict = self.model(examples, labels, **additional_infer_dict)
            
            # torch.cuda.synchronize()  # Synchronize again to ensure batch loading is finished
            # end_time = time.time()
            # self.logger.info(f"Time taken for inference: {end_time - start_time:.4f} seconds")
            
            loss = c_loss
            for add_loss, weight in additional_loss_dict.values():
                loss = loss + add_loss * weight
            loss_value = loss.item()
            c_loss_value = c_loss.item()
            if not math.isfinite(loss_value):
                self.logger.error("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            effective_loss = loss / accum_iter

            # torch.cuda.synchronize()  # Ensure previous operations are complete
            # start_time = time.time()
            with (
                self.model.no_sync()
                if self.args.data_parallel in ["sdp", "hsdp"] and not is_gradient_accumulation_boundary
                else contextlib.nullcontext()
            ):
                effective_loss.backward()

            if is_gradient_accumulation_boundary:
                grad_norm = self.model.clip_grad_norm_(max_norm=self.args.clip_grad)
                metric_logger.update(grad_norm=grad_norm)

                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                # self.logger.info("Param update.")
            
            # torch.cuda.synchronize()  # Synchronize again to ensure batch loading is finished
            # end_time = time.time()
            # self.logger.info(f"Time taken for backward: {end_time - start_time:.4f} seconds")

            torch.cuda.synchronize()

            metric_logger.update(closs=c_loss_value)
            metric_logger.update(**{key: val[0].item() for key, val in additional_loss_dict.items()})
            lr = self.optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)

            for metric_name, metric in metric_logger.meters.items():
                metric_value = metric.value
                metric_value = util.dist.all_reduce_mean(metric_value)
                if log_writer is not None:
                    log_writer.add_scalar(
                        metric_name, metric_value, data_iter_step + len(self.dataloader_train) * epoch
                    )

            # save within epoch
            n_update_per_save = self.args.save_iteration_interval // accum_iter
            if (
                is_gradient_accumulation_boundary and ((data_iter_step + 1) // accum_iter) % n_update_per_save == 0
            ) or (data_iter_step + 1 == accum_iter and epoch == 0):
                util.ckpt.save(
                    self.args.output_dir,
                    self.global_rank == 0,
                    self.model,
                    self.optimizer,
                    self.tokenizer,
                    self.args,
                    epoch=epoch,
                    iteration=data_iter_step,
                    additional_rank_specific={
                        "metric_logger": metric_logger,
                    },
                    max_keep=self.args.ckpt_max_keep,
                )
            

            # V2: Run eval in-between training loop
            if self.args.eval_in_epoch is not None:
                # self.logger.info(f"{(data_iter_step + 1) // accum_iter} {self.update_eval_list}")
                if (is_gradient_accumulation_boundary and \
                    ((data_iter_step + 1) // accum_iter) in self.update_eval_list):
                    if self.args.run_eval:
                        for eval_one_mode in self.args.eval_mode:
                            eval_stats = self.eval_one_epoch(
                                epoch,
                                self.start_iter,
                                eval_mode=eval_one_mode,
                                log_writer=self.log_writer,
                                metric_logger=self.metric_logger_to_resume,
                            )
                        self.model.train(True)
        
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        self.logger.info(f"Training averaged stats:\n{metric_logger}")
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

        
    def eval_one_epoch(
        self,
        epoch: int,
        start_iter: int,
        eval_mode: str,
        log_writer=None,
        metric_logger=None,
    ):
        # self.model.train(False)
        self.model.eval()
        if metric_logger is None:
            metric_logger = misc.MetricLogger(delimiter="  ")
            # metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))

        header = "Epoch: [{}]".format(epoch)
        print_freq = 10  # todo arg

        # accum_iter = self.args.accum_iter
        # accum_counter = 0

        # self.optimizer.zero_grad()

        with torch.no_grad():
            for data_iter_step, batch_data in enumerate(
                metric_logger.log_every(
                    self.dataloader_eval,
                    print_freq,
                    header,
                    start_iter,
                    self.args.batch_size * fs_init.get_data_parallel_world_size(),
                ),
                start=start_iter,
            ):  
                # ### For debugging
                # if data_iter_step<22:
                #     continue

                # accum_counter = (accum_counter + 1) % accum_iter
                # is_gradient_accumulation_boundary = accum_counter == 0

                examples, labels = batch_data
                # if is_gradient_accumulation_boundary or data_iter_step == start_iter:
                #     lr_sched.adjust_learning_rate_epoch(
                #         self.optimizer, data_iter_step / len(self.dataloader_train) + epoch, self.args
                #     )
                
                if "predict_video" in self.args.pretrain_task:
                    '''
                        We need to replace the special tokens in the token list (i.e., <video_resolution_placeholder>, <video_frames_placeholder>, <video_fps_placeholder>)
                            with specific tokens.
                    '''     

                    fps_duration = random.choice(self.all_fps_duration)
                    for ex, la in zip(examples, labels):
                        # if -100 in labels: 
                        #     # This means that we do not do the classifier-free guidance drop.
                        #     # If we do the drop, then 65539 and 65540 will be dropped.
                        ex = self.replace_with_multiple(
                            lst=ex, to_replace=self.video_placeholder_token["<video_frames_placeholder>"], # 65539 
                            new_tokens=self.num2tokens[str(fps_duration[0][0]*fps_duration[0][1])], labels=False
                        )
                        ex = self.replace_with_multiple(
                            lst=ex, to_replace=self.video_placeholder_token["<video_fps_placeholder>"], # 65540 
                            new_tokens=self.num2tokens[str(fps_duration[0][0])], labels=False
                        )
                        
                        la = self.replace_with_multiple(
                            lst=la, to_replace=self.video_placeholder_token["<video_frames_placeholder>"], # 65539, 
                            new_tokens=self.num2tokens[str(fps_duration[0][0]*fps_duration[0][1])], labels=True
                        )
                        la = self.replace_with_multiple(
                            lst=la, to_replace=self.video_placeholder_token["<video_fps_placeholder>"], # 65540 
                            new_tokens=self.num2tokens[str(fps_duration[0][0])], labels=True
                        )

                    # Masks the initial non-visible frames, and drops frame tokens based on the given FPS and duration.
                    examples, labels = self.mask_and_drop_video_random(
                        examples, 
                        labels, 
                        fps_duration[1],
                        masking_mode = [eval_mode],
                        # masking_mode = self.args.masking_mode,
                        text_to_video_prob = self.args.text_to_video_prob,
                    )

                    ### Replace the video duration and fps tokens in the video token sequence (not text sequence).  
                    examples, labels = self.update_video_duration_fps_token_in_video_seq(examples, labels, fps_duration[0])

                    ### NOTE: to supoport for video classifier-free gudiance (CFG) training
                    for el_idx, (ex, la) in enumerate(zip(examples, labels)):
                        if ex[-2] == la[-2] == 9005 and ex.count(9005) == 1 and ex.count(9004) == 1:  # video generation data
                            if random.random() < 0.1:
                            # if True: # for debugging
                                # import ipdb
                                # ipdb.set_trace()
                                # print(f"Before cfg drop    ex {len(ex)} la {len(la)}")

                                # Randomly choose between dropping text only or dropping both text and masked video
                                cfg_mode = random.choice(self.args.cfg_mode)
                                # print(f"cfg_mode {cfg_mode}")

                                if cfg_mode == "text_and_video":
                                    ### Classifier-free guidance with text and masked video
                                    examples[el_idx] = labels[el_idx] = [_ for _ in la[:-1] if _ != -100]
                                
                                elif cfg_mode == "text_only":
                                    ### Classifier-free guidance with only text
                                    video_start_idx = ex.index(9004) # index only finds the first one.
                                    examples[el_idx] = [e for e_idx, e in enumerate(ex) if e_idx>=video_start_idx]
                                    labels[el_idx] = [l for l_idx, l in enumerate(la) if l_idx>=video_start_idx]
                                
                                else:
                                    assert False, f"{cfg_mode} not implemented."
        

                additional_infer_dict = {'cfg':self.args , 'mask_schedule': self.mask_schedule, 'special_token':self.special_token}   
                with {
                    "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
                    "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
                    "fp32": contextlib.nullcontext(),
                    "tf32": contextlib.nullcontext(),
                }[self.args.precision]:
                    # c_loss, additional_loss_dict = self.model(examples, labels)
                    c_loss, additional_loss_dict = self.model(examples, labels, **additional_infer_dict)

                
                loss_value = c_loss.item()
                if not math.isfinite(loss_value):
                    self.logger.error("Loss is {}, stopping evaluation".format(loss_value))
                    sys.exit(1)
                metric_logger.update(closs=loss_value)
                metric_logger.update(**{key: val[0].item() for key, val in additional_loss_dict.items()})
                
                for metric_name, metric in metric_logger.meters.items():
                    metric_value = metric.value
                    metric_value = util.dist.all_reduce_mean(metric_value)
                    if log_writer is not None:
                        log_writer.add_scalar(
                            metric_name, metric_value, data_iter_step + len(self.dataloader_eval) * epoch
                        )
                

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        self.logger.info(f"Eval {eval_mode} averaged stats:\n{metric_logger}")
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        

    def build_eval_data(self):
        eff_batch_size = self.args.batch_size * self.args.accum_iter * fs_init.get_data_parallel_world_size()
        self.logger.info("effective batch size: %d" % eff_batch_size)

        dataset_eval = self._eval_dataset_func()
        self.logger.info(self.args.eval_data_config)
        self.logger.info(dataset_eval)

        sampler_eval = FinetuneDistSampler(
            dataset_eval,
            num_replicas=self.dp_world_size,
            rank=self.dp_rank,
            shuffle=True, # NOTE We do not need to do the shuffling but it is not implemented.
            batch_size=self.args.batch_size,
            acc_grad=self.args.accum_iter,
            seed=self.args.seed,
            length_clustering=self.args.length_clustering,
        )

        dataloader_eval = torch.utils.data.DataLoader(
            dataset_eval,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_mem,
            sampler=sampler_eval,
            collate_fn=lambda batch: tuple(zip(*batch)),
            # drop_last=True,
            drop_last=False,
            # persistent_workers=True  # Keeps workers alive between iterations
        )

        return dataset_eval, sampler_eval, dataloader_eval
    
    def build_image_data(self):
        '''
            This function aims to initialize the image class for image-video joint training.
        '''
        eff_batch_size = self.args.img_batch_size * self.args.accum_iter * fs_init.get_data_parallel_world_size()
        self.logger.info("effective batch size: %d" % eff_batch_size)

        dataset_img = self._img_dataset_func()
        self.logger.info(self.args.img_data_config)
        self.logger.info(dataset_img)

        sampler_img = FinetuneDistSampler(
            dataset_img,
            num_replicas=self.dp_world_size,
            rank=self.dp_rank,
            shuffle=True, # NOTE We do not need to do the shuffling but it is not implemented.
            batch_size=self.args.img_batch_size,
            acc_grad=self.args.accum_iter,
            seed=self.args.seed,
            length_clustering=self.args.length_clustering,
        )

        dataloader_img = torch.utils.data.DataLoader(
            dataset_img,
            batch_size=self.args.img_batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_mem,
            sampler=sampler_img,
            collate_fn=lambda batch: tuple(zip(*batch)),
            # drop_last=True,
            drop_last=False,
            # persistent_workers=True  # Keeps workers alive between iterations
        )

        return dataset_img, sampler_img, dataloader_img
    
    
    def _eval_dataset_func(self):
        item_processor = self._item_processor_func()

        dataset = FinetuneConversationDataset(
            self.args.eval_data_config, item_processor=item_processor, cache_on_disk=self.args.cache_ann_on_disk
        )
        return dataset
    

    def _img_dataset_func(self):
        item_processor = self._item_processor_func()

        dataset = FinetuneConversationDataset(
            self.args.img_data_config, item_processor=item_processor, cache_on_disk=self.args.cache_ann_on_disk
        )
        return dataset

    
    def define_fps_duration_combinations(self):
        """
        1. Check if any fps or duration exceeds the maximum allowed values and 
        2. Generate all possible combinations of frame rate (fps) and duration for video processing.

        Args:
            None

        Returns:
            list of tuples: A list containing tuples where each tuple consists of:
                - rgb_fps_duration (tuple): A frame rate (fps) and a duration for RGB video processing.
                - latent_fps_duration (tuple): A frame rate (fps) and a compressed duration for latent video processing.
        """
        if "Cosmos-Tokenizer" in self.args.visual_tokenizer:
            # NOTE 1 Currently, we only support the original fps (for video) and fps=1&duration=1 (for image) since we use a video visual tokenizer.
            # NOTE 2 We do not change the latent fps. We only change the duration to latent duration.
            assert (len(self.args.video_fps) == 1 and self.args.video_fps[0] == self.data_video_fps) or \
                   (len(self.args.video_fps) == 1 and self.args.video_fps[0] == 1 and len(self.args.video_duration) == 1 and self.args.video_duration[0] == 1), "video_fps and video_duration are not supported."

            rgb_fps_duration = [(f, d) for f in self.args.video_fps for d in self.args.video_duration]

            # V1
            # compressed_duration = [math.ceil((vd - 1) / self.temporal_compression) + 1 for vd in self.args.video_duration]
            # latent_fps_duration = [(f, d) for f in self.args.video_fps for d in compressed_duration]

            # V2: Calculate the number of frames first and then calculate the duration.
            rgb_frame_num = [f * d for (f, d) in rgb_fps_duration]
            latent_frame_num = [math.ceil((rfn - 1) / self.temporal_compression) + 1 for rfn in rgb_frame_num]
            latent_fps_duration = [(f, lfn/f) for lfn, (f, _) in zip(latent_frame_num, rgb_fps_duration)]
            self.logger.info(f"latent_frame_num {latent_frame_num}")
            self.logger.info(f"latent_fps_duration {latent_fps_duration}")

            return [(rfd, lfd) for rfd, lfd in zip(rgb_fps_duration, latent_fps_duration)]

        for f in self.args.video_fps:
            if f > self.data_video_fps:
                raise ValueError(f"Video data have a maximum fps of {self.data_video_fps}.")

        # Check if any duration exceeds the maximum data duration
        for d in self.args.video_duration:
            if d > self.data_video_duration:
                raise ValueError(f"Video data have a maximum duration of {self.data_video_duration}.")

        # Create all combinations of fps and duration
        rgb_fps_duration = [(f, d) for f in self.args.video_fps for d in self.args.video_duration]
        latent_fps_duration = deepcopy(rgb_fps_duration)

        return [(rfd, lfd) for rfd, lfd in zip(rgb_fps_duration, latent_fps_duration)]


    ### V2: Suitable for Cosmos and Chameleon tokenizer.
    def replace_with_multiple(self, lst, to_replace, new_tokens, video_start_token=9004, labels=False):
        """
        Replaces an element in a list with multiple elements.
        NOTE: This function is compatible with COSMOS (whose visual tokens start from 65536) 
        because 1. ".index" operation only finds the first token in the sequence.
                2. we locate the old tokens (#frames and fps) only before the video_start_token
                   since if we use Cosmos, we might have visual tokens the same as the special tokens (#frames 65539 and fps 65540).

        Parameters:
        lst (list): The original list where the replacement is to be made.
        to_replace: The element in the list to be replaced.
        new_tokens (list): A list of new elements to replace the original element.
        labels (bool): A flag indicating whether it is the 'labels' variable.

        Returns:
        list: A new list with the specified replacement.
        """
        # import ipdb
        # ipdb.set_trace()

        video_start_index = lst.index(video_start_token)
        before_video_lst = lst[:video_start_index]

        # Find the index of the number to replace
        if not labels: # examples
            index = before_video_lst.index(to_replace)
            # assert index < video_start_index, "lst does not have a valid special token."

        else:          # labels
            if to_replace not in before_video_lst:
                # We need an inplace operation, so that we can reveal the change in examples and labels. 
                # So, we do not use lst = [-100]*(len(new_tokens)-1)+lst.
                lst[:0] = [-100]*(len(new_tokens)-1)
                return lst
            else:
                index = before_video_lst.index(to_replace)
        
        # Remove the element at the found index
        lst.pop(index)
        
        # Insert the new numbers at the same position
        lst[index:index] = new_tokens
        
        return lst

    
    ### Version 2 of mask_and_drop_video_random, supporting "text_to_video" and "video_next_frame", with a pre-defined probability. 
    def mask_and_drop_video_random(self, examples, labels, fps_duration, masking_mode=[], text_to_video_prob=0.5):
        """
        Masks the initial non-visible frames, and drops frame tokens based on the given FPS and duration.
        Tokens with indices set to `-100` are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        
        Parameters:
            examples: list of token lists, where each list represents tokens for a specific example.
            labels: list of token labels, with tokens unnecessary for training set to -100.
            fps_duration: a list containing two elements: [video_fps, video_duration]
                        video_fps: the frame rate (frames per second)
                        video_duration: the total duration of the video in seconds
            masking_mode: list of masking modes, 
                            SUPPORTED: ["text_to_video"], ["video_next_frame"] and ["text_to_video", "video_next_frame"]
            text_to_video_prob: probability for selecting text_to_video masking mode.

        Returns:
            examples: updated examples with frames dropped.
            labels: updated labels with frames dropped and some masked as non-visible.
        """
        tgt_video_fps, tgt_video_duration = fps_duration
        total_frames = round(tgt_video_fps * tgt_video_duration) # tgt_video_fps * tgt_video_duration
        assert len(masking_mode) > 0, "Masking mode should not be []."
        # !!!! total_frames是真实的想要的frame数量

        mask_frame_num = []
        mask_prob = []
        if "text_to_video" in masking_mode:
            # No frames are masked (i.e., predict the whole video)
            # NOTE that if mask_frame_num==0, meaning that we should predict the whole video.
            # Then, we should not mask the video_start_token, fps and duration token, because they should be predicted.
            mask_frame_num.append(0)
            if len(masking_mode) == 1:
                mask_prob.append(1.0)
            else:
                mask_prob.append(text_to_video_prob)
        
        if "video_next_frame" in masking_mode:
            # Randomly select a number of frames to be non-visible
            mask_frame_num += list(range(1, total_frames))
            remaining_prob = 1 - sum(mask_prob)
            mask_prob += [remaining_prob / (total_frames - 1)] * (total_frames - 1)
        
        
        # NOTE Currently, this is supported only during eval, so len(masking_mode)==1.
        video_pred_keys = [mode for mode in masking_mode if mode.startswith("video_prediction")]
        if len(video_pred_keys) == 1:
        # if any([mode.startswith("video_prediction") for mode in masking_mode]):
            mask_ratio = float(video_pred_keys[0].split("_")[-1])
            mask_frame_num.append(int(mask_ratio * total_frames))

            if len(masking_mode) == 1:
                mask_prob.append(1.0) # NOTE Since this indicates that it is used in eval mode and len(masking_mode)==1, then the prob should be 1.0.
            elif len(masking_mode) == 2:
                assert "text_to_video" in masking_mode, f"masking_mode combination {masking_mode} not supported."
                remaining_prob = 1 - sum(mask_prob)
                mask_prob.append(remaining_prob) # NOTE Since this is only used in eval mode and len(masking_mode)==1, then the prob should be 1.0.
            else:
                assert False, f"masking_mode combination {masking_mode} not supported."
            # print(f"mask_frame_num {mask_frame_num} mask_prob {mask_prob}")
        elif len(video_pred_keys) == 0:
            None
        else:
            assert False, "len(video_pred_keys) > 1 not supported."
        
        # print(f"mask_prob {mask_prob}, sum {sum(mask_prob)}")
        # print(f"masking_mode {masking_mode} mask_frame_num {mask_frame_num}")
        
        random_mask_frame_num = random.choices(mask_frame_num, weights = mask_prob, k = 1)[0]

        for ex, la in zip(examples, labels):
            # Find the video_start_token
            video_tokens_with_indices = self.find_media_tokens_in_sublist(
                token_list=la, start_index=0, end_index=len(la),
                media_start_token=9004, media_end_token=9005
            )
            
            for vti in video_tokens_with_indices:
                _, media_start_idx, media_end_idx = vti
                
                frame_tokens_with_indices = self.find_media_tokens_in_sublist(
                    token_list=la, start_index=media_start_idx, end_index=media_end_idx,
                    media_start_token=8197, media_end_token=8196
                )

                ### V1: assertion in a function
                self.assert_image_video_numbers(frame_tokens_with_indices)
                # ### V0
                # if self.args.visual_tokenizer == "Chameleon":
                #     assert len(frame_tokens_with_indices) == self.data_video_frames, (
                #         "Number of frame tokens doesn't match the expected video frames."
                #     )
                # elif "Cosmos-Tokenizer" in self.args.visual_tokenizer:
                #     assert len(frame_tokens_with_indices) == math.ceil((self.data_video_frames - 1)/self.temporal_compression) + 1, (
                #         "Number of frame tokens doesn't match the expected video frames."
                #     ) # ((self.data_video_frames - 1)//self.temporal_compression + 1) + 1
                # else:
                #     assert False, f"{self.args.visual_tokenizer} is not implemented."


                ### Step 1: Mask first few non-visible frames (we will mask frame_interval times frames)
                # Calculate the interval for keeping frames (i.e., which frames to keep based on fps and duration)
                frame_interval = self.data_video_fps // tgt_video_fps

                if random_mask_frame_num > 0: # we only need to do this when we have at least one frame to mask.
                    mask_frame_num_x_interval = random_mask_frame_num * frame_interval
                    for idx, (_, frame_start_idx, frame_end_idx) in enumerate(frame_tokens_with_indices):
                        if idx < mask_frame_num_x_interval:
                            # Mask the frames by setting their corresponding labels to -100
                            la[frame_start_idx:frame_end_idx] = [-100] * (frame_end_idx - frame_start_idx)
                            
                            # Mask the video_start_token, duration token and fps token
                            if idx == 0: # There are prefix token in the font part of the label list.
                                assert la[frame_start_idx-3] == 9004, 'Missing video_start_token.'
                                la[frame_start_idx-3:frame_start_idx] = [-100] * 3
                  
                
                ### Step 2: Drop frame tokens to align with the given FPS and duration
                # Drop frames that 1) are not in the interval list or
                #                  2) are longer than tgt_video_duration (for both examples and labels)
                # We MUST do this in a reverse order, otherwise we cannot use the frame_start_idx and frame_end_idx index.
                for idx, (_, frame_start_idx, frame_end_idx) in enumerate(frame_tokens_with_indices[::-1]):
                    forward_frame_index = len(frame_tokens_with_indices) - 1 - idx  # Calculate forward index
                    if forward_frame_index % frame_interval != 0 or \
                                (forward_frame_index + 1) / self.data_video_fps > tgt_video_duration:
                        # Drop frames in both examples and labels
                        ex[frame_start_idx:frame_end_idx] = []  # Remove the frame from examples
                        la[frame_start_idx:frame_end_idx] = []  # Remove the frame from labels
        
        return examples, labels
    
    def update_video_duration_fps_token_in_video_seq(self, examples, labels, fps_duration, video_start_token_id=9004, zero_token_id=8804):
        '''
        Update the duration_token and fps_token in the video sequence after replacing mask tokens.
        Specifically, it replaces the tokens that represent the video start token, duration, and fps values.
        
        Args:
            examples (list of list of ints): The list of tokenized video sequences in the input examples.
            labels (list of list of ints): The corresponding list of labels for the examples, typically the target sequences.
            fps_duration (tuple): A tuple containing the fps and duration of the video. Example: (fps, duration).
            video_start_token_id (int): The token id that marks the start of the video. Default is 9004.
            zero_token_id (int): The token id used as the zero padding token. Default is 8804.

        Returns:
            tuple: A tuple (updated_examples, updated_labels) with the updated video sequence examples and labels.
        '''
        fps, duration = fps_duration
        duration_token = duration + zero_token_id
        fps_token = fps + zero_token_id
        # print(duration_token, fps_token)

        for example, label in zip(examples, labels):
            ### Must replace the duration and fps tokens in examples.
            assert video_start_token_id in example, f"Video start token {video_start_token_id} not found in example."
            video_start_token_id_idx_ex = example.index(video_start_token_id)
            example[(video_start_token_id_idx_ex + 1) : (video_start_token_id_idx_ex + 3)] = [duration_token, fps_token]
            
            ### Replace the duration and fps tokens in examples if there are.
            ### Assume that examples and labels are matched.
            if video_start_token_id in label: # it means that the whole video will be predicted
                video_start_token_id_idx_la = label.index(video_start_token_id)
                assert video_start_token_id_idx_la == video_start_token_id_idx_ex, \
                        f"Mismatch in video start token indices in example ({video_start_token_id_idx_ex}) and label ({video_start_token_id_idx_la})."
                label[(video_start_token_id_idx_ex + 1) : (video_start_token_id_idx_ex + 3)] = [duration_token, fps_token]
            else: # it means that we are doing video prediction.
                assert label[(video_start_token_id_idx_ex + 1) : (video_start_token_id_idx_ex + 3)] == [-100, -100], \
                f"Expected [-100, -100] at positions ({video_start_token_id_idx_ex + 1}, {video_start_token_id_idx_ex + 2}) in label, but got {label[(video_start_token_id_idx_ex + 1):(video_start_token_id_idx_ex + 3)]}."

        return examples, labels
    
    
    def assert_image_video_numbers(self, frame_tokens_with_indices):
        ### V1: ssertion in a function
        if self.args.visual_tokenizer == "Chameleon":
            assert len(frame_tokens_with_indices) == self.data_video_frames, (
                "Number of frame tokens doesn't match the expected video frames."
            )
        elif "Cosmos-Tokenizer" in self.args.visual_tokenizer:
            if self.args.joint_img_video:
                video_flag = len(frame_tokens_with_indices) == math.ceil((self.data_video_frames - 1)/self.temporal_compression) + 1
                image_flag = len(frame_tokens_with_indices) == 1
                assert video_flag or image_flag, (
                    "Number of frame tokens doesn't match the expected video frames and image frames."
                )
            else:
                assert len(frame_tokens_with_indices) == math.ceil((self.data_video_frames - 1)/self.temporal_compression) + 1, (
                    "Number of frame tokens doesn't match the expected video frames."
                ) # ((self.data_video_frames - 1)//self.temporal_compression + 1) + 1
        else:
            assert False, f"{self.args.visual_tokenizer} is not implemented."



    def find_media_tokens_in_sublist(self, token_list, start_index, end_index, media_start_token, media_end_token):
        """
        Finds all media tokens (e.g., for video or image) between the media_start_token and media_end_token 
        in a sublist of the token list, defined by the start_index and end_index. Returns the media tokens 
        along with their global start and end indices in the original token_list.

        Parameters:
        token_list (list): The list of tokens to search through.
        start_index (int): The starting index of the sublist to search in.
        end_index (int): The ending index of the sublist to search in (exclusive).
        media_start_token (int): The token indicating the start of a media (default 9004).
        media_end_token (int): The token indicating the end of a media (default 9005).

        Returns:
        list: A list of tuples, where each tuple contains:
                    - A list of tokens for a specific media found in the sublist, CONTAINING start and end token.
                    - The global start index of the media in the original token_list.
                    - The global end index of the media in the original token_list.
        """

        media_tokens_with_indices = []
        current_media = None
        media_start_idx = None

        # Ensure the sublist boundaries are within valid range
        sublist = token_list[start_index:end_index]

        for idx, token in enumerate(sublist):
            global_idx = start_index + idx  # Calculate global index relative to the original token_list
            
            if token == media_start_token:
                # Start a new media token collection and record the global start index
                # current_media = []
                current_media = [token]
                media_start_idx = global_idx  # Store global start index
            elif token == media_end_token and current_media is not None:
                # End the current media token collection, record the global end index, and append to the list
                current_media.append(token)
                media_tokens_with_indices.append((current_media, media_start_idx, global_idx+1)) # +1 for easy usage
                current_media = None
                media_start_idx = None
            elif current_media is not None:
                # If we are within a media, collect tokens
                current_media.append(token)

        return media_tokens_with_indices


if __name__ == "__main__":
    args = Solver2.get_args_parser().parse_args()
    solver = Solver2(args)
    solver.run()