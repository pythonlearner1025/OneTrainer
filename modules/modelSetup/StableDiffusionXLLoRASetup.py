from typing import Iterable

import torch
from torch.nn import Parameter

from modules.model.StableDiffusionXLModel import StableDiffusionXLModel
from modules.modelSetup.BaseStableDiffusionXLSetup import BaseStableDiffusionXLSetup
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util import create
from modules.util.TrainProgress import TrainProgress
from modules.util.args.TrainArgs import TrainArgs
from modules.util.enum.LearningRateScaler import LearningRateScaler


class StableDiffusionXLLoRASetup(BaseStableDiffusionXLSetup):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            debug_mode: bool,
    ):
        super(StableDiffusionXLLoRASetup, self).__init__(
            train_device=train_device,
            temp_device=temp_device,
            debug_mode=debug_mode,
        )

    def create_parameters(
            self,
            model: StableDiffusionXLModel,
            args: TrainArgs,
    ) -> Iterable[Parameter]:
        params = list()
        batch_size = 1 if args.learning_rate_scaler in [LearningRateScaler.NONE, LearningRateScaler.GRADIENT_ACCUMULATION] else args.batch_size
        gradient_accumulation_steps = 1 if args.learning_rate_scaler in [LearningRateScaler.NONE, LearningRateScaler.BATCH] else args.gradient_accumulation_steps

        if args.train_text_encoder:
            params += list(model.text_encoder_1_lora.parameters())

        if args.train_text_encoder_2:
            params += list(model.text_encoder_2_lora.parameters())

        if args.train_unet:
            params += list(model.unet_lora.parameters())

        return params

    def create_parameters_for_optimizer(
            self,
            model: StableDiffusionXLModel,
            args: TrainArgs,
    ) -> Iterable[Parameter] | list[dict]:
        param_groups = list()

        if args.train_text_encoder:
            lr = args.text_encoder_learning_rate if args.text_encoder_learning_rate is not None else args.learning_rate
            lr = lr = lr * ((batch_size * gradient_accumulation_steps) ** 0.5)

            param_groups.append({
                'params': model.text_encoder_1_lora.parameters(),
                'lr': lr,
                'initial_lr': lr,
            })

        if args.train_text_encoder_2:
            lr = args.text_encoder_2_learning_rate if args.text_encoder_2_learning_rate is not None else args.learning_rate
            lr = lr = lr * ((batch_size * gradient_accumulation_steps) ** 0.5)

            param_groups.append({
                'params': model.text_encoder_2_lora.parameters(),
                'lr': lr,
                'initial_lr': lr,
            })

        if args.train_unet:
            lr = args.unet_learning_rate if args.unet_learning_rate is not None else args.learning_rate
            lr = lr = lr * ((batch_size * gradient_accumulation_steps) ** 0.5)

            param_groups.append({
                'params': model.unet_lora.parameters(),
                'lr': lr,
                'initial_lr': lr,
            })

        return param_groups

    def setup_model(
            self,
            model: StableDiffusionXLModel,
            args: TrainArgs,
    ):
        if model.text_encoder_1_lora is None and args.train_text_encoder:
            model.text_encoder_1_lora = LoRAModuleWrapper(
                model.text_encoder_1, args.lora_rank, "lora_te1", args.lora_alpha
            )

        if model.text_encoder_2_lora is None and args.train_text_encoder_2:
            model.text_encoder_2_lora = LoRAModuleWrapper(
                model.text_encoder_2, args.lora_rank, "lora_te2", args.lora_alpha
            )

        if model.unet_lora is None and args.train_unet:
            model.unet_lora = LoRAModuleWrapper(
                model.unet, args.lora_rank, "lora_unet", args.lora_alpha, ["attentions"]
            )

        model.text_encoder_1.requires_grad_(False)
        model.text_encoder_2.requires_grad_(False)
        model.unet.requires_grad_(False)
        model.vae.requires_grad_(False)

        if model.text_encoder_1_lora is not None:
            train_text_encoder_1 = args.train_text_encoder and (model.train_progress.epoch < args.train_text_encoder_epochs)
            model.text_encoder_1_lora.requires_grad_(train_text_encoder_1)
        if model.text_encoder_2_lora is not None:
            train_text_encoder_2 = args.train_text_encoder_2 and (model.train_progress.epoch < args.train_text_encoder_2_epochs)
            model.text_encoder_2_lora.requires_grad_(train_text_encoder_2)
        if model.unet_lora is not None:
            train_unet = args.train_unet and (model.train_progress.epoch < args.train_unet_epochs)
            model.unet_lora.requires_grad_(train_unet)

        if model.text_encoder_1_lora is not None:
            model.text_encoder_1_lora.hook_to_module()
            model.text_encoder_1_lora.to(dtype=args.lora_weight_dtype.torch_dtype())
        if model.text_encoder_2_lora is not None:
            model.text_encoder_2_lora.hook_to_module()
            model.text_encoder_2_lora.to(dtype=args.lora_weight_dtype.torch_dtype())
        if model.unet_lora is not None:
            model.unet_lora.hook_to_module()
            model.unet_lora.to(dtype=args.lora_weight_dtype.torch_dtype())

        model.optimizer = create.create_optimizer(
            self.create_parameters_for_optimizer(model, args), model.optimizer_state_dict, args
        )
        del model.optimizer_state_dict

        model.ema = create.create_ema(
            self.create_parameters(model, args), model.ema_state_dict, args
        )
        del model.ema_state_dict

        self.setup_optimizations(model, args)

    def setup_train_device(
            self,
            model: StableDiffusionXLModel,
            args: TrainArgs,
    ):
        vae_on_train_device = args.align_prop
        text_encoder_1_on_train_device = args.train_text_encoder or args.align_prop or not args.latent_caching
        text_encoder_2_on_train_device = args.train_text_encoder_2 or args.align_prop or not args.latent_caching

        model.text_encoder_1_to(self.train_device if text_encoder_1_on_train_device else self.temp_device)
        model.text_encoder_2_to(self.train_device if text_encoder_2_on_train_device else self.temp_device)
        model.vae_to(self.train_device if vae_on_train_device else self.temp_device)
        model.unet_to(self.train_device)

        if args.train_text_encoder:
            model.text_encoder_1.train()
        else:
            model.text_encoder_1.eval()

        if args.train_text_encoder_2:
            model.text_encoder_2.train()
        else:
            model.text_encoder_2.eval()

        model.vae.eval()

        if args.train_unet:
            model.unet.train()
        else:
            model.unet.eval()

    def after_optimizer_step(
            self,
            model: StableDiffusionXLModel,
            args: TrainArgs,
            train_progress: TrainProgress
    ):
        if model.text_encoder_1_lora is not None:
            train_text_encoder_1 = args.train_text_encoder and (model.train_progress.epoch < args.train_text_encoder_epochs)
            model.text_encoder_1_lora.requires_grad_(train_text_encoder_1)

        if model.text_encoder_2_lora is not None:
            train_text_encoder_2 = args.train_text_encoder_2 and (model.train_progress.epoch < args.train_text_encoder_2_epochs)
            model.text_encoder_2_lora.requires_grad_(train_text_encoder_2)

        if model.unet_lora is not None:
            train_unet = args.train_unet and (model.train_progress.epoch < args.train_unet_epochs)
            model.unet_lora.requires_grad_(train_unet)
