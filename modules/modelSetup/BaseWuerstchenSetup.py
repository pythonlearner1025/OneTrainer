from abc import ABCMeta
from contextlib import nullcontext

import torch
from diffusers.models.attention_processor import AttnProcessor, XFormersAttnProcessor, AttnProcessor2_0, Attention
from diffusers.utils import is_xformers_available
from torch import Tensor

from modules.model.WuerstchenModel import WuerstchenModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupDebugMixin import ModelSetupDebugMixin
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.modelSetup.mixin.ModelSetupDiffusionNoiseMixin import ModelSetupDiffusionNoiseMixin
from modules.modelSetup.stableDiffusion.checkpointing_util import enable_checkpointing_for_clip_encoder_layers, \
    enable_checkpointing_for_stable_cascade_blocks
from modules.util.TrainProgress import TrainProgress
from modules.util.config.TrainConfig import TrainConfig
from modules.util.dtype_util import create_autocast_context, disable_fp16_autocast_context
from modules.util.enum.AttentionMechanism import AttentionMechanism
from modules.util.enum.TrainingMethod import TrainingMethod


class BaseWuerstchenSetup(
    BaseModelSetup,
    ModelSetupDiffusionLossMixin,
    ModelSetupDebugMixin,
    ModelSetupDiffusionNoiseMixin,
    metaclass=ABCMeta,
):

    def setup_optimizations(
            self,
            model: WuerstchenModel,
            config: TrainConfig,
    ):
        if config.attention_mechanism == AttentionMechanism.DEFAULT:
            model.prior_prior.set_attn_processor(AttnProcessor())
        elif config.attention_mechanism == AttentionMechanism.XFORMERS and is_xformers_available():
            try:
                for name, child_module in model.prior_prior.named_modules():
                    if isinstance(child_module, Attention):
                        child_module.set_processor(XFormersAttnProcessor())
            except Exception as e:
                print(
                    "Could not enable memory efficient attention. Make sure xformers is installed"
                    f" correctly and a GPU is available: {e}"
                )
        elif config.attention_mechanism == AttentionMechanism.SDP:
            for name, child_module in model.prior_prior.named_modules():
                if isinstance(child_module, Attention):
                    child_module.set_processor(AttnProcessor2_0())

        if config.gradient_checkpointing:
            if model.model_type.is_wuerstchen_v2():
                model.prior_prior.enable_gradient_checkpointing()
                enable_checkpointing_for_clip_encoder_layers(model.prior_text_encoder, self.train_device)
            else:
                enable_checkpointing_for_stable_cascade_blocks(model.prior_text_encoder, self.train_device)
                enable_checkpointing_for_clip_encoder_layers(model.prior_text_encoder, self.train_device)

        model.autocast_context, model.train_dtype = create_autocast_context(self.train_device, config.train_dtype, [
            config.weight_dtypes().decoder_text_encoder,
            config.weight_dtypes().decoder,
            config.weight_dtypes().decoder_vqgan,
            config.weight_dtypes().effnet_encoder,
            config.weight_dtypes().text_encoder,
            config.weight_dtypes().prior,
            config.weight_dtypes().lora if config.training_method == TrainingMethod.LORA else None,
            config.weight_dtypes().embedding if config.training_method == TrainingMethod.EMBEDDING else None,
        ])

        if model.model_type.is_wuerstchen_v3():
            model.prior_autocast_context, model.prior_train_dtype = disable_fp16_autocast_context(
                self.train_device,
                config.train_dtype,
                config.fallback_train_dtype,
                [
                    config.weight_dtypes().prior,
                    config.weight_dtypes().lora if config.training_method == TrainingMethod.LORA else None,
                ],
            )

    def __alpha_cumprod(
            self,
            timesteps: Tensor,
            dim: int,
    ):
        # copied and modified from https://github.com/dome272/wuerstchen
        s = torch.tensor([0.008], device=timesteps.device, dtype=torch.float32)
        init_alpha_cumprod = torch.cos(s / (1 + s) * torch.pi * 0.5) ** 2
        alpha_cumprod = torch.cos((timesteps + s) / (1 + s) * torch.pi * 0.5) ** 2 / init_alpha_cumprod
        alpha_cumprod = alpha_cumprod.clamp(0.0001, 0.9999)
        alpha_cumprod = alpha_cumprod.view(timesteps.shape[0])
        while alpha_cumprod.dim() < dim:
            alpha_cumprod = alpha_cumprod.unsqueeze(-1)
        return alpha_cumprod

    def predict(
            self,
            model: WuerstchenModel,
            batch: dict,
            config: TrainConfig,
            train_progress: TrainProgress,
            *,
            deterministic: bool = False,
    ) -> dict:
        with model.autocast_context:
            latent_mean = 42.0
            latent_std = 1.0

            latent_image = batch['latent_image']
            scaled_latent_image = latent_image.add(latent_std).div(latent_mean)

            generator = torch.Generator(device=config.train_device)
            generator.manual_seed(train_progress.global_step)

            latent_noise = self._create_noise(scaled_latent_image, config, generator)

            timestep = self._get_timestep_continuous(
                deterministic,
                generator,
                scaled_latent_image.shape[0],
                config,
                train_progress.global_step,
            ).mul(1.08).add(0.001).clamp(0.001, 1.0)

            scaled_noisy_latent_image = self._add_noise_continuous(
                scaled_latent_image,
                latent_noise,
                timestep,
                self.__alpha_cumprod,
            )

            if config.train_text_encoder or config.training_method == TrainingMethod.EMBEDDING:
                text_encoder_output = model.prior_text_encoder(
                    batch['tokens'], output_hidden_states=True, return_dict=True
                )
                final_layer_norm = model.prior_text_encoder.text_model.final_layer_norm
                text_encoder_output = final_layer_norm(
                    text_encoder_output.hidden_states[-(1 + config.text_encoder_layer_skip)]
                )
                if model.model_type.is_wuerstchen_v3():
                    pooled_text_encoder_output = model.prior_text_encoder.text_projection(text_encoder_output)
            else:
                text_encoder_output = batch['text_encoder_hidden_state']
                pooled_text_encoder_output = batch['pooled_text_encoder_output']

            latent_input = scaled_noisy_latent_image

            if model.model_type.is_wuerstchen_v2():
                prior_kwargs = {
                    'c': text_encoder_output.to(dtype=model.prior_train_dtype.torch_dtype()),
                }
            elif model.model_type.is_wuerstchen_v3():
                prior_kwargs = {
                    'clip_text': text_encoder_output.to(dtype=model.prior_train_dtype.torch_dtype()),
                    'clip_text_pooled': pooled_text_encoder_output.to(dtype=model.prior_train_dtype.torch_dtype()),
                }

            with model.prior_autocast_context:
                predicted_latent_noise = model.prior_prior(
                    latent_input.to(dtype=model.prior_train_dtype.torch_dtype()),
                    timestep.to(dtype=model.prior_train_dtype.torch_dtype()),
                    **prior_kwargs,
                )

            model_output_data = {
                'loss_type': 'target',
                'predicted': predicted_latent_noise,
                'target': latent_noise,
                'timestep': timestep,
            }

            if config.debug_mode:
                with torch.no_grad():
                    self._save_text(
                        self._decode_tokens(batch['tokens'], model.prior_tokenizer),
                        config.debug_dir + "/training_batches",
                        "7-prompt",
                        train_progress.global_step,
                    )

                    # noise
                    self._save_image(
                        self._project_latent_to_image(latent_noise).clamp(-1, 1),
                        config.debug_dir + "/training_batches",
                        "1-noise",
                        train_progress.global_step
                    )

                    # predicted noise
                    self._save_image(
                        self._project_latent_to_image(predicted_latent_noise).clamp(-1, 1),
                        config.debug_dir + "/training_batches",
                        "2-predicted_noise",
                        train_progress.global_step
                    )

                    # noisy image
                    self._save_image(
                        self._project_latent_to_image(scaled_noisy_latent_image).clamp(-1, 1),
                        config.debug_dir + "/training_batches",
                        "3-noisy_image",
                        train_progress.global_step
                    )

                    # predicted image
                    alpha_cumprod = self.__alpha_cumprod(timestep, latent_noise.dim())
                    sqrt_alpha_prod = alpha_cumprod ** 0.5
                    sqrt_alpha_prod = sqrt_alpha_prod.flatten().reshape(-1, 1, 1, 1)

                    sqrt_one_minus_alpha_prod = (1 - alpha_cumprod) ** 0.5
                    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten().reshape(-1, 1, 1, 1)

                    scaled_predicted_latent_image = \
                        (scaled_noisy_latent_image - predicted_latent_noise * sqrt_one_minus_alpha_prod) \
                        / sqrt_alpha_prod
                    self._save_image(
                        self._project_latent_to_image(scaled_predicted_latent_image).clamp(-1, 1),
                        config.debug_dir + "/training_batches",
                        "4-predicted_image",
                        model.train_progress.global_step
                    )

                    # image
                    self._save_image(
                        self._project_latent_to_image(scaled_latent_image).clamp(-1, 1),
                        config.debug_dir + "/training_batches",
                        "5-image",
                        model.train_progress.global_step
                    )

        return model_output_data

    def calculate_loss(
            self,
            model: WuerstchenModel,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ) -> Tensor:
        losses = self._diffusion_losses(
            batch=batch,
            data=data,
            config=config,
            train_device=self.train_device,
            betas=None,
        )

        k = 1.0
        gamma = 1.0
        alpha_cumprod = self.__alpha_cumprod(data['timestep'], losses.dim())
        p2_loss_weight = (k + alpha_cumprod / (1 - alpha_cumprod)) ** -gamma

        return (losses * p2_loss_weight).mean()
