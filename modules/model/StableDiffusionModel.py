from diffusers import AutoencoderKL, UNet2DConditionModel, StableDiffusionDepth2ImgPipeline, \
    StableDiffusionInpaintPipeline, StableDiffusionPipeline, DiffusionPipeline, DDIMScheduler
from torch import Tensor
from transformers import CLIPTextModel, CLIPTokenizer, DPTImageProcessor, DPTForDepthEstimation

from modules.model.BaseModel import BaseModel
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.TrainProgress import TrainProgress
from modules.util.convert.rescale_noise_scheduler_to_zero_terminal_snr import \
    rescale_noise_scheduler_to_zero_terminal_snr
from modules.util.enum.ModelType import ModelType
from modules.util.modelSpec.ModelSpec import ModelSpec


class StableDiffusionModelEmbedding:
    def __init__(self, name: str, vector: Tensor, token_count: int):
        self.name = name
        self.vector = vector
        self.token_count = token_count


class StableDiffusionModel(BaseModel):
    # base model data
    model_type: ModelType
    tokenizer: CLIPTokenizer
    noise_scheduler: DDIMScheduler
    text_encoder: CLIPTextModel
    vae: AutoencoderKL
    unet: UNet2DConditionModel
    image_depth_processor: DPTImageProcessor
    depth_estimator: DPTForDepthEstimation

    # persistent training data
    embeddings: list[StableDiffusionModelEmbedding] | None
    text_encoder_lora: LoRAModuleWrapper | None
    unet_lora: LoRAModuleWrapper | None
    sd_config: dict | None

    def __init__(
            self,
            model_type: ModelType,
            tokenizer: CLIPTokenizer,
            noise_scheduler: DDIMScheduler,
            text_encoder: CLIPTextModel,
            vae: AutoencoderKL,
            unet: UNet2DConditionModel,
            image_depth_processor: DPTImageProcessor | None = None,
            depth_estimator: DPTForDepthEstimation | None = None,
            optimizer_state_dict: dict | None = None,
            ema_state_dict: dict | None = None,
            train_progress: TrainProgress = None,
            embeddings: list[StableDiffusionModelEmbedding] = None,
            text_encoder_lora: LoRAModuleWrapper | None = None,
            unet_lora: LoRAModuleWrapper | None = None,
            sd_config: dict | None = None,
            model_spec: ModelSpec | None = None,
    ):
        super(StableDiffusionModel, self).__init__(
            model_type=model_type,
            optimizer_state_dict=optimizer_state_dict,
            ema_state_dict=ema_state_dict,
            train_progress=train_progress,
            model_spec=model_spec,
        )

        self.tokenizer = tokenizer
        self.noise_scheduler = noise_scheduler
        self.text_encoder = text_encoder
        self.vae = vae
        self.unet = unet
        self.image_depth_processor = image_depth_processor
        self.depth_estimator = depth_estimator

        self.embeddings = embeddings if embeddings is not None else []
        self.text_encoder_lora = text_encoder_lora
        self.unet_lora = unet_lora
        self.sd_config = sd_config

    def create_pipeline(self) -> DiffusionPipeline:
        if self.model_type.has_depth_input():
            return StableDiffusionDepth2ImgPipeline(
                vae=self.vae,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                unet=self.unet,
                scheduler=self.noise_scheduler,
                depth_estimator=self.depth_estimator,
                feature_extractor=self.image_depth_processor,
            )
        elif self.model_type.has_conditioning_image_input():
            return StableDiffusionInpaintPipeline(
                vae=self.vae,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                unet=self.unet,
                scheduler=self.noise_scheduler,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
            )
        else:
            return StableDiffusionPipeline(
                vae=self.vae,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                unet=self.unet,
                scheduler=self.noise_scheduler,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
            )

    def force_v_prediction(self):
        self.noise_scheduler.config.prediction_type = 'v_prediction'
        self.sd_config['model']['params']['parameterization'] = 'v'

    def force_epsilon_prediction(self):
        self.noise_scheduler.config.prediction_type = 'epsilon'
        self.sd_config['model']['params']['parameterization'] = 'epsilon'

    def rescale_noise_scheduler_to_zero_terminal_snr(self):
        rescale_noise_scheduler_to_zero_terminal_snr(self.noise_scheduler)
