# This file is modified from https://github.com/haotian-liu/LLaVA/
from .vision_encoder import VisionTower
from transformers import (
    PretrainedConfig,
    Dinov2Model,
    AutoImageProcessor,
    Dinov2Config,
)
from .vision_processors import ProcessorWrapper
import torch


class DinoVisionTower(VisionTower):
    def __init__(self, model_name_or_path: str, config: PretrainedConfig, vision_config: PretrainedConfig):
        if vision_config is None:
            vision_config = Dinov2Config.from_pretrained(model_name_or_path)

        super().__init__(model_name_or_path, config, vision_config)
        self.vision_config = vision_config
        vision_processor = AutoImageProcessor.from_pretrained(model_name_or_path)
        
        self.vision_processor = ProcessorWrapper(processor=vision_processor,
                                                 height=vision_processor.crop_size['height'],
                                                 width=vision_processor.crop_size['width'],
                                                 image_mean=vision_processor.image_mean)

        self.vision_tower = Dinov2Model.from_pretrained(model_name_or_path, torch_dtype=eval(config.model_dtype))
        self.hidden_size = self.vision_config.hidden_size
        self.W = self.H = vision_processor.crop_size['height']//self.vision_config.patch_size
        self.T = 1