# This file is modified from https://github.com/haotian-liu/LLaVA/
from .vision_encoder import VisionTower, VisionTowerS2
from transformers import (
    PretrainedConfig,
    CLIPVisionModel,
    CLIPImageProcessor,
    CLIPVisionConfig,
)
from .vision_processors import ProcessorWrapper
import torch

class CLIPVisionTower(VisionTower):
    def __init__(self, model_name_or_path: str, config: PretrainedConfig, vision_config: PretrainedConfig):
        if vision_config is None:
            vision_config = CLIPVisionConfig.from_pretrained(model_name_or_path)

        super().__init__(model_name_or_path, config, vision_config)
        self.vision_config = vision_config    
        vision_processor = CLIPImageProcessor.from_pretrained(model_name_or_path)
        self.vision_processor = ProcessorWrapper(processor=vision_processor,
                                                 height=self.vision_config.image_size, 
                                                 width=self.vision_config.image_size, 
                                                 image_mean=vision_processor.image_mean)

        self.vision_tower = CLIPVisionModel.from_pretrained(
            model_name_or_path, torch_dtype=eval(config.model_dtype)
        )

        self.hidden_size = self.vision_config.hidden_size
        self.W = self.H = self.vision_config.image_size // self.vision_config.patch_size
        self.T = 1


class CLIPVisionTowerS2(VisionTowerS2):
    def __init__(self, model_name_or_path: str, config: PretrainedConfig, vision_config: PretrainedConfig):
        super().__init__(model_name_or_path, config, vision_config)
        if vision_config is None:
            self.vision_config = CLIPVisionConfig.from_pretrained(model_name_or_path)
        else: 
            self.vision_config = vision_config
        self.vision_processor = CLIPImageProcessor.from_pretrained(model_name_or_path)
        self.vision_tower = CLIPVisionModel.from_pretrained(
            model_name_or_path, torch_dtype=eval(config.model_dtype)
        )

        # Make sure it crops/resizes the image to the largest scale in self.scales to maintain high-res information
        self.vision_processor.size['shortest_edge'] = self.scales[-1]
        self.vision_processor.crop_size['height'] = self.vision_processor.crop_size['width'] = self.scales[-1]
        self.hidden_size = self.vision_config.hidden_size
        self.W = self.H = self.vision_config.image_size//self.vision_config.patch_size
        self.T = 1
