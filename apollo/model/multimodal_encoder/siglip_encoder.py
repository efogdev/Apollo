
from transformers import SiglipImageProcessor, SiglipVisionConfig, PretrainedConfig
from transformers.models.siglip.modeling_siglip import SiglipVisionModel
from .vision_encoder import VisionTower, VisionTowerS2
import torch

class SiglipVisionTower(VisionTower):
    def __init__(self, model_name_or_path:str, config: PretrainedConfig, vision_config: PretrainedConfig):
        if vision_config is None:
            vision_config = SiglipVisionConfig.from_pretrained(model_name_or_path)

        super().__init__(model_name_or_path, config, vision_config)
        self.vision_config = vision_config
        self.vision_tower_name = model_name_or_path
        self.vision_processor = SiglipImageProcessor.from_pretrained(self.vision_tower_name)
        
        self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name)
        
        self.hidden_size = self.vision_config.hidden_size
        self.W = self.H = self.vision_config.image_size // self.vision_config.patch_size
        self.T = 1

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        return image_features


class SiglipVisionTowerS2(VisionTowerS2):
    def __init__(self, model_name_or_path:str, config: PretrainedConfig, vision_config: PretrainedConfig):
        super().__init__(model_name_or_path, config, vision_config)
        if vision_config is not None:
            self.vision_config = SiglipVisionConfig.from_pretrained(model_name_or_path)
        else: 
            self.vision_config = vision_config
            
        self.vision_tower_name = model_name_or_path
        self.vision_processor = SiglipImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name)
        
        # Make sure it crops/resizes the image to the largest scale in self.scales to maintain high-res information
        self.vision_processor.size['shortest_edge'] = self.scales[-1]
        self.vision_processor.crop_size['height'] = self.vision_processor.crop_size['width'] = self.scales[-1]
        
        self.hidden_size = self.vision_config.hidden_size
        self.W = self.H = self.vision_config.image_size//self.vision_config.patch_size
        self.T = 1

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        return image_features