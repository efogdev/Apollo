import torch
from torch import nn
from .jepa import VJEPAModel
from typing import Dict, List
from .utils import transforms, volume_transforms
from .vision_processors import VideoProcessor, ProcessorWrapper
from .vision_encoder import VisionTower
from transformers import PretrainedConfig, AutoConfig

normalize=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


class VJepaVisionTower(VisionTower):
    def __init__(self, model_name_or_path: str, config: PretrainedConfig, vision_config: PretrainedConfig = None):
        # Prepare the vision configuration by either using the provided one or loading from pretrained
        if vision_config is None:
            vision_config = AutoConfig.from_pretrained(model_name_or_path)
        
        # Initialize the superclass with the necessary configurations
        super().__init__(model_name_or_path, config, vision_config)
        self.vision_config = vision_config

        self.model_name_or_path = model_name_or_path
        self.vision_tower = VJEPAModel.from_pretrained(model_name_or_path, self.vision_config)

        preprocess = transforms.Compose([
            transforms.Resize(self.vision_config.crop_size, interpolation='bilinear'),
            transforms.CenterCrop(size=(self.vision_config.crop_size, self.vision_config.crop_size)),
            volume_transforms.ClipToTensor(),
            transforms.Normalize(mean=normalize[0], std=normalize[1])
        ])

        self.vision_processor = ProcessorWrapper(preprocess, 
                                                 height=self.vision_config.crop_size, 
                                                 width=self.vision_config.crop_size, 
                                                 frames_per_clip=self.vision_config.num_frames, 
                                                 image_mean=normalize[0])
        
        self.vision_config.embed_dim = self.vision_tower.embed_dim 
        self.vision_config.num_heads = self.vision_tower.num_heads 
        self.W = self.H = self.vision_config.crop_size//self.vision_config.patch_size
        self.T = self.vision_config.num_frames // self.vision_config.tubelet_size
        self.hidden_size = self.vision_config.embed_dim
        self.num_frames = self.vision_config.num_frames
        self.encode_batch_size = getattr(config, "encode_batch_size", 0)
        self.num_encode_batch = getattr(config, "num_encode_batch", 0)
        
    def _forward(self, videos, out_T=1):
        if type(videos) is list:
            video_features = []
            for video in videos:
                video_feature = self.video_forward(video)
                video_features.append(video_feature)
        else:
            if videos.shape[-3] != self.num_frames:
                videos = videos.repeat_interleave(self.num_frames, dim=-3)
                
            video_features = self.vision_tower(videos.to(device=self.device, dtype=self.dtype))
            video_features = video_features.reshape(video_features.shape[0], self.T, self.W, self.H, self.hidden_size)
        return video_features
    
    @property
    def device(self):
        # Assume vision_tower has a parameter named embeddings for simplicity
        return self.vision_tower.pos_embed.device

    @property
    def dtype(self):
        return self.vision_tower.pos_embed.dtype
                
    @property
    def num_heads(self):
        return self.vision_tower.num_heads
        
    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2