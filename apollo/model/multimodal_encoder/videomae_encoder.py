from .vision_encoder import VisionTower, VisionTowerS2
from .jepa.datasets.utils.video import transforms, volume_transforms


from transformers import (
    PretrainedConfig,
    VideoMAEModel,
    VideoMAEImageProcessor,
    VideoMAEConfig,
)
from .vision_processors import ProcessorWrapper
import torch
from torchvision.transforms import Lambda


normalize = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

class MAEVideoTower(VisionTower):
    def __init__(self, model_name_or_path: str, config: PretrainedConfig, vision_config: PretrainedConfig):
        if vision_config is None:
            vision_config = VideoMAEConfig.from_pretrained(model_name_or_path)

        super().__init__(model_name_or_path, config, vision_config)
        self.vision_config = vision_config
        vision_processor = VideoMAEImageProcessor.from_pretrained(model_name_or_path)
        transform = transforms.Compose([
            transforms.Resize(self.vision_config.image_size, interpolation='bilinear'),
            transforms.CenterCrop(size=(self.vision_config.image_size, self.vision_config.image_size)),
            volume_transforms.ClipToTensor(),
            transforms.Normalize(mean=normalize[0], std=normalize[1]),
            Lambda(lambda img: img.swapaxes(0, 1))
        ])

        self.vision_processor = ProcessorWrapper(transform=transform,
                                                 height=self.vision_config.image_size,
                                                 width=self.vision_config.image_size,
                                                 frames_per_clip=self.vision_config.num_frames,
                                                 image_mean=normalize[0])

        self.vision_tower = VideoMAEModel.from_pretrained(model_name_or_path, torch_dtype=eval(config.model_dtype))
        self.hidden_size = self.vision_config.hidden_size
        self.W = self.H = self.vision_config.image_size // self.vision_config.patch_size
        self.T = self.vision_config.num_frames // self.vision_config.tubelet_size
        self.num_frames = self.vision_config.num_frames

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        return image_features

    def video_forward(self, video):
        if video.shape[-4] < self.num_frames:
            video = video.repeat_interleave(self.num_frames, dim=-4)
        elif video.shape[-4] > self.num_frames:
            video = video[:,::video.shape[-4]//self.num_frames,...]
        
        video_feature = self.vision_tower(video.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        video_feature = self.feature_select(video_feature)
        video_feature = video_feature.reshape(video_feature.shape[0], self.T, self.W, self.H, self.hidden_size)
        return video_feature
        
    def _forward(self, videos, out_T=1):
        if type(videos) is list:
            video_features = []
            for video in videos:
                video_feature = self.video_forward(video)
                video_features.append(video_feature)
        else:
            video_features = self.video_forward(videos)
        return video_features