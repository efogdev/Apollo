import torch
from .vision_encoder import VisionTower
from transformers import PretrainedConfig
from .vision_processors import ProcessorWrapper

from torchvision import transforms
from .utils import volume_transforms, transforms as video_transforms

from .languagebind.languagebind_video import LanguageBindVideoEncoder, LanguageBindVideoEncoderConfig
from .languagebind.languagebind_image import LanguageBindImageEncoder, LanguageBindImageEncoderConfig, LanguageBindImageProcessor


OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


class LanguageBindImageTower(VisionTower):
    def __init__(self, model_name_or_path:str, config: PretrainedConfig, vision_config: PretrainedConfig):
        if vision_config is None:
            vision_config = LanguageBindImageEncoderConfig.from_pretrained(model_name_or_path)
            
        super().__init__(model_name_or_path, config, vision_config)
        self.vision_config = vision_config

        self.vision_tower_name = model_name_or_path
        vision_processor = LanguageBindImageProcessor(vision_config)

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                transforms.CenterCrop(224),
                transforms.Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD)  # assume image
            ]
        )
        self.vision_processor = ProcessorWrapper(processor=vision_processor,
                                                 height=self.vision_config.image_size, 
                                                 width=self.vision_config.image_size, 
                                                 image_mean=OPENAI_DATASET_MEAN)
        
        self.vision_tower = LanguageBindImageEncoder.from_pretrained(model_name_or_path)
        self.hidden_size = self.vision_config.hidden_size
        self.W = self.H = self.vision_config.image_size // self.vision_config.patch_size
        self.T = 1

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.embeddings.class_embedding.dtype

    @property
    def device(self):
        return self.vision_tower.embeddings.class_embedding.device

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


class LanguageBindVideoTower(VisionTower):
    def __init__(self, model_name_or_path:str, config: PretrainedConfig, vision_config: PretrainedConfig):
        if vision_config is None:
            vision_config = LanguageBindVideoEncoderConfig.from_pretrained(model_name_or_path)

        super().__init__(model_name_or_path, config, vision_config)
        self.vision_config = vision_config
        self.vision_tower_name = model_name_or_path
        transform = video_transforms.Compose(
            [
                video_transforms.Resize(self.vision_config.image_size, interpolation='bilinear'),
                video_transforms.CenterCrop(size=(self.vision_config.image_size, self.vision_config.image_size)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
            ]
        )
        self.vision_processor = ProcessorWrapper(transform=transform, 
                                                 height=self.vision_config.image_size, 
                                                 width=self.vision_config.image_size, 
                                                 frames_per_clip=self.vision_config.num_frames, 
                                                 image_mean=OPENAI_DATASET_MEAN)
        
        self.vision_tower = LanguageBindVideoEncoder.from_pretrained(model_name_or_path)
        
        self.hidden_size = self.vision_config.hidden_size
        self.W = self.H = self.vision_config.image_size // self.vision_config.patch_size
        self.T = self.vision_config.num_frames 
        self.num_frames = self.vision_config.num_frames

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer] ## (b t) n d
        BT, N, _ = image_features.shape
        B = BT // self.T
        image_features = image_features.view(B, self.T, N, self.hidden_size) ## b t n d

        if self.select_feature == 'patch':
            image_features = image_features[..., 1:, :]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    def video_forward(self, video):
        if video.shape[-3] < self.num_frames:
            video = video.repeat_interleave(self.num_frames, dim=-3)
        elif video.shape[-3] > self.num_frames:
            video = video[:,:,::video.shape[-3]//self.num_frames,...]

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

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.embeddings.class_embedding.dtype  #############

    @property
    def device(self):
        return self.vision_tower.embeddings.class_embedding.device  ##############

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
