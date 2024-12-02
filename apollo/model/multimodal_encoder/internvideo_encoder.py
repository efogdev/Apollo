from transformers import AutoConfig, AutoModel, PretrainedConfig
from .vision_encoder import VisionTower
from .vision_processors import ProcessorWrapper

from .internvideo.config import Config, eval_dict_leaf
from .internvideo.utils import setup_internvideo2
from .internvideo.internvideo2 import pretrain_internvideo2_1b_patch14_224
from .internvideo.pos_embed import interpolate_pos_embed_internvideo2_new
from .jepa.datasets.utils.video import transforms, volume_transforms

import torch


from safetensors import safe_open

def load_safe_tensors(model_path):
    tensors = {}
    with safe_open(f"{model_path}/model.safetensors", framework="pt", device=0) as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    return tensors

normalize = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


class InternVideoTower(VisionTower):
    def __init__(self, model_name_or_path: str, config: PretrainedConfig, vision_config: PretrainedConfig):
        if vision_config is None:
            vision_config = AutoConfig.from_pretrained(model_name_or_path)

        super().__init__(model_name_or_path, config, vision_config)
        self.vision_config = vision_config
        model = pretrain_internvideo2_1b_patch14_224(vision_config)
        state_dict = load_safe_tensors(model_name_or_path)
        interpolate_pos_embed_internvideo2_new(state_dict, model, orig_t_size=4)
        message = model.load_state_dict(state_dict, strict=False)
        print(message)
        self.vision_tower = model.to(dtype=eval(config.model_dtype))

        transform = transforms.Compose([
            transforms.Resize(self.vision_config.img_size, interpolation='bilinear'),
            transforms.CenterCrop(size=(self.vision_config.img_size, self.vision_config.img_size)),
            volume_transforms.ClipToTensor(),
            transforms.Normalize(mean=normalize[0], std=normalize[1])
        ])

        self.vision_processor = ProcessorWrapper(transform=transform, 
                                                 height=self.vision_config.img_size, 
                                                 width=self.vision_config.img_size, 
                                                 frames_per_clip=self.vision_config.num_frames, 
                                                 image_mean=normalize[0])

        self.W = self.H = vision_config.img_size // vision_config.patch_size
        self.T = self.vision_config.num_frames // self.vision_config.tubelet_size 
        self.num_frames = self.vision_config.num_frames 
        self.hidden_size = vision_config.d_model
        
    def feature_select(self, image_features):
        if self.select_feature == "patch":
            image_features = image_features[:, 1:]
        elif self.select_feature == "cls_patch":
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
        return image_features

    def video_forward(self, video):
        if video.shape[-3] < self.num_frames:
            video = video.repeat_interleave(self.num_frames, dim=-3)
        elif video.shape[-3] > self.num_frames:
            video = video[:,:,::video.shape[-3]//self.num_frames,...]

        video_feature = self.vision_tower(video.to(device=self.device, dtype=self.dtype), x_vis_return_idx=self.select_layer,  x_vis_only=True)
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
    def device(self):
        return self.vision_tower.pos_embed.device