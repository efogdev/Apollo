from .models import vision_transformer as vit
from ..utils import transforms as video_transforms
from ..utils import volume_transforms as volume_transforms
from decord import VideoReader, cpu
import numpy as np
from typing import Dict, List, Union
import torch
import os, yaml, math

from transformers import PretrainedConfig
from apollo.utils import mprint

class JEPAVisionConfig(PretrainedConfig):
    model_type = "vjepa_model"
    def __init__(
        self,
        crop_size=224,
        num_frames=16,
        tubelet_size=2,
        patch_size=555,
        uniform_power=True,
        use_silu=False,
        tight_silu=False,
        use_sdpa=True,
        modality="video",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.crop_size = crop_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size
        self.uniform_power = uniform_power
        self.use_silu = use_silu
        self.tight_silu = tight_silu
        self.use_sdpa = use_sdpa
        self.image_size = self.crop_size 
        self.modality=modality

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "JEPAVisionConfig":
        cls._set_token_in_kwargs(kwargs)
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        return cls.from_dict(config_dict, **kwargs)

    @classmethod
    def get_config_dict(cls, pretrained_model_name_or_path, **kwargs):
        config_file = pretrained_model_name_or_path.split(".pth")[0] + ".yaml"
        with open(config_file, 'r') as file:
            vision_config = yaml.safe_load(file)
        
        return vision_config, kwargs
    


def pad_to_center_square(frames, mean_values):
    """
    Pad the given frame or frames numpy array to square dimensions using the mean values as the padding color.
    Handles both single frames (H, W, C) and batches of frames (N, H, W, C).

    Args:
        frames (np.array): The input frame array of shape (H, W, C) or (N, H, W, C).
        mean_values (tuple): Mean values for each channel, typically derived from dataset normalization parameters.

    Returns:
        np.array: The padded frame array with square dimensions.
    """
    if frames.ndim == 3:  # Single frame
        frames = frames[np.newaxis, :]  # Add a batch dimension
    elif frames.ndim != 4:
        raise ValueError("Input array must be either of shape (H, W, C) or (N, H, W, C)")

    N, height, width, channels = frames.shape
    size = max(width, height)
    background_color = np.array(mean_values, dtype=frames.dtype)
    
    # Create a background array with the size and fill it with the mean values
    padded_frames = np.full((N, size, size, channels), background_color, dtype=frames.dtype)

    # Calculate padding offsets
    top, left = (size - height) // 2, (size - width) // 2

    # Place the original frames in the center of the square canvas
    padded_frames[:, top:top + height, left:left + width, :] = frames
    return padded_frames




class VJEPAModel(object):
    def load_pretrained(
        encoder,
        pretrained,
        checkpoint_key='target_encoder'
    ):
        mprint(f'Loading pretrained model from {pretrained}')
        
        if pretrained.endswith(".pth.tar"):
            pretrained_dict = torch.load(pretrained, map_location='cpu')
            try:
                pretrained_dict = pretrained_dict[checkpoint_key]
            except Exception:
                pretrained_dict = pretrained_dict['encoder']
        else:
            if os.path.exists(os.path.join(pretrained, "model.pth.tar")):
                pretrained_dict = torch.load(os.path.join(pretrained, "model.pth.tar"), map_location='cpu')
                if checkpoint_key in pretrained_dict:
                    pretrained_dict = pretrained_dict[checkpoint_key]
                elif 'encoder' in pretrained_dict:
                    pretrained_dict = pretrained_dict['encoder']

            else:
                from safetensors import safe_open
                pretrained_dict = {}
                pretrained = os.path.join(pretrained, 'model.safetensors')
                with safe_open(pretrained, framework="pt", device=0) as f:
                    for k in f.keys():
                        pretrained_dict[k] = f.get_tensor(k)
                
                pretrained_dict = {k.split('vision_tower.')[-1]: v for k, v in pretrained_dict.items()}

    
        pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
        pretrained_dict = {k.replace('backbone.', ''): v for k, v in pretrained_dict.items()}
        for k, v in encoder.state_dict().items():
            try:
                if k not in pretrained_dict:
                    mprint(f'key "{k}" could not be found in loaded state dict')
                elif pretrained_dict[k].shape != v.shape:
                    mprint(f'key "{k}" is of different shape in model and loaded state dict')
                    pretrained_dict[k] = v
            except:
                import ipdb; ipdb.set_trace()
        msg = encoder.load_state_dict(pretrained_dict, strict=False)
        #print(encoder)
        mprint(f'loaded pretrained model with msg: {msg}')
        return encoder 
        
    def from_pretrained(
        pretrained_path,
        config,
        checkpoint_key='target_encoder'
    ):
        encoder = vit.__dict__[config.model_name](
            img_size=config.crop_size,
            patch_size=config.patch_size,
            num_frames=config.num_frames,
            tubelet_size=config.tubelet_size,
            uniform_power=config.uniform_power,
            use_sdpa=config.use_sdpa,
            use_silu=config.use_silu,
            tight_silu=config.tight_silu,
            checkpoint_key = checkpoint_key,
            model_name = config.model_name,
            _name_or_path = pretrained_path,
            model_type = 'vit'
        )
        encoder = VJEPAModel.load_pretrained(encoder=encoder, pretrained=pretrained_path, checkpoint_key=config.checkpoint_key)
        return encoder



normalize=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

class JEPAVideoProcessor(object):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        short_side_size = int(config.crop_size * 256 / 224)
        self.video_mean = normalize[0]
        self.video_std = normalize[1]
        self.image_mean = normalize[0]
        self.image_str = normalize[1]
        self.size = {'height': config.crop_size, 'width': config.crop_size, 'frames_per_clip': config.num_frames}

        self.transform = video_transforms.Compose([
            video_transforms.Resize(short_side_size, interpolation='bilinear'),
            video_transforms.CenterCrop(size=(config.crop_size, config.crop_size)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=normalize[0], std=normalize[1])
        ])

        self.frames_per_clip = config.num_frames
        self.sample_video_fps = config.sample_video_fps
        self.clip_sampling_ratio = 1 ## TODO: should be in config. if >1, then will overap more. if <1, then will more sparcely sample. 
        self.clip_duration = self.frames_per_clip / self.sample_video_fps 
    
    def __call__(self, video_path, return_tensors=None, image_processor=None):
        def split_into_clips(video):
            """ Split video into a list of clips """
            fpc = self.frames_per_clip
            nc = len(video) // self.frames_per_clip
            return [video[i*fpc:(i+1)*fpc] for i in range(nc)]

        buffer, _ = self.loadvideo_decord(video_path)        
        if image_processor:
            center_frames = pad_to_center_square(buffer[::2],  tuple(int(x*255) for x in image_processor.image_mean))
            center_frames = image_processor(center_frames, return_tensors='pt')['pixel_values']
            center_frames = center_frames.reshape(-1, self.frames_per_clip //  2, *center_frames.shape[1:])
            
        buffer = split_into_clips(buffer)
        for i, clip in enumerate(buffer):
            buffer[i] = pad_to_center_square(clip, tuple(int(x*255) for x in normalize[0]))

        if self.transform is not None:
            buffer = [self.transform(clip) for clip in buffer]
        buffer = torch.stack(buffer)
         
        #if self.transform is not None:
        #    buffer = self.transform(buffer)
        ## TODO: do something with clip_indices and validate split_into_clips 
        return buffer, center_frames

    def preprocess(self, video_path, return_tensors, image_processor=None):
        return self.__call__(video_path=video_path, return_tensors=return_tensors, image_processor=image_processor)

    def loadvideo_decord(self, fname):
        """ Load video content using Decord """
        assert os.path.exists(fname), f'video path not found {fname}'
        _fsize = os.path.getsize(fname)
        assert _fsize >= 1 * 1024, f"video too short {fname}"

        vr = VideoReader(fname, num_threads=-1, ctx=cpu(0))            
        # Get the total number of frames and the original fps of the video
        total_frames = len(vr)
        original_fps = vr.get_avg_fps()
        video_duration = total_frames / original_fps
        
        # get the clip indicies
        clip_indices, all_indices = self.calculate_sample_indices(total_frames, original_fps, video_duration)
        
        # Go to start of video
        vr.seek(0) 
        buffer = vr.get_batch(all_indices).asnumpy()
        return buffer, clip_indices
    
    def calculate_sample_indices(self, total_frames, original_fps, video_duration):
        num_clips = math.ceil((video_duration / self.clip_duration) * self.clip_sampling_ratio)
        frame_step = original_fps / self.sample_video_fps
        partition_len = total_frames // num_clips
        all_indices, clip_indices = [], []
        if frame_step > 0.5:
            frame_step = max(1, int(original_fps / self.sample_video_fps)) #was int/floor
            clip_len = int(self.frames_per_clip * frame_step) #was int/floor
            sample_len = min(clip_len, total_frames)
            clip_step = (total_frames - clip_len) // max(1, (num_clips - 1)) if total_frames > clip_len else 0
            for i in range(num_clips):
                if partition_len > clip_len:
                    start_idx = (partition_len - clip_len) // 2
                    end_idx = start_idx + clip_len
                    indices = np.arange(start_idx, end_idx, frame_step)
                    indices = np.clip(indices, 0, partition_len-1).astype(np.int64)
                    indices = indices+ i * partition_len

                else:
                    
                    indices = np.arange(0, sample_len, frame_step)
                    if len(indices) < self.frames_per_clip:
                        padding = np.full(self.frames_per_clip - len(indices), sample_len)
                        indices = np.concatenate((indices, padding))
                        
                    indices = np.clip(indices, 0, sample_len-1).astype(np.int64)
                    indices = indices + i * clip_step

                clip_indices.append(indices)
                all_indices.extend(list(indices))

        else:
            ## original video FPS too low, we need to sample the same frame multiple times. 
            ##  Generally should not happen.
            # Calculate the number of times each frame should be sampled
            num_sample = int(np.ceil(1 / frame_step))
        
            # Compute the effective clip length considering the frame step
            clip_len = int(self.frames_per_clip * frame_step)
        
            # Create an expanded list of indices with each frame repeated num_sample times
            indices = np.repeat(np.arange(clip_len), num_sample)

            # Ensure the clip length does not exceed the total number of frames
            clip_len = min(clip_len, len(indices))
            clip_step = (total_frames - clip_len) // max(1, (num_clips - 1)) if total_frames > clip_len else 0
            
            sample_len = min(clip_len, total_frames)
            if len(indices) < self.frames_per_clip:
                padding = np.full(self.frames_per_clip - len(indices), sample_len)
                indices = np.concatenate((indices, padding))
        
            # Distribute the indices into clips
            for i in range(num_clips):
                current_clip_indices = np.clip(indices, 0, sample_len-1).astype(np.int64)
                current_clip_indices = current_clip_indices + i * clip_step
    
                # Append the current clip indices to the list of all clips
                clip_indices.append(current_clip_indices)
                all_indices.extend(current_clip_indices)
        
        return clip_indices, all_indices

    def __len__(self):
        return len(self.samples)
        