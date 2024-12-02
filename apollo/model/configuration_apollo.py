from transformers import PretrainedConfig


class ApolloConfig(PretrainedConfig):
    model_type = "apollo"

    def __init__(
        self,
        llm_cfg=None,
        vision_tower_cfg=None,
        mm_connector_cfg=None,
        architectures=None,
        resume_path=None,
        mm_hidden_size=None,
        image_aspect_ratio=None,
        num_video_frames=None,
        mm_vision_select_layer=None,
        mm_vision_select_feature=None,
        use_mm_start_end=False,
        use_mm_patch_token=True,
        mm_connector_lr=None,
        vision_resolution=None,
        interpolate_mode=None,
        clip_duration=None,
        **kwargs
    ):
        super().__init__()
        
        self.architectures = architectures
        self.llm_cfg = llm_cfg
        self.vision_tower_cfg = vision_tower_cfg
        self.mm_connector_cfg = mm_connector_cfg
        self.resume_path = resume_path
        
        self.mm_hidden_size = mm_hidden_size
        self.image_aspect_ratio = image_aspect_ratio
        self.num_video_frames = num_video_frames
        self.mm_vision_select_layer = mm_vision_select_layer
        self.mm_vision_select_feature = mm_vision_select_feature
        self.use_mm_start_end = use_mm_start_end
        self.use_mm_patch_token = use_mm_patch_token
        self.mm_connector_lr = mm_connector_lr
        self.vision_resolution = vision_resolution
        self.interpolate_mode = interpolate_mode
        self.clip_duration = clip_duration