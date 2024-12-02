import torch.nn as nn
from typing import Dict, List



class ImageAveragePooler(nn.Module):
    def __init__(self, image_size, patch_size, tokens_per_image, embedding_dim):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.tokens_per_image = tokens_per_image
        self.embeddings_per_image = tokens_per_image
        self.embedding_dim = embedding_dim

        # Calculate the dimensions of the grid after reshaping
        self.grid_size = image_size // patch_size  # This is the size of the grid (height and width)

        # Calculate the kernel size and stride for the pooling layer to match the required number of tokens
        width = int(self.embeddings_per_image**0.5)
        hieght = self.embeddings_per_image//width
        self.pool = nn.AdaptiveAvgPool2d((width, hieght))

    def forward(self, x, att_mask=None):
        #x = x[:, 1:] ## remove global token
        # Reshape to [batch, channels, height, width] - treat each embedding dimension as a channel
        x = x.view(-1, self.grid_size, self.grid_size, self.embedding_dim)
        x = x.permute(0, 3, 1, 2)  # Reorder to [batch, channels, height, width] for applyinnn.Conv2d or nn.AvgPool2d

        # Apply average pooling
        x = self.pool(x)

        # Flatten the last two dimensions and then permute to match desired output
        x = x.flatten(start_dim=2)  # Flatten height and width into a single dimension
        x = x.permute(0, 2, 1)  # Swap the last two dimensions
        return x
        

class VideoAveragePooler(nn.Module):
    def __init__(self, frames_per_clip, tublet_size, image_size, patch_size, tokens_per_clip, embedding_dim):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.tokens_per_clip = tokens_per_clip
        self.embeddings_per_image = tokens_per_clip
        self.embedding_dim = embedding_dim
        
        width = hieght = image_size // patch_size
        time = frames_per_clip // tublet_size
        self.width = width
        self.hieght = hieght
        self.time = time

        if time > tokens_per_clip:
            width = hieght = 1
            time = tokens_per_clip
        else: 
            width = int((tokens_per_clip // time)**0.5)
            hieght = tokens_per_clip //(width * time)
            
        self.pool = nn.AdaptiveAvgPool3d((time, width, hieght))

    def forward(self, x, att_mask=None):
        # Reshape to [batch, channels, height, width] - treat each embedding dimension as a channel
        x = x.contiguous().view(-1, self.time, self.width, self.hieght, self.embedding_dim)
        x = x.permute(0, 4, 1, 2, 3)  # Reorder to [batch, channels, height, width] for applyinnn.Conv2d or nn.AvgPool2d

        # Apply average pooling
        x = self.pool(x)

        # Flatten the last two dimensions and then permute to match desired output
        x = x.flatten(start_dim=2)  # Flatten height and width into a single dimension
        x = x.permute(0, 2, 1)  # Swap the last two dimensions
        return x

class VideoAveragePooler(nn.Module):
    def __init__(self, frames_per_clip, tublet_size, image_size, patch_size, tokens_per_clip, embedding_dim):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.tokens_per_clip = tokens_per_clip
        self.embeddings_per_image = tokens_per_clip
        self.embedding_dim = embedding_dim
        
        width = hieght = image_size // patch_size
        time = frames_per_clip // tublet_size
        self.width = width
        self.hieght = hieght
        self.time = time

        if time > tokens_per_clip:
            width = hieght = 1
            time = tokens_per_clip
        else: 
            width = int((tokens_per_clip // time)**0.5)
            hieght = tokens_per_clip //(width * time)
            
        self.pool = nn.AdaptiveAvgPool3d((time, width, hieght))

    def forward(self, x, att_mask=None):
        # Reshape to [batch, channels, height, width] - treat each embedding dimension as a channel
        x = x.contiguous().view(-1, self.time, self.width, self.hieght, self.embedding_dim)
        x = x.permute(0, 4, 1, 2, 3)  # Reorder to [batch, channels, height, width] for applyinnn.Conv2d or nn.AvgPool2d

        # Apply average pooling
        x = self.pool(x)

        # Flatten the last two dimensions and then permute to match desired output
        x = x.flatten(start_dim=2)  # Flatten height and width into a single dimension
        x = x.permute(0, 2, 1)  # Swap the last two dimensions
        return x


class VideoAveragePooler(nn.Module):
    def __init__(self, frames_per_clip, tublet_size, image_size, patch_size, tokens_per_clip, embedding_dim):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.tokens_per_clip = tokens_per_clip
        self.embeddings_per_image = tokens_per_clip
        self.embedding_dim = embedding_dim
        
        width = hieght = image_size // patch_size
        time = frames_per_clip // tublet_size
        self.width = width
        self.hieght = hieght
        self.time = time

        if time > tokens_per_clip:
            width = hieght = 1
            time = tokens_per_clip
        else: 
            width = int((tokens_per_clip // time)**0.5)
            hieght = tokens_per_clip //(width * time)
            
        self.pool = nn.AdaptiveAvgPool3d((time, width, hieght))

    def forward(self, x, att_mask=None):
        # Reshape to [batch, channels, height, width] - treat each embedding dimension as a channel
        x = x.contiguous().view(-1, self.time, self.width, self.hieght, self.embedding_dim)
        x = x.permute(0, 4, 1, 2, 3)  # Reorder to [batch, channels, height, width] for applyinnn.Conv2d or nn.AvgPool2d

        # Apply average pooling
        x = self.pool(x)

        # Flatten the last two dimensions and then permute to match desired output
        x = x.flatten(start_dim=2)  # Flatten height and width into a single dimension
        x = x.permute(0, 2, 1)  # Swap the last two dimensions
        return x


class AveragePooler(nn.Module):
    def __init__(self, connector_config, **kwargs):
        super().__init__()
        self.config = connector_config
        self.token_input_shape = connector_config.token_input_shape
        self.token_output_shape = connector_config.token_output_shape
        self.num_output_tokens = connector_config.num_output_tokens
        self.pool = nn.AdaptiveAvgPool3d(self.token_output_shape)

    def forward(self, x, att_mask=None):
        # Apply average pooling
        x = x.permute(0, 4, 1, 2, 3)
        x = self.pool(x)
        # Flatten the last two dimensions and then permute to match desired output
        x = x.flatten(start_dim=2)  # Flatten height and width into a single dimension
        x = x.permute(0, 2, 1)  # Swap the last two dimensions
        return x