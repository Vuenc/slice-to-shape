import sys
from typing import Union, Literal, Optional
import torch
import torch.nn as nn
import numpy as np

sys.path.append("pytorch-3dunet")
from pytorch3dunet.unet3d.model import UNet3D

class SdfVoxelMatchNet(nn.Module):
    def __init__(self, embedding_size=128,  ultrasound_patches_shape=(32,32,32), sdf_patches_shape=(32,32,32),
            normalize_embeddings=True,
            encoder_type: Union[Literal["3dunet-encoder"], Literal["3dunet-encoder-decoder"]] = "3dunet-encoder"):
        super().__init__()
        unet3d_1 = UNet3D(in_channels=1, out_channels=1) #, out_channels=embedding_size)
        unet3d_2 = UNet3D(in_channels=1, out_channels=1) #, out_channels=embedding_size)
        
        # unet3d_3 = UNet3D(in_channels=1, out_channels=embedding_size)
        # unet3d_4 = UNet3D(in_channels=1, out_channels=embedding_size)
        

        self.encoder_type = encoder_type

        if encoder_type == "3dunet-encoder":
            self.ultrasound_encoder = nn.Sequential(*unet3d_1.encoders)
            self.sdf_encoder = nn.Sequential(*unet3d_2.encoders)
            
        elif encoder_type == "3dunet-encoder-decoder":
            raise NotImplementedError()
            # self.ultrasound_unet = nn.Sequential(unet3d_3)
            # self.sdf_unet = nn.Sequential(unet3d_4)
        else:
            raise ValueError("`encoder_type` must be one of: ['3dunet-encoder', '3dunet-encoder-decoder']")

        self.flatten = nn.Flatten()

        # calculate encoder output sizes by forward calls
        ultrasound_num_output_features = np.prod(self.ultrasound_encoder.forward(torch.zeros((1, 1, *ultrasound_patches_shape))).shape[1:])
        sdf_num_output_features = np.prod(self.sdf_encoder.forward(torch.zeros((1, 1, *sdf_patches_shape))).shape[1:])

        self.ultrasound_linear = nn.Sequential(
            nn.Linear(ultrasound_num_output_features, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size))
        self.sdf_linear = nn.Sequential(
            nn.Linear(sdf_num_output_features, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size))
        
        self.normalize_embeddings = normalize_embeddings

    # `ultrasound_patch` has shape [minibatch_size, width_x, width_y, width_z]
    # `sdf_patch` has shape [minibatch_size, width_x, width_y, width_z]
    def forward(self, ultrasound_patch, sdf_patch):
        return (self.forward_ultrasound(ultrasound_patch), self.forward_sdf(sdf_patch))
        #return (self.ultrasound_unet(ultrasound_patch), self.sdf_unet(sef_patch))
    
    def forward_ultrasound(self, ultrasound_patch):
        out_ultrasound = self.ultrasound_encoder(ultrasound_patch.unsqueeze(1))
        out_ultrasound = self.ultrasound_linear(self.flatten(out_ultrasound))

        if self.normalize_embeddings:
            out_ultrasound = out_ultrasound / torch.norm(out_ultrasound, dim=1, keepdim=True)
        
        return out_ultrasound

    
    def forward_sdf(self, sdf_patch):
        out_sdf = self.sdf_encoder(sdf_patch.unsqueeze(1))
        out_sdf = self.sdf_linear(self.flatten(out_sdf))

        if self.normalize_embeddings:
            out_sdf = out_sdf / torch.norm(out_sdf, dim=1, keepdim=True)
            
        return out_sdf


    

    def add_sink_value_parameter(self, value: Optional[torch.Tensor]=None):
        if value is None:
            value = torch.randn(1)
        self.sink_value = nn.parameter.Parameter(value)
