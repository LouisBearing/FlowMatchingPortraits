import numpy as np
import torch
import torch.nn as nn
from .resnet import FeaturePyramidNetwork, DeepMLP

# --------

###
## TODO plus besoin de ça je pense
###
# deep_mlp_layers = {
#     '4': [4],
#     '6': [1, 1, 1, 1, 1, 1],
#     '7': [1, 2, 2, 2], # 12M params
#     '8': [2, 2, 2, 2], # 20M params
#     '10': [1, 3, 3, 3], 
#     '16': [3, 4, 6, 3] # 35M params
# }

# --------

###
## Static parameters, TODO voir ce qu'on garde
###

KEYPOINTS_DIM = 10
MAX_CONTEXT_LENGTH = 40

## Choose to run on GPU or CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --------

###
# TODO: à merger avec trainer.py en 1 seul fichier
# Garder audio encoder ?
# réécrire archi propre de (denoising) transformer multimodal


# class DynamicalModel(nn.Module):
    
#     def __init__(self, config):
#         super(DynamicalModel, self).__init__()
        
#         self.config = config
#         self.h_size = config.hidden_size

#         ## Keypoint encoder
#         self.input_dim = KEYPOINTS_DIM
#         self.emb_dim = self.input_dim * config.data_dim

#         self.encoder_x = nn.Sequential(*[
#             nn.Linear(self.emb_dim, 272),
#             nn.LayerNorm(272),
#             nn.Linear(272, 4 * 272),
#             nn.GELU(),
#             nn.Linear(4 * 272, config.coord_dim)
#         ])

#         ## Audio encoder
#         fpn_inner_dim = config.audio_dim
#         self.encoder_a = FeaturePyramidNetwork(config.audio_dim, fpn_inner_dim, bottom_up_only=True)

#         ## Deep output net / Multi-scale module
#         do_in_dim = self.h_size + config.audio_dim + config.coord_dim
#         self.instantiate_deep_output_net(do_in_dim)

#         ## Temporal module
#         tf_layer = nn.TransformerEncoderLayer(d_model=config.coord_dim, nhead=config.nheads_tf, dim_feedforward=1024, dropout=0)
#         self.temp_module = nn.TransformerEncoder(tf_layer, num_layers=config.nlayers_tf)
#         self.lin_out = nn.Linear(config.coord_dim, self.h_size)


#     def instantiate_deep_output_net(self, in_dim):

#         groups = self.config.pyramid_layers_g

#         self.deep_output = DeepMLP(in_dim * groups, 272 * groups, layers=deep_mlp_layers[str(self.config.nblocks_do)][:-1], expansion_fact=self.config.expansion_fact_do, 
#             groups=groups)
#         self.mask = DeepMLP(272 * groups, self.input_dim * groups, layers=[deep_mlp_layers[str(self.config.nblocks_do)][-1]], expansion_fact=self.config.expansion_fact_do, 
#             groups=groups)
#         self.leaf = DeepMLP(272 * groups, self.input_dim * self.config.data_dim * groups, layers=[deep_mlp_layers[str(self.config.nblocks_do)][-1]], 
#             expansion_fact=self.config.expansion_fact_do, groups=groups)

#         # Bias term
#         groups += 1
#         self.deep_output_bias = DeepMLP(in_dim - self.config.audio_dim, 272, layers=deep_mlp_layers[str(self.config.nblocks_do)][:-1],
#             expansion_fact=self.config.expansion_fact_do)
#         self.mask_bias = DeepMLP(272, self.input_dim, layers=[deep_mlp_layers[str(self.config.nblocks_do)][-1]], expansion_fact=self.config.expansion_fact_do)
#         self.leaf_bias = DeepMLP(272, self.input_dim * self.config.data_dim, layers=[deep_mlp_layers[str(self.config.nblocks_do)][-1]],
#             expansion_fact=self.config.expansion_fact_do)

#         # Final layer
#         self.final_block = nn.Linear(self.input_dim * self.config.data_dim * groups, self.input_dim * self.config.data_dim)


#     def embed_audio(self, audio):
#         '''
#         Audio pyramid input are processed by independent CNN encoders with increasing recep. field, then interpolated in the embedding space 
#         back to a sequence of same duration as the higher res input --> ideally, audio inputs should get smoother as one moves upward in the pyramid
#         '''

#         tgt_len = int(audio.shape[1] / 4)
#         fpn_out = self.encoder_a(audio)
#         output = [
#             torch.nn.functional.interpolate(tens, size=tgt_len, mode='linear', align_corners=True).transpose(1, 2) for tens in fpn_out
#         ]
#         return tuple(output)
    
    
#     def deep_out_forward(self, deep_out_input):

#         bckbone_out = self.deep_output(deep_out_input)

#         leaf_output = self.leaf(bckbone_out)
#         leaf_output = torch.stack(leaf_output.split(int(leaf_output.shape[-1] / self.config.pyramid_layers_g), dim=-1), dim=0)

#         mask = (self.mask(bckbone_out).repeat_interleave(self.config.data_dim, dim=1))
#         mask = torch.stack(mask.split(int(mask.shape[-1] / self.config.pyramid_layers_g), dim=-1), dim=0)

#         temp_mod_out_dim = int(deep_out_input.shape[1] / self.config.pyramid_layers_g) - self.config.audio_dim
#         bckbone_out_bias = self.deep_output_bias(deep_out_input[:, :temp_mod_out_dim])
        
#         leaf_output_bias = self.leaf_bias(bckbone_out_bias)
#         leaf_output = torch.cat([leaf_output_bias[None], leaf_output], dim=0)

#         mask_bias = (self.mask_bias(bckbone_out_bias).repeat_interleave(self.config.data_dim, dim=1))
#         mask = torch.cat([mask_bias[None], mask], dim=0)


#         if self.config.streams_merging_activation == 'softmax':
#             mask = torch.softmax(mask, dim=0)
#         else:
#             mask = torch.sigmoid(mask)
#         out = mask * leaf_output

#         return self.final_block(out.transpose(0, 1).flatten(start_dim=1))


#     def reset_mask(self):
#         setattr(self, 'soft_mask', [])
        

#     def forward(self, inpt, audio):
#         '''
#         Params:
#         ------
#         'inpt' shape: bs, obs_len, 10 (keypoints dim), data_dim
#         'audio' spectrogram of shape bs, 4 * (obs_len + seq_len), audio_dim

#         Outputs:
#         -------
#         Reconstructed sequence, shape: bs, obs_len - 1, 10, data_dim
#         Predicted sequence, shape: bs, seq_len, 10, data_dim
#         '''

#         bs, obs_len, _, _ = inpt.size()

#         # Flattening of coordinate inputs
#         running_inpt = inpt.flatten(start_dim=-2)
        
#         # Encode audio -> bs, obs_len + seq_len, config.audio_dim
#         seq_len = int(audio.shape[1] / 4) - obs_len
#         audio_pyramid = self.embed_audio(audio)
#         audio = audio_pyramid[0]

#         #####
#         ### Encoding of observed sequence / warm-up of the autoregressive model
#         #####

#         # Embedding 
#         coord = self.encoder_x(running_inpt)
#         # Temp module
#         forward_mask = torch.triu(torch.ones(obs_len, obs_len), diagonal=1).bool().to(device)
#         deep_out_input = self.lin_out(self.temp_module(coord.transpose(0, 1), mask=forward_mask).transpose(0, 1))
#         # Mutli-scale module
#         deep_out_input = torch.cat([deep_out_input, coord], dim=-1)
#         deep_out_input = torch.cat([torch.cat([deep_out_input, a[:, 1:obs_len + 1]], dim=-1) for a in audio_pyramid], dim=-1)
#         last_vel = self.deep_out_forward(deep_out_input.flatten(end_dim=1)).view(bs, obs_len, self.input_dim, self.config.data_dim)
#         # Residual addition of instantaneous velocities
#         last_pos = inpt + last_vel
        
#         if last_pos.size(1) > 1:
#             obs_outputs = last_pos[:, :-1]
#         else:
#             obs_outputs = torch.empty(0, 0, 0, 0).cuda()
#         last_pos = last_pos[:, -1]

#         #####
#         ### Decoding of hidden sequence
#         #####
#         len_to_decode = seq_len - 1
#         running_inpt = last_pos.view(bs, -1)
#         outputs = [last_pos.unsqueeze(1)]

#         for i in range(len_to_decode):
            
#             coord_i = self.encoder_x(running_inpt)

#             coord = torch.cat([coord, coord_i.unsqueeze(1)], dim=1)[:, -MAX_CONTEXT_LENGTH:]
#             forward_mask = torch.triu(torch.ones(coord.shape[1], coord.shape[1]), diagonal=1).bool().cuda()
#             h_n = self.lin_out(self.temp_module(coord.transpose(0, 1), mask=forward_mask).transpose(0, 1))
#             h_n = h_n[:, -1]

#             deep_out_input = torch.cat([h_n, coord_i], dim=-1)
#             deep_out_input = torch.cat([torch.cat([deep_out_input, a[:, i + obs_len + 1]], dim=-1) for a in audio_pyramid], dim=-1)

#             # Velocity output
#             last_vel = self.deep_out_forward(deep_out_input).view(bs, self.input_dim, self.config.data_dim)
#             last_pos = last_pos + last_vel

#             running_inpt = last_pos.view(bs, -1)

#             outputs.append(last_pos.unsqueeze(1))

#         outputs = torch.cat(outputs, dim=1)

#         return obs_outputs, outputs
