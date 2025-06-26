# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from protenix.utils.logger import get_logger
from protenix.model.modules.embedders import FourierEmbedding
from protenix.openfold_local.model.primitives import LayerNorm
from protenix.model.modules.primitives import LinearNoBias
from protenix.model.generator import TrainingNoiseSampler

from protenix.model.utils import broadcast_token_to_atom, aggregate_atom_to_token
from protenix.model.utils import centre_random_augmentation
from typing import Union

from .atom_encoder import Atom_Encoder
from .token_encoder import Token_Encoder
from .atten_decoder import Atten_Decoder
from .utlis import expand_at_dim

logger = get_logger(__name__)


class Docking(nn.Module):

    def __init__(
        self, 
        configs,
        sigma_data: float = 16.0,
        c_s: int = 384,
        c_z: int = 128,
        c_s_inputs: int = 449,
        atom_blocks: int = 6,
        token_blocks: int = 6,
        atten_blocks: int = 12,
        n_heads: int = 16,
        c_noise_embedding: int = 256,
        ) -> None:
        super(Docking, self).__init__()
        self.sigma_data = sigma_data
        self.diffusion_batch_size = configs.diffusion_batch_size

        self.train_noise_sampler = TrainingNoiseSampler(**configs.train_noise_sampler)
        self.fourier_embedding = FourierEmbedding(c=c_noise_embedding)

        self.atom_encoder = Atom_Encoder(
                                c_s=c_s,
                                c_z=c_z,
                                c_s_inputs=c_s_inputs,
                                n_blocks=atom_blocks)
        self.token_encoder = Token_Encoder(
                                c_s=c_s,
                                c_z=c_z,
                                c_s_inputs=c_s_inputs,
                                n_blocks=token_blocks)
        
        self.atten_blocks = nn.ModuleList()
        for _ in range(atten_blocks):
            block = Atten_Decoder(
                    c_z=c_z, n_heads=n_heads,
                )
            self.atten_blocks.append(block)

        self.protein_layernorm = LayerNorm(c_z)
        self.protein_linear_no_bias_out = LinearNoBias(in_features=c_z, out_features=3)

        self.ligand_layernorm = LayerNorm(c_z)
        self.ligand_linear_no_bias_out = LinearNoBias(in_features=c_z, out_features=3)

    def forward(
        self,
        label_dict: dict[str, Any],
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
        smile_emb: str,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """One step denoise: x_noisy, noise_level -> x_denoised

        Args:
            x_noisy (torch.Tensor): the noisy version of the input atom coords
                [..., N_sample, N_atom,3]
            t_hat_noise_level (torch.Tensor): the noise level, as well as the time step t
                [..., N_sample]
            input_feature_dict (dict[str, Union[torch.Tensor, int, float, dict]]): input meta feature dict
            s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
                [..., N_tokens, c_s_inputs]
            s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
                [..., N_tokens, c_s]
            z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
                [..., N_tokens, N_tokens, c_z]
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            torch.Tensor: the denoised coordinates of x
                [..., N_sample, N_atom,3]
        """
        # Scale positions to dimensionless vectors with approximately unit variance
        # As in EDM:
        #     r_noisy = (c_in * x_noisy)
        #     where c_in = 1 / sqrt(sigma_data^2 + sigma^2)
        pred_dict = {}

        batch_size_shape = label_dict["coordinate"].shape[:-2]
        device = label_dict["coordinate"].device
        dtype = label_dict["coordinate"].dtype
        N_sample = self.diffusion_batch_size

        x_gt_augment = centre_random_augmentation(
            x_input_coords=label_dict["coordinate"],
            N_sample=N_sample,
            mask=label_dict["coordinate_mask"],
        ).to(
            dtype
        ) 

        sigma = self.train_noise_sampler(size=(*batch_size_shape, N_sample), device=device).to(dtype)
        noise = torch.randn_like(x_gt_augment, dtype=dtype) * sigma[..., None, None]
        x_noisy = x_gt_augment + noise
        t_hat_noise_level = sigma

        noise_n = self.fourier_embedding(t_hat_noise_level=torch.log(input=t_hat_noise_level / self.sigma_data) / 4).to(s_inputs.dtype)

        torch.set_printoptions(profile="full")
        r_noisy = (
            x_noisy
            / torch.sqrt(self.sigma_data**2 + t_hat_noise_level**2)[..., None, None]
        )

        token_len, _ = s_inputs.shape
        b, n, _ = r_noisy.shape
        device = r_noisy.device

        is_protein = input_feature_dict['is_protein'].unsqueeze(0).repeat(b, 1).view(-1) # 哪些原子属于protein
        is_ligand = input_feature_dict['is_ligand'].unsqueeze(0).repeat(b, 1).view(-1) # 那些原子属于ligand

        token_input = s_inputs.unsqueeze(0).repeat(b, 1, 1).view(b*token_len, -1)
        token_truck = s_trunk.unsqueeze(0).repeat(b, 1, 1).view(b*token_len, -1)
        token_emb = torch.cat([token_input, token_truck], dim=-1)

        atom_to_token_idx = input_feature_dict['atom_to_token_idx']
        atom_to_token_idx_convert = self.batch_shifted_atom_to_token(atom_to_token_idx, b)
        token_batch = torch.arange(b, device=device).repeat_interleave(token_len)

        n_ligand = input_feature_dict['is_ligand'].sum() 
        
        protein_atom_to_token_idx = atom_to_token_idx_convert[is_protein.bool()]
        _, protein_atom_to_token_idx = torch.unique(protein_atom_to_token_idx, return_inverse=True)
        
        protein_token, ligand_token = self.token_encoder(r_noisy=r_noisy, token_emb=token_emb, smile_emb=smile_emb, 
                        atom_to_token_idx=atom_to_token_idx_convert, noise_n=noise_n, is_protein=is_protein, 
                        is_ligand=is_ligand, token_batch=token_batch, protein_atom_to_token_idx=protein_atom_to_token_idx)


        token_to_atom_emb = broadcast_token_to_atom(
                        x_token=token_emb, atom_to_token_idx=atom_to_token_idx_convert
                    )
        atom_batch = torch.arange(b, device=device).repeat_interleave(n)
        protein_atom, ligand_atom = self.atom_encoder(r_noisy=r_noisy, input_feature_dict=input_feature_dict, token_to_atom_emb=token_to_atom_emb,
                         smile_emb=smile_emb, noise_n=noise_n, is_protein=is_protein, is_ligand=is_ligand, atom_batch=atom_batch)

        protein_token_to_atom = broadcast_token_to_atom(x_token=protein_token, atom_to_token_idx=protein_atom_to_token_idx)
        # try:
        #     ligand_token = ligand_token.view(b, n_ligand, -1)
        #     ligand_atom = ligand_atom.view(b, n_ligand, -1)
        # except:
        #     print(input_feature_dict['is_ligand'])
        #     print(n_ligand)

        for block in self.atten_blocks:
            protein_atom, ligand_atom = block(protein_atom=protein_atom, protein_token=protein_token_to_atom, ligand_atom=ligand_atom, 
                                              ligand_token=ligand_token, atom_to_token_idx=protein_atom_to_token_idx)

        protein_update = self.protein_linear_no_bias_out(self.protein_layernorm(protein_atom))
        ligand_update = self.ligand_linear_no_bias_out(self.ligand_layernorm(ligand_atom))

        r_update = torch.zeros_like(r_noisy, dtype=protein_update.dtype)
        r_update = r_update.view(-1, r_noisy.shape[-1])
        r_update[is_protein.bool()] = protein_update
        r_update[is_ligand.bool()] = ligand_update

        r_update = r_update.view(b, n, -1)
        s_ratio = (t_hat_noise_level / self.sigma_data)[..., None, None].to(
            r_update.dtype
        )

        x_denoised = (
            1 / (1 + s_ratio**2) * x_noisy
            + t_hat_noise_level[..., None, None] / torch.sqrt(1 + s_ratio**2) * r_update
        ).to(r_update.dtype)

        pred_dict.update(
            {
                "coordinate": x_denoised,
                "noise_level": sigma,
            }
        )
        return pred_dict
    
    def batch_shifted_atom_to_token(self, atom_to_token_idx, batch):
        per_batch = atom_to_token_idx
        n_token = int(per_batch.max().item()) + 1
        shifted = [per_batch + i * n_token for i in range(batch)]
        return torch.cat(shifted, dim=0)
