
import torch
import torch.nn as nn

from typing import Optional, Union
from torch_scatter import scatter_mean
from torch_geometric.nn import radius_graph, radius

from protenix.model.modules.embedders import FourierEmbedding
from protenix.openfold_local.model.primitives import LayerNorm
from protenix.model.modules.primitives import LinearNoBias
from protenix.model.utils import aggregate_atom_to_token
from torch_scatter import scatter_max

from .atten_layer import *
from .utlis import *


class Protein_Token_Layer(nn.Module):

    def __init__(self, c_z: int = 128, 
                    num_heads: int = 12, 
                    act=nn.SiLU(), 
                    dropout=0.0):
        super().__init__()
        self.act = act
        self.edge_emb = nn.Linear(c_z//4, c_z//4)
        self.atom_mpnn = Attention_Layer(c_z, c_z//4, c_z // num_heads, num_heads, dropout)
        self.dist_layer = GaussianSmearing(stop=15, num_gaussians=c_z//4)

    def forward(self, pos, h, edge_index):
        row, col = edge_index
        distance = (pos[row] - pos[col]).pow(2).sum(dim=-1).sqrt().unsqueeze(-1)
        distance = self.dist_layer(distance)
        edge_attr = self.edge_emb(distance)

        h_node = self.atom_mpnn(h, edge_index, edge_attr)

        return h_node

class Ligand_Token_Layer(nn.Module):

    def __init__(self, c_z: int = 128, 
                    num_heads: int = 12, 
                    act=nn.SiLU(), 
                    dropout=0.0):
        super().__init__()
        self.act = act
        self.edge_emb = nn.Linear(c_z//4, c_z//4)
        self.atom_mpnn = Attention_Layer(c_z, c_z//4, c_z // num_heads, num_heads, dropout)
        self.dist_layer = GaussianSmearing(stop=5, num_gaussians=c_z//4)

    def forward(self, pos, h, edge_index):
        row, col = edge_index
        distance = (pos[row] - pos[col]).pow(2).sum(dim=-1).sqrt().unsqueeze(-1)
        distance = self.dist_layer(distance)
        edge_attr = self.edge_emb(distance)
        h_node = self.atom_mpnn(h, edge_index, edge_attr)

        return h_node

class Token_Encoder(nn.Module):
    def __init__(
        self,
        c_s: int = 384,
        c_z: int = 128,
        c_s_inputs: int = 449,
        c_noise_embedding: int = 256,
        n_blocks: int = 6,
        
    ) -> None:
        
        """
        Args:
            c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
            c_s_inputs (int, optional): input embedding dim from InputEmbedder. Defaults to 449.
            c_noise_embedding (int, optional): noise embedding dim. Defaults to 256.
        """
        super(Token_Encoder, self).__init__()
        self.c_s = c_s
        self.c_s_inputs = c_s_inputs
        self.n_blocks = n_blocks
        
        self.protein_token_emb = nn.Linear(c_s + c_s_inputs, c_z)
        self.protein_token_norm = LayerNorm(c_s + c_s_inputs)
        self.ligand_token_emb = nn.Linear(c_s + c_s_inputs + 2048, c_z)
        self.ligand_token_norm = LayerNorm(c_s + c_s_inputs + 2048)

        # self.fourier_embedding = FourierEmbedding(c=c_noise_embedding)
        self.layernorm_n = LayerNorm(c_noise_embedding)
        self.linear_no_bias_n = LinearNoBias(in_features=c_noise_embedding, out_features=c_z)
        
        for i in range(n_blocks):
            self.add_module("block_intra_protein_%d" % i, Protein_Token_Layer(c_z=c_z, num_heads=16, dropout=0.1))
            
            self.add_module("block__inter_l2p_%d" % i, L2P_Inter_Layer(c_z=c_z, num_heads=16, dropout=0.1))

            self.add_module("block_intra_ligand_%d" % i, Ligand_Token_Layer(c_z=c_z, num_heads=16, dropout=0.1))

            self.add_module("block__inter_p2l_%d" % i, P2L_Inter_Layer(c_z=c_z, num_heads=16, dropout=0.1))

    def forward(
        self,
        r_noisy: torch.Tensor,
        token_emb: torch.Tensor,
        smile_emb: torch.Tensor,
        atom_to_token_idx: torch.Tensor,
        noise_n: torch.Tensor,
        is_protein: torch.Tensor,
        is_ligand: torch.Tensor,
        token_batch: torch.Tensor,
        protein_atom_to_token_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
                [..., N_tokens, c_s_inputs]
            s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
                [..., N_tokens, c_s]
        Returns:
            tuple[torch.Tensor, torch.Tensor]: embeddings s and z
                - s (torch.Tensor): [..., N_sample, N_tokens, c_s]
                - z (torch.Tensor): [..., N_tokens, N_tokens, c_z]
        """
        
        b, n, _ = r_noisy.shape # b, n, 3
        is_protein = is_protein.bool()
        is_ligand = is_ligand.bool()
        
        r_noisy = r_noisy.view(b * n, 3)

        num_tokens = atom_to_token_idx.max().item() + 1
        is_token, _ = scatter_max(is_protein.long(), atom_to_token_idx, dim=0, dim_size=num_tokens)

        protein_token_mask = (is_token == 1)
        protein_token = token_emb[protein_token_mask.bool()]
        protein_pos = r_noisy[is_protein]
        protein_token_pos = scatter_mean(protein_pos, protein_atom_to_token_idx, dim=0)
        protein_batch = token_batch[protein_token_mask.bool()]
        protein_edge_index = radius_graph(x=protein_token_pos, batch=protein_batch, r=15, max_num_neighbors=30, loop=False)

        ligand_token_mask = (is_token != 1)
        ligand_token = token_emb[ligand_token_mask.bool()]
        ligand_pos = r_noisy[is_ligand]
        ligand_batch = token_batch[ligand_token_mask.bool()]
        ligand_edge_index = radius_graph(x=ligand_pos, batch=ligand_batch, r=6, max_num_neighbors=10, loop=False)

        ligand2protein_edge_index = self.build_edge_index(protein_token_pos, ligand_pos,
                            protein_batch, ligand_batch, 12, max_num_neighbors=10)

        protein2ligand_edge_index = self.build_edge_index(protein_token_pos, ligand_pos,
                            protein_batch, ligand_batch, 12, max_num_neighbors=30, protein2ligand=True)
        
        smile_emb = smile_emb.expand(ligand_token.shape[0], -1)
        ligand_token = torch.cat([ligand_token, smile_emb],dim=-1)

        noise_emb = self.linear_no_bias_n(self.layernorm_n(noise_n))
        protein_token = self.protein_token_emb(self.protein_token_norm(protein_token)) + noise_emb[protein_batch]
        ligand_token = self.ligand_token_emb(self.ligand_token_norm(ligand_token)) +  noise_emb[ligand_batch]

        for i in range(self.n_blocks):

            protein_token_update = self._modules['block_intra_protein_%d' % i](protein_pos, protein_token, protein_edge_index)
            protein_token = protein_token + protein_token_update
            l2p_token_update = self._modules['block__inter_l2p_%d' % i](protein_pos, protein_token, ligand2protein_edge_index, ligand_pos, ligand_token)
            protein_token = protein_token + l2p_token_update

            ligand_token_update = self._modules['block_intra_ligand_%d' % i](ligand_pos, ligand_token, ligand_edge_index)
            ligand_token = ligand_token + ligand_token_update

            p2l_token_update = self._modules['block__inter_p2l_%d' % i](ligand_pos, ligand_token, protein2ligand_edge_index, protein_pos, protein_token)
            ligand_token = ligand_token + p2l_token_update


        return protein_token, ligand_token

    def build_edge_index(self, protein_pos, ligand_pos, protein_batch, ligand_batch, r,
                         max_num_neighbors=30, protein2ligand=False):
        
        if protein2ligand is True:
            edge_index = radius(protein_pos, ligand_pos, r,
                            protein_batch, ligand_batch, max_num_neighbors=max_num_neighbors)
            # edge index[0] 是 protein
        
        else:
            edge_index = radius(ligand_pos, protein_pos, r,
                            ligand_batch, protein_batch, max_num_neighbors=max_num_neighbors)
            # edge index[0] 是 ligand
            
            
        
        return edge_index
    


