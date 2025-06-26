
import torch
import torch.nn as nn

from typing import Optional, Union
from torch_scatter import scatter_mean
from torch_geometric.nn import radius_graph, radius

from protenix.openfold_local.model.primitives import LayerNorm
from protenix.model.modules.primitives import LinearNoBias

from .utlis import * 

class Protein_Atom_Layer(nn.Module):

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

class Ligand_Atom_Layer(nn.Module):

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

class Atom_Encoder(nn.Module):
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
        super(Atom_Encoder, self).__init__()
        self.c_s = c_s
        self.c_s_inputs = c_s_inputs
        self.n_blocks = n_blocks
        
        self.protein_token_emb = LinearNoBias(c_s + c_s_inputs, c_z)
        self.protein_token_norm = LayerNorm(c_s + c_s_inputs)
        self.ligand_token_emb = LinearNoBias(c_s + c_s_inputs + 2048, c_z)
        self.ligand_token_norm = LayerNorm(c_s + c_s_inputs + 2048)

        self.layernorm_n = LayerNorm(c_noise_embedding)
        self.linear_no_bias_n = LinearNoBias(in_features=c_noise_embedding, out_features=c_z)

        self.input_feature = {
            "ref_pos": 3,
            "ref_charge": 1,
            "ref_mask": 1,
            "ref_element": 128,
            "ref_atom_name_chars": 4 * 64,
        }
        self.linear_no_bias_f = LinearNoBias(
            in_features=sum(self.input_feature.values()), out_features=c_z
        )
        
        for i in range(n_blocks):
            self.add_module("block_intra_protein_%d" % i, Protein_Atom_Layer(c_z=c_z, num_heads=16, dropout=0.1))
            
            self.add_module("block__inter_l2p_%d" % i, L2P_Inter_Layer(c_z=c_z, num_heads=16, dropout=0.1))

            self.add_module("block_intra_ligand_%d" % i, Ligand_Atom_Layer(c_z=c_z, num_heads=16, dropout=0.1))

            self.add_module("block__inter_p2l_%d" % i, P2L_Inter_Layer(c_z=c_z, num_heads=16, dropout=0.1))


    def forward(
        self,
        r_noisy: torch.Tensor,
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
        token_to_atom_emb: torch.Tensor,
        smile_emb: torch.Tensor,
        noise_n: torch.Tensor,
        is_protein: torch.Tensor,
        is_ligand: torch.Tensor,
        atom_batch: torch.Tensor,

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
        noise_emb = self.linear_no_bias_n(self.layernorm_n(noise_n))

        device = r_noisy.device
        b, n, _ = r_noisy.shape # b, n, 3
        r_noisy = r_noisy.view(b * n , 3)

        is_protein = is_protein.bool()
        is_ligand = is_ligand.bool()

        batch_shape = input_feature_dict["ref_pos"].shape[:-2]
        N_atom = input_feature_dict["ref_pos"].shape[-2]
        c_l = self.linear_no_bias_f(
            torch.cat(
                [
                    input_feature_dict[name].reshape(
                        *batch_shape, N_atom, self.input_feature[name]
                    )
                    for name in self.input_feature
                ],
                dim=-1,
            )
        )  # c_L torch.Size([2815, 128])

        c_l_expanded = c_l.unsqueeze(0).repeat(b, 1, 1).view(b * n, -1)

        protein_atom_emb = c_l_expanded[is_protein] + self.protein_token_emb(self.protein_token_norm(token_to_atom_emb[is_protein]))
        smile_emb = smile_emb.expand(c_l_expanded[is_ligand].shape[0], -1)
        ligand_atom_emb = c_l_expanded[is_ligand] + self.ligand_token_emb(self.ligand_token_norm(torch.cat([token_to_atom_emb[is_ligand], smile_emb], dim=-1)))
        protein_pos = r_noisy[is_protein]
        ligand_pos = r_noisy[is_ligand]

        protein_batch = atom_batch[is_protein]
        protein_edge_index = radius_graph(x=protein_pos, batch=protein_batch, r=6, max_num_neighbors=30, loop=False)

        ligand_batch = atom_batch[is_ligand]
        ligand_edge_index = radius_graph(x=ligand_pos, batch=ligand_batch, r=6, max_num_neighbors=6, loop=False)

        ligand2protein_edge_index = self.build_edge_index(protein_pos, ligand_pos,
                            protein_batch, ligand_batch, 6, max_num_neighbors=10)

        protein2ligand_edge_index = self.build_edge_index(protein_pos, ligand_pos,
                            protein_batch, ligand_batch, 6, max_num_neighbors=30, protein2ligand=True)

        protein_atom_emb = protein_atom_emb + noise_emb[protein_batch]
        ligand_atom_emb = ligand_atom_emb +  noise_emb[ligand_batch]

        for i in range(self.n_blocks):
            protein_atom_update = self._modules['block_intra_protein_%d' % i](protein_pos, protein_atom_emb, protein_edge_index)
            protein_atom_emb = protein_atom_emb + protein_atom_update
            l2p_atom_update = self._modules['block__inter_l2p_%d' % i](protein_pos, protein_atom_emb, ligand2protein_edge_index, ligand_pos, ligand_atom_emb)
            protein_atom_emb = protein_atom_emb + l2p_atom_update

            ligand_token_update = self._modules['block_intra_ligand_%d' % i](ligand_pos, ligand_atom_emb, ligand_edge_index)
            ligand_atom_emb = ligand_atom_emb + ligand_token_update

            p2l_atom_update = self._modules['block__inter_p2l_%d' % i](ligand_pos, ligand_atom_emb, protein2ligand_edge_index, protein_pos, protein_atom_emb)
            ligand_atom_emb = ligand_atom_emb + p2l_atom_update

        return protein_atom_emb, ligand_atom_emb

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
    


