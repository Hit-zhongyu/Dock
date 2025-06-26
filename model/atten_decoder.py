import torch


from torch import nn
from protenix.openfold_local.model.primitives import LayerNorm
from protenix.model.modules.primitives import AdaptiveLayerNorm
from protenix.model.modules.primitives import LinearNoBias

from .atten_layer import Local_Attention, Global_Attention

class Atten_Decoder(nn.Module):
    """
    Implements Algorithm 24 in AF3
    """

    def __init__(
        self,
        n_heads: int = 16,
        c_z: int = 128,
    ) -> None:
        """
        Args:
            has_s (bool, optional):  whether s is None as stated in Algorithm 24 Line1. Defaults to True.
            n_heads (int, optional): number of attention-like head in AttentionPairBias. Defaults to 16.
            c_a (int, optional): the embedding dim of a(single feature aggregated atom info). Defaults to 768.
            c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
            c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
            biasinit (float, optional): biasinit for BiasInitLinear. Defaults to -2.0.
        """
        super(Atten_Decoder, self).__init__()
        self.n_heads = n_heads

        self.layernorm_protein_atom = AdaptiveLayerNorm(c_a=c_z, c_s=c_z)
        self.layernorm_ligand_atom = AdaptiveLayerNorm(c_a=c_z, c_s=c_z)
        
        # Line 6-11
        # self.local_attention_method = "local_cross_attention"
        self.local_attention = Local_Attention(
            c_q=c_z,
            c_k=c_z,
            c_v=c_z,
            c_hidden=c_z // n_heads,
            num_heads=n_heads,
        )

        self.global_attention = Global_Attention(
            c_q=c_z,
            c_k=c_z,
            c_v=c_z,
            c_hidden=c_z // n_heads,
            num_heads=n_heads,
        )

        self.protein_layernorm = LayerNorm(c_z)
        # Alg24. Line8 is scalar, but this is different for different heads
        self.protein_linear_nobias = LinearNoBias(in_features=c_z, out_features=c_z)

        self.ligand_layernorm = LayerNorm(c_z)
        # Alg24. Line8 is scalar, but this is different for different heads
        self.ligand_linear_nobias = LinearNoBias(in_features=c_z, out_features=c_z)

    def glorot_init(self):
        nn.init.xavier_uniform_(self.local_attention.linear_q.weight)
        nn.init.xavier_uniform_(self.local_attention.linear_k.weight)
        nn.init.xavier_uniform_(self.local_attention.linear_v.weight)
        nn.init.zeros_(self.local_attention.linear_q.bias)

        nn.init.xavier_uniform_(self.global_attention.linear_q.weight)
        nn.init.xavier_uniform_(self.global_attention.linear_k.weight)
        nn.init.xavier_uniform_(self.global_attention.linear_v.weight)
        nn.init.zeros_(self.global_attention.linear_q.bias)


    def forward(
        self,
        protein_atom: torch.Tensor,
        protein_token: torch.Tensor,
        ligand_atom: torch.Tensor,
        ligand_token: torch.Tensor,
        atom_to_token_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Details are given in local_forward and standard_forward"""

        protein_atom = self.layernorm_protein_atom(protein_atom, protein_token)
        ligand_atom = self.layernorm_ligand_atom(ligand_atom, ligand_token)
        
        # Multihead attention with pair bias
        protein_atom = self.local_attention(
            protein_atom,
            protein_token,
            atom_to_token_idx,
        )

        ligand_atom = self.global_attention(
            ligand_atom,
            ligand_token,
        )

        return protein_atom, ligand_atom