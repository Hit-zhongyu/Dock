import torch
from torch import nn
from .atten_layer import * 


class GaussianSmearing(torch.nn.Module):
    # used to embed the edge distances
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

    
class P2L_Inter_Layer(nn.Module):
    """Equivariant block based on graph relational transformer layer, without extra heads."""

    def __init__(self, c_z: int = 128, 
                    num_heads: int = 12,
                    act=nn.SiLU(), 
                    dropout=0.0):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.act = act


        self.edge_emb = nn.Linear(c_z//4, c_z//4)

        # # message passing layer
        self.attn_mpnn_inter = Inter_attention(c_z, c_z // num_heads, num_heads, c_z//4, dropout=dropout)

        self.dist_layer = GaussianSmearing(stop=30, num_gaussians=c_z//4)

        self.ln_out = nn.Sequential(
                    nn.Linear(c_z, c_z * 2),
                    nn.SiLU(),
                    nn.Linear(c_z * 2, c_z)
                )


    def forward(self, pos, h, edge_index, pos_p, p):

        row, col = edge_index
        coord_diff = pos[row] - pos_p[col]
        distance = torch.sum(coord_diff ** 2, 1).unsqueeze(1)
        distance = self.dist_layer(distance)
        edge_attr = self.edge_emb(distance)

        h_node = self.attn_mpnn_inter(h, p, edge_index, edge_attr)

        return self.ln_out(h_node)


class L2P_Inter_Layer(nn.Module):
    """Equivariant block based on graph relational transformer layer, without extra heads."""

    def __init__(self, c_z: int = 128, 
                    num_heads: int = 12,
                    act=nn.SiLU(), 
                    dropout=0.0):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.act = act

        self.edge_emb = nn.Linear(c_z//4, c_z//4)

        # # message passing layer
        self.attn_mpnn_inter = Inter_attention(c_z, c_z // num_heads, num_heads, c_z//4, dropout=dropout)

        self.dist_layer = GaussianSmearing(stop=25, num_gaussians=c_z//4)

        self.ln_out = nn.Sequential(
                    nn.Linear(c_z, c_z * 2),
                    nn.SiLU(),
                    nn.Linear(c_z * 2, c_z)
                )


    def forward(self, pos, h, edge_index, pos_p, p):

        row, col = edge_index
        coord_diff = pos[row] - pos_p[col]
        distance = torch.sum(coord_diff ** 2, 1).unsqueeze(1)
        distance = self.dist_layer(distance)
        edge_attr = self.edge_emb(distance)

        h_node = self.attn_mpnn_inter(h, p, edge_index, edge_attr)

        return self.ln_out(h_node)


def expand_at_dim(x: torch.Tensor, dim: int, n: int) -> torch.Tensor:
    """expand a tensor at specific dim by n times

    Args:
        x (torch.Tensor): input
        dim (int): dimension to expand
        n (int): expand size

    Returns:
        torch.Tensor: expanded tensor of shape [..., n, ...]
    """
    x = x.unsqueeze(dim=dim)
    if dim < 0:
        dim = x.dim() + dim
    before_shape = x.shape[:dim]
    after_shape = x.shape[dim + 1 :]
    return x.expand(*before_shape, n, *after_shape)