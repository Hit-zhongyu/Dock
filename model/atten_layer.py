import torch
import math
import numpy as np
import torch.nn.functional as F

from torch import nn
from torch_scatter import scatter_softmax, scatter_sum
from torch_geometric.nn.conv import MessagePassing
from protenix.openfold_local.model.primitives import LayerNorm
from protenix.model.modules.primitives import LinearNoBias
from typing import Optional


class Attention_Layer(nn.Module):
    def __init__(self, c_z, c_edge, c_out, heads, dropout):
        super(Attention_Layer, self).__init__()

        self.c_z = c_z
        self.c_out = c_out
        self.heads = heads
        self.dropout = dropout

        self.lin_norm = LayerNorm(c_z * 2 + c_edge)

        self.lin_query = nn.Linear(c_z, heads * c_out)
        self.lin_key = nn.Linear(c_z * 2 + c_edge, heads * c_out)
        self.lin_value = nn.Linear(c_z * 2 + c_edge, heads * c_out)
        
        self.ln_out = nn.Sequential(
                        nn.Linear(c_z, c_z * 2),
                        nn.SiLU(),
                        nn.Linear(c_z * 2, c_z)
                    )

    def forward(self, x, edge_index, edge_attr):

        H, C = self.heads, self.c_out

        hi, hj = x[edge_index[0]], x[edge_index[1]]
        x_feat = self.lin_norm(torch.cat([edge_attr, hi, hj], dim=-1))
        query = self.lin_query(x).view(-1, H, C)
        key = self.lin_key(x_feat).view(-1, H, C)
        value = self.lin_value(x_feat).view(-1, H, C)

        alpha = scatter_softmax((query[edge_index[1]] * key / np.sqrt(key.shape[-1])).sum(-1), edge_index[1], dim=0, dim_size=x.size(0))
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        m = alpha.unsqueeze(-1) * value
        out_x = scatter_sum(m, edge_index[1], dim=0, dim_size=x.size(0))
        out_x = out_x.view(-1, self.heads * self.c_out)

        out_x = self.ln_out(out_x)

        return out_x
    

class Inter_attention(MessagePassing):
    def __init__(self, c_z: int = 128, 
                    c_out: int = 384,
                 heads: int = 1, edge_dim: int = 64, dropout: float = 0.
                 ):
        super(Inter_attention, self).__init__()

        self.heads = heads
        self.c_out = c_out
        self.dropout = dropout
        self.edge_dim = edge_dim

        self.lin_key = nn.Linear(c_z, heads * c_out)
        self.lin_query = nn.Linear(c_z, heads * c_out)
        self.lin_value = nn.Linear(c_z, heads * c_out)

        self.lin_edge0 = nn.Linear(edge_dim, heads * c_out, bias=False)
        self.lin_edge1 = nn.Linear(edge_dim, heads * c_out, bias=False)

        # self.reset_parameters()

    # def reset_parameters(self):
    #     self.lin_key.reset_parameters()
    #     self.lin_query.reset_parameters()
    #     self.lin_value.reset_parameters()
    #     self.lin_edge0.reset_parameters()
    #     self.lin_edge1.reset_parameters()

    def forward(self, x, p, edge_index, edge_attr):

        H, C = self.heads, self.c_out
        x_feat = x
        p_feat = p
        query = self.lin_query(x_feat).view(-1, H, C)
        key = self.lin_key(p_feat).view(-1, H, C)
        value = self.lin_value(p_feat).view(-1, H, C)

        out_x = self.propagate(edge_index, query=query, key=key, value=value, edge_attr=edge_attr, size=(x.size(0), p.size(0)))
        out_x = out_x.view(-1, self.heads * self.c_out)
        return out_x

    def message(self, query_i, key_j, value_j, edge_attr, index, ptr, size_i):

        edge_attn = self.lin_edge0(edge_attr).view(-1, self.heads, self.c_out)
        edge_attn = torch.tanh(edge_attn)
        alpha = (query_i * key_j * edge_attn).sum(dim=-1) / math.sqrt(self.c_out)

        alpha = nn.softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # node feature message
        msg = value_j
        msg = msg * torch.tanh(self.lin_edge1(edge_attr).view(-1, self.heads, self.c_out))
        msg = msg * alpha.view(-1, self.heads, 1)

        return msg


class Inter_attention(nn.Module):
    def __init__(self, c_z, c_out, heads, c_edge, dropout):
        super(Inter_attention, self).__init__()

        self.c_z = c_z
        self.c_out = c_out
        self.heads = heads
        self.dropout = dropout

        self.lin_norm = LayerNorm(c_z * 2 + c_edge)

        self.lin_query = nn.Linear(c_z, heads * c_out)
        self.lin_key = nn.Linear(c_z * 2 + c_edge, heads * c_out)
        self.lin_value = nn.Linear(c_z * 2 + c_edge, heads * c_out)
        self.lin_edge = nn.Linear(c_edge, heads * c_out,  bias=False)
        
        self.ln_out = nn.Sequential(
                        nn.Linear(c_z, c_z * 2),
                        nn.SiLU(),
                        nn.Linear(c_z * 2, c_z)
                    )

    def forward(self, x, p, edge_index, edge_attr):

        H, C = self.heads, self.c_out

        hi, hj = x[edge_index[0]], p[edge_index[1]]
        x_feat = self.lin_norm(torch.cat([edge_attr, hi, hj], dim=-1))
        query = self.lin_query(x).view(-1, H, C)
        key = self.lin_key(x_feat).view(-1, H, C)
        value = self.lin_value(x_feat).view(-1, H, C)

        alpha = scatter_softmax((query[edge_index[0]] * key / np.sqrt(key.shape[-1])).sum(-1), edge_index[0], dim=0, dim_size=x.size(0))
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        value = value * self.lin_edge(edge_attr).view(-1, self.heads, self.c_out)
        m = alpha.unsqueeze(-1) * value 
        out_x = scatter_sum(m, edge_index[0], dim=0, dim_size=x.size(0))
        out_x = out_x.view(-1, self.heads * self.c_out)

        out_x = self.ln_out(out_x)

        return out_x


class Local_Attention(nn.Module):
    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_hidden: int,
        num_heads: int,
        attn_weight_dropout_p: float = 0.0,
    ) -> None:

        super(Local_Attention, self).__init__()
        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.num_heads = num_heads
        self.attn_weight_dropout_p = attn_weight_dropout_p

        # DISCREPANCY: c_hidden is not the per-head channel dimension, as
        # stated in the supplement, but the overall channel dimension.
        self.linear_q = nn.Linear(in_features=self.c_q, out_features=self.c_hidden * self.num_heads)
        self.linear_k = LinearNoBias(self.c_k, self.c_hidden * self.num_heads)
        self.linear_v = LinearNoBias(self.c_v, self.c_hidden * self.num_heads)
        self.linear_o = nn.Linear(self.c_hidden * self.num_heads, self.c_q)

        self.linear_g = LinearNoBias(self.c_q, self.c_hidden * self.num_heads)
        self.sigmoid = nn.Sigmoid()

        # Zero init the output layer
        # nn.init.zeros_(self.linear_o.weight)

    def _prep_qkv(
        self, q_x: torch.Tensor, kv_x: torch.Tensor, apply_scale: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare qkv

        Args:
            q_x (torch.Tensor): the input x for q
                [..., c_q]
            kv_x (torch.Tensor): the input x for kv
                [..., c_k]
                [..., c_v]
            apply_scale (bool, optional): apply scale to dot product qk. Defaults to True.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: the return q/k/v
                # [..., H, Q/K/V, C_hidden]
        """
        # [*, Q/K/V, H * C_hidden]
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        q = q.view(-1, self.num_heads, self.c_hidden)
        k = k.view(-1, self.num_heads, self.c_hidden)
        v = v.view(-1, self.num_heads, self.c_hidden)

        if apply_scale:
            q = q / math.sqrt(self.c_hidden)

        return q, k, v

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        atom_to_token_idx: Optional[int] = None,
    ) -> torch.Tensor:

        q, k, v = self._prep_qkv(q_x=q_x, kv_x=kv_x, apply_scale=True)

        alpha = scatter_softmax((q * k / np.sqrt(k.shape[-1])).sum(-1), atom_to_token_idx, dim=0, dim_size=q.size(0))
        m = alpha.unsqueeze(-1) * v
        out_x = scatter_sum(m, atom_to_token_idx, dim=0, dim_size=q.size(0))
        out_x = out_x.view(-1, self.num_heads* self.c_hidden)

        gate = self.sigmoid(self.linear_g(q_x))

        out_x = self.linear_o(out_x * gate)

        return out_x



class Global_Attention(nn.Module):
    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_hidden: int,
        num_heads: int,
        attn_weight_dropout_p: float = 0.0,
    ) -> None:

        super(Global_Attention, self).__init__()
        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.num_heads = num_heads
        self.attn_weight_dropout_p = attn_weight_dropout_p

        # DISCREPANCY: c_hidden is not the per-head channel dimension, as
        # stated in the supplement, but the overall channel dimension.
        self.linear_q = nn.Linear(in_features=self.c_q, out_features=self.c_hidden * self.num_heads)

        self.linear_k = LinearNoBias(self.c_k, self.c_hidden * self.num_heads)
        self.linear_v = LinearNoBias(self.c_v, self.c_hidden * self.num_heads)
        self.linear_o = nn.Linear(self.c_hidden * self.num_heads, self.c_q)

        self.linear_g = LinearNoBias(self.c_q, self.c_hidden * self.num_heads)
        self.sigmoid = nn.Sigmoid()

        # Zero init the output layer
        # nn.init.zeros_(self.linear_o.weight)

    def _prep_qkv(
        self, q_x: torch.Tensor, kv_x: torch.Tensor, apply_scale: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        q = q.view(q.shape[:-1] + (self.num_heads, -1))
        k = k.view(k.shape[:-1] + (self.num_heads, -1))
        v = v.view(v.shape[:-1] + (self.num_heads, -1))

        if apply_scale:
            q = q / math.sqrt(self.c_hidden)

        return q, k, v

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
    ) -> torch.Tensor:
        q, k, v = self._prep_qkv(q_x=q_x, kv_x=kv_x, apply_scale=True)
        out_x = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            dropout_p=self.attn_weight_dropout_p,
            )
        
        out_x = out_x.reshape(q_x.shape[0], self.num_heads* self.c_hidden)
        gate = self.sigmoid(self.linear_g(q_x))
        
        out_x = self.linear_o(out_x * gate)

        return out_x
    