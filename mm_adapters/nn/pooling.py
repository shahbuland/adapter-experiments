import torch
from torch import nn
import torch.nn.functional as F
import einops as eo

from torchtyping import TensorType

from .normalization import RMSNorm
from .mlp import MLP

class PerceiverAttn(nn.Module):
    def __init__(self, n_heads, hidden_size):
        super().__init__()

        self.q = nn.Linear(hidden_size, hidden_size)
        self.kv = nn.Linear(hidden_size, hidden_size * 2)
        self.out = nn.Linear(hidden_size, hidden_size)

        self.mlp = MLP(hidden_size)
    
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        self.postnorm = RMSNorm(hidden_size)

        self.n_heads = n_heads
        self.hidden_size = hidden_size

        assert hidden_size % n_heads == 0, "Number of heads must evenly divide hidden size"
    
    def forward(self, x : TensorType["b", "in_seq_len", "d"], latents : TensorType["b", "out_seq_len", "d"], attention_mask : TensorType["b", "in_seq_len"] = None):
        p = x.size(1)
        k = latents.size(1)

        residual = latents.clone()

        h = torch.cat([x, latents], dim = 1) # n = h.size(1) = p + k

        q = self.q(latents)
        q = eo.rearrange(q, 'b k (h d) -> b k h d', h = self.n_heads, d = self.hidden_size//self.n_heads)
        kv = self.kv(h)
        kv = eo.rearrange(kv, 'b n (h d kv) -> kv b n h d', h = self.n_heads, d = self.hidden_size//self.n_heads, kv = 2)
        k, v = kv # all [b n h d] now
        #q = q[:,p:] # queries from latents [b n k d]

        attn_weights = eo.einsum(q, k, 'b k h d, b n h d -> b h k n') # Attention scores per each head
        attn_weights /= (self.hidden_size//self.n_heads)

        if attention_mask is not None:
            # Deal with the attention mask now
            # It's [b, p], but we want it to become [b h k (p+k)] to add it to the latents
            latent_mask = torch.ones_like(latents[...,0]) # [b, k]
            attention_mask = torch.cat([attention_mask, latent_mask], dim = -1) # [b, n]
            attention_mask = eo.repeat(attention_mask, 'b n -> b 1 k n', k = k)
            attention_mask = attention_mask.bool() # Cast to trues/falses
            attention_mask = torch.where(attention_mask, torch.zeros_like(attn_weights), torch.ones_like(attn_weights)*float('-inf'))
            attn_weights += attention_mask

        attn_weights = F.softmax(attn_weights, dim = -1, dtype = torch.float32).to(q.dtype)
        attn_out = eo.einsum(attn_weights, v, 'b h k n, b n h d -> b h k d')
        attn_out = eo.rearrange(attn_out, 'b h k d -> b k (h d)')
        
        attn_out = self.out(attn_out)
        attn_out += residual

        out = self.postnorm(attn_out)
        out = self.mlp(out)
        out += residual

        return out
        
class PerceiverPooling(nn.Module):
    def __init__(self, out_seq_len, n_layers, n_heads, hidden_size):
        super().__init__()

        self.latents = nn.Parameter(torch.randn(out_seq_len, hidden_size))
        self.blocks = nn.ModuleList([PerceiverAttn(n_heads, hidden_size)] * n_layers)
        self.postnorm = RMSNorm(hidden_size)

    def forward(self, x : TensorType["b", "in_seq_len", "d"], attention_mask : TensorType["b", "in_seq_len"] = None):
        latents = eo.repeat(self.latents, 'k d -> b k d', b = x.size(0))

        for block in self.blocks:
            latents = block(x, latents, attention_mask)
        
        latents = self.postnorm(latents)
        return latents

if __name__ == "__main__":
    layer = PerceiverPooling(256, 4, 8, 768)
    layer.cuda()
    x = torch.randn(4, 512, 768).to('cuda')
    mask = torch.ones(4, 512).to('cuda')

    y = layer(x)
    print(y.shape)