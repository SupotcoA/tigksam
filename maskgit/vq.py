import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init


class EmbeddingEMA(nn.Module):
    def __init__(self, num_tokens, codebook_dim, decay=0.99, eps=1e-5):
        super().__init__()
        self.decay = decay
        self.eps = eps
        weight = torch.randn(num_tokens, codebook_dim)
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.cluster_size = nn.Parameter(torch.zeros(num_tokens), requires_grad=False)
        self.embed_avg = nn.Parameter(weight.clone(), requires_grad=False)
        self.update = True

    def forward(self, embed_id):
        return F.embedding(embed_id, self.weight)

    def cluster_size_ema_update(self, new_cluster_size):
        self.cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)

    def embed_avg_ema_update(self, new_embed_avg):
        self.embed_avg.data.mul_(self.decay).add_(new_embed_avg, alpha=1 - self.decay)

    def weight_update(self, num_tokens):
        n = self.cluster_size.sum()
        smoothed_cluster_size = (
                (self.cluster_size + self.eps) / (n + num_tokens * self.eps) * n
        )
        # normalize embedding average with smoothed cluster size
        embed_normalized = self.embed_avg / smoothed_cluster_size.unsqueeze(1)
        self.weight.data.copy_(embed_normalized)

    def reinit(self, probs):
        n, l = self.weight.shape
        for i in range(n):
            if probs[i] == 0:
                init.uniform_(self.weight[i], -1.0, 1.0)


class CodeBook(nn.Module):

    def __init__(self, embed_dim=256, n_embed=512,
                 beta=0.25, decay=0.99, eps=1e-5):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.embed = EmbeddingEMA(num_tokens=n_embed, codebook_dim=embed_dim,
                                  decay=decay, eps=eps)
        # weight[num_embeddings, embedding_dim]
        init.uniform_(self.embed.weight, -1.0, 1.0)  # ?
        self.beta = beta
        self.decay = decay
        self.eps = eps

    def forward(self, z_e: torch.Tensor):
        # b, c, h, w -> b, h, w, c
        z_e = torch.permute(z_e, [0, 2, 3, 1]).contiguous()
        z_flat = z_e.view(-1, self.embed_dim)

        d = torch.cdist(z_flat, self.embed.weight, p=2)

        closest_indices = torch.argmin(d, dim=1)
        z_q = self.embed(closest_indices).view(z_e.shape)
        loss = torch.mean(self.beta * (z_e - z_q.detach()) ** 2)

        encodings = F.one_hot(closest_indices, self.n_embed).type(z_e.dtype)
        # EMA cluster size
        encodings_sum = encodings.sum(axis=0).detach()

        if self.training and self.embed.update:
            with torch.no_grad():
                self.embed.cluster_size_ema_update(encodings_sum)
                # EMA embedding average
                embed_sum = encodings.transpose(0, 1) @ z_flat
                self.embed.embed_avg_ema_update(embed_sum)
                # normalize embed_avg and update weight
                self.embed.weight_update(self.n_embed)

        z_q = z_e + (z_q - z_e).detach()
        # b, h, w, c -> b, c, h, w
        z_q = (torch.permute(z_q, [0, 3, 1, 2])).contiguous()

        return {'z_q': z_q, 'loss': loss, 'encodings_sum': encodings_sum,
                'encodings': closest_indices.detach()}
