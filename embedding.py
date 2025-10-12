from timm.layers import PatchEmbed
import torch
import torch.nn as nn
import torch.nn.functional as F

class RoutingModule(nn.Module):

    def __init__(self, embed_dim, device=None, dtype=None):
        self.embed_dim = embed_dim
        super().__init__()
        self.q_proj_layer = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj_layer = nn.Linear(embed_dim, embed_dim, bias=False)
        with torch.no_grad():
            self.q_proj_layer.weight.copy_(torch.eye(embed_dim))
            self.k_proj_layer.weight.copy_(torch.eye(embed_dim))
        # Ensures this initialization is not overwritten by global weight init
        self.q_proj_layer.weight._no_reinit = True
        self.k_proj_layer.weight._no_reinit = True

    def forward(self, hidden_states, cu_seqlens=None, mask=None):
        
        assert (mask is not None) or (cu_seqlens is not None), "Either mask or cu_seqlens must be provided"

        if cu_seqlens is not None:
            # We are in packed mode, so hidden_states is (T, D). Make it (B, T, D)
            hidden_states = hidden_states.unsqueeze(0)

        cos_sim = torch.einsum(
            "b l d, b l d -> b l",
            F.normalize(self.q_proj_layer(hidden_states[:, :-1]), dim=-1),
            F.normalize(self.k_proj_layer(hidden_states[:, 1:]), dim=-1),
        )
        # this clamp should no-op as long as no precision issues are encountered
        boundary_prob = torch.clamp(((1 - cos_sim) / 2), min=0.0, max=1.0)

        # Force boundary probability of the first element to 1.0
        PAD_PROB = 1.0
        boundary_prob = F.pad(boundary_prob, (1, 0), "constant", PAD_PROB)

        if cu_seqlens is not None:
            boundary_prob = boundary_prob.squeeze(0)
            boundary_prob[cu_seqlens[:-1]] = PAD_PROB

        boundary_prob = torch.stack(((1 - boundary_prob), boundary_prob), dim=-1)

        selected_idx = torch.argmax(boundary_prob, dim=-1)

        boundary_mask = selected_idx == 1  # (shape hidden_states.shape[:-1])
        if mask is not None:
            # No invalid tokens can be selected
            boundary_mask = boundary_mask & mask

        selected_probs = boundary_prob.gather(
            dim=-1, index=selected_idx.unsqueeze(-1)
        )  # (shape hidden_states.shape[:-1], 1)

        return boundary_prob, boundary_mask, selected_probs

        # return RoutingModuleOutput(
        #     boundary_prob=boundary_prob,  # (shape hidden_states.shape[:-1], 2)
        #     boundary_mask=boundary_mask,  # (shape hidden_states.shape[:-1])
        #     selected_probs=selected_probs,  # (shape hidden_states.shape[:-1], 1)
        # )


class ChunkLayer(nn.Module):

    def forward(self, hidden_states, boundary_mask, cu_seqlens=None, mask=None):
        assert (mask is not None) or (
            cu_seqlens is not None
        ), "Either mask or cu_seqlens must be provided"

        if cu_seqlens is not None:
            next_hidden_states = hidden_states[boundary_mask]
            next_cu_seqlens = F.pad(boundary_mask.cumsum(dim=0)[cu_seqlens[1:] - 1], (1, 0))
            next_max_seqlen = int((next_cu_seqlens[1:] - next_cu_seqlens[:-1]).max())
            next_mask = None
        else:
            next_cu_seqlens = None
            num_tokens = boundary_mask.sum(dim=-1)
            next_max_seqlen = int(num_tokens.max())

            device = hidden_states.device
            L = hidden_states.shape[1]
            token_idx = (
                torch.arange(L, device=device)[None, :] + (~boundary_mask).long() * L
            )
            seq_sorted_indices = torch.argsort(token_idx, dim=1)

            next_hidden_states = torch.gather(
                hidden_states,
                dim=1,
                index=seq_sorted_indices[:, :next_max_seqlen, None].expand(
                    -1, -1, hidden_states.shape[-1]
                ),
            )

            next_mask = (
                torch.arange(next_max_seqlen, device=device)[None, :]
                < num_tokens[:, None]
            )
            next_max_seqlen = None

        return next_hidden_states, next_cu_seqlens, next_max_seqlen, next_mask


class Embeddings(nn.Module):

    def __init__(self, img_size, patch_size, embed_dim):
        super().__init__()

        self.embed_dim = embed_dim
        self.patch_embeddings = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)

        self.routing_module = RoutingModule(embed_dim=embed_dim)
        self.chunk_layer = ChunkLayer()
        
        self.ratio_loss_cache = None
        self.downsampling_factor = 3

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        # Create position embeddings for the [CLS] token and the patch embeddings
        # Add 1 to the sequence length for the [CLS] token
        # self.position_embeddings = nn.Parameter(torch.randn(1, self.patch_embeddings.num_patches + 1, embed_dim))

    def _packed_to_batches(self, x, cu_seqlens, max_len, pad_value=0.0):
        B = cu_seqlens.numel() - 1
        D = x.shape[-1]

        x_dense = x.new_full((B, max_len, D), pad_value)

        for b in range(B):
            start, end = cu_seqlens[b].item(), cu_seqlens[b + 1].item()
            length = end - start
            x_dense[b, :length] = x[start:end]

        return x_dense

    def get_patch_boundaries(self, x):
        with torch.no_grad():
            imgs = x.clone()
            x = self.patch_embeddings(x)

            batch_size, _, _ = x.size()
            
            lengths = torch.tensor([x.shape[1]] * batch_size)
            cu_seqlens = torch.cat([torch.tensor([0]), torch.cumsum(lengths, dim=0)])

            x = x.reshape(-1, self.embed_dim)

            # Pass through some encoder
            # x = self.enc(x)

            # Predicting boundaries
            boundary_prob, boundary_mask, selected_probs = self.routing_module(x, cu_seqlens)

            return imgs, boundary_mask.reshape(batch_size, self.patch_embeddings.grid_size[0], self.patch_embeddings.grid_size[1])
        
    def ratio_loss(self, boundary_prob, boundary_mask):

        tokenized_prob = boundary_prob[..., -1]
        true_ratio = boundary_mask.float().mean()
        average_prob = tokenized_prob.float().mean()
        N = self.downsampling_factor

        return (
            (1 - true_ratio) * (1 - average_prob) +
            (true_ratio) * (average_prob) * (N-1)
        ) * N / (N-1)
    
    def forward(self, x):
        x = self.patch_embeddings(x)

        batch_size, _, _ = x.size()
        
        lengths = torch.tensor([x.shape[1]] * batch_size)
        cu_seqlens = torch.cat([torch.tensor([0]), torch.cumsum(lengths, dim=0)])

        x = x.reshape(-1, self.embed_dim)

        # Pass through some encoder
        # x = self.enc(x)

        # Predicting boundaries
        boundary_prob, boundary_mask, selected_probs = self.routing_module(x, cu_seqlens)
        x, next_cu_seqlens, next_max_seqlen, _ = self.chunk_layer(x, boundary_mask, cu_seqlens)

        # Save ratio loss
        self.ratio_loss_cache = self.ratio_loss(boundary_prob, boundary_mask)

        x = self._packed_to_batches(x, next_cu_seqlens, next_max_seqlen)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)        
        
        # x = x + self.position_embeddings
        # return x, boundary_mask.reshape(batch_size, self.patch_embeddings.grid_size[0], self.patch_embeddings.grid_size[1])
        return x
