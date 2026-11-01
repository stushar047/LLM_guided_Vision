from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LiveVisionEncoder(nn.Module):
    """A ViT conditioned by a pooled embedding from a frozen text tower.

    ``vision`` follows the common timm VisionTransformer interface. The query
    token is inserted after the class token and before image patches, allowing
    self-attention to route instruction information through the image tokens.
    """

    def __init__(
        self,
        vision: nn.Module,
        text_encoder: nn.Module,
        text_dim: int,
        *,
        embedding_dim: int | None = None,
        train_text_encoder: bool = False,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.vision = vision
        self.text_encoder = text_encoder
        self.normalize = normalize
        width = int(vision.cls_token.shape[-1])
        self.query_projection = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, width))
        self.vision_projection = (
            nn.Identity() if embedding_dim is None or embedding_dim == width else nn.Linear(width, embedding_dim, bias=False)
        )
        self.query_position = nn.Parameter(torch.zeros(1, 1, width))
        nn.init.trunc_normal_(self.query_position, std=0.02)

        self.train_text_encoder = train_text_encoder
        for parameter in self.text_encoder.parameters():
            parameter.requires_grad_(train_text_encoder)
        if not train_text_encoder:
            self.text_encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.train_text_encoder:
            self.text_encoder.eval()
        return self

    def _text_features(self, prompts: Sequence[str], tokenize: Callable) -> Tensor:
        tokens = tokenize(list(prompts))
        if isinstance(tokens, dict):
            tokens = {key: value.to(next(self.text_encoder.parameters()).device) for key, value in tokens.items()}
        elif isinstance(tokens, Tensor):
            tokens = tokens.to(next(self.text_encoder.parameters()).device)

        context = torch.enable_grad() if self.train_text_encoder else torch.no_grad()
        with context:
            if isinstance(tokens, dict):
                features = self.text_encoder(**tokens)
            else:
                features = self.text_encoder(tokens)
        if hasattr(features, "pooler_output"):
            features = features.pooler_output
        elif hasattr(features, "text_embeds"):
            features = features.text_embeds
        elif isinstance(features, (tuple, list)):
            features = features[0]
        if features.ndim == 3:
            features = features[:, 0]
        if features.ndim != 2:
            raise ValueError(f"Expected pooled text features with shape [batch, dim], got {tuple(features.shape)}")
        return features

    def encode_text(self, prompts: Sequence[str], tokenize: Callable) -> Tensor:
        features = self._text_features(prompts, tokenize)
        return F.normalize(features, dim=-1) if self.normalize else features

    def encode_image(self, images: Tensor, instructions: Sequence[str], tokenize: Callable) -> Tensor:
        if images.shape[0] != len(instructions):
            raise ValueError("The image batch and instruction list must have the same length")
        query = self._text_features(instructions, tokenize)
        query = self.query_projection(query).to(dtype=images.dtype).unsqueeze(1)

        vision = self.vision
        x = vision.patch_embed(images)
        if x.ndim != 3:
            raise ValueError("patch_embed must return [batch, patches, channels]")
        cls_token = vision.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        if getattr(vision, "pos_embed", None) is not None:
            if vision.pos_embed.shape[1] != x.shape[1]:
                raise ValueError("Backbone positional embedding length does not match patch_embed output")
            x = x + vision.pos_embed
        x = vision.pos_drop(x) if hasattr(vision, "pos_drop") else x
        x = vision.patch_drop(x) if hasattr(vision, "patch_drop") else x
        x = vision.norm_pre(x) if hasattr(vision, "norm_pre") else x

        x = torch.cat((x[:, :1], query + self.query_position, x[:, 1:]), dim=1)
        for block in vision.blocks:
            x = block(x)
        x = vision.norm(x)
        embedding = self.vision_projection(x[:, 0])
        return F.normalize(embedding, dim=-1) if self.normalize else embedding

    def forward(self, images: Tensor, instructions: Sequence[str], tokenize: Callable) -> Tensor:
        return self.encode_image(images, instructions, tokenize)
