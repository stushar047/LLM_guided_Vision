from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def live_contrastive_loss(
    image_embeddings: Tensor,
    answer_embeddings: Tensor,
    *,
    temperature: float = 0.07,
    negative_embeddings: Tensor | None = None,
) -> Tensor:
    """Symmetric image-to-answer contrastive loss with optional hard negatives.

    Other examples in the batch serve as negatives. Explicit negatives have
    shape ``[batch, negatives, dim]`` and are associated with each image.
    """
    if image_embeddings.ndim != 2 or answer_embeddings.ndim != 2:
        raise ValueError("Image and answer embeddings must be rank-2 tensors")
    if image_embeddings.shape != answer_embeddings.shape:
        raise ValueError("Image and answer embeddings must have matching shapes")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    image = F.normalize(image_embeddings, dim=-1)
    answer = F.normalize(answer_embeddings, dim=-1)
    logits = image @ answer.T / temperature
    targets = torch.arange(logits.shape[0], device=logits.device)
    image_to_answer = F.cross_entropy(logits, targets)
    answer_to_image = F.cross_entropy(logits.T, targets)
    loss = (image_to_answer + answer_to_image) / 2

    if negative_embeddings is not None:
        if negative_embeddings.ndim != 3 or negative_embeddings.shape[0] != image.shape[0]:
            raise ValueError("negative_embeddings must have shape [batch, negatives, dim]")
        negatives = F.normalize(negative_embeddings, dim=-1)
        positive_score = (image * answer).sum(dim=-1, keepdim=True)
        negative_scores = torch.einsum("bd,bnd->bn", image, negatives)
        local_logits = torch.cat((positive_score, negative_scores), dim=1) / temperature
        local_targets = torch.zeros(image.shape[0], dtype=torch.long, device=image.device)
        loss = (loss + F.cross_entropy(local_logits, local_targets)) / 2
    return loss
