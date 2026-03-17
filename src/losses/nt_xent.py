"""
NT-Xent Loss (Normalized Temperature-scaled Cross-Entropy).

This is the contrastive objective used in SimCLR:
  Chen et al., "A Simple Framework for Contrastive Learning of Visual
  Representations", ICML 2020. https://arxiv.org/abs/2002.05709

For a batch of N images we obtain 2N augmented views.
For each view i, its positive pair is the other view j of the SAME image.
All other 2(N-1) views in the batch are treated as negatives (in-batch negatives).

Loss for a single positive pair (i, j):
    l(i,j) = -log [ exp(sim(zi, zj) / τ) / Σ_{k≠i} exp(sim(zi, zk) / τ) ]

Total loss averages over all 2N anchors (both directions of each pair).

Implementation note:
  The positive pair index for view i is:
    - view1[i] (index i)       -> positive is view2[i] (index i + N)
    - view2[i] (index i + N)   -> positive is view1[i] (index i)
  Encoded as: labels = [N, N+1, ..., 2N-1, 0, 1, ..., N-1]
  This ensures CrossEntropyLoss treats the correct view as the target class.
"""

import torch
import torch.nn as nn


class NTXentLoss(nn.Module):
    """
    Args:
        temperature : τ in the loss formula (default 0.07, as in SimCLR paper).
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1, z2 : (N, D) L2-normalised embeddings from the projection head.
        Returns:
            Scalar loss value.
        """
        N = z1.shape[0]
        device = z1.device

        # Concatenate both views: shape (2N, D)
        z = torch.cat([z1, z2], dim=0)

        # Pairwise cosine similarity matrix: (2N, 2N)
        # z is already L2-normalised, so dot product == cosine similarity
        sim = torch.mm(z, z.T) / self.temperature

        # Mask self-similarity on the diagonal (a view is not its own negative)
        mask = torch.eye(2 * N, dtype=torch.bool, device=device)
        sim.masked_fill_(mask, float("-inf"))

        # Ground-truth positive pair indices:
        #   for z1[i] at row i         -> positive column is i + N
        #   for z2[i] at row i + N     -> positive column is i
        labels = torch.cat([
            torch.arange(N, 2 * N, device=device),
            torch.arange(0, N, device=device),
        ])

        loss = self.criterion(sim, labels)
        return loss
