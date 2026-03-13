"""
SimCLR projection head: 2-layer MLP that maps the encoder representation
to the contrastive embedding space.

This head is used ONLY during pre-training and discarded afterwards.
The encoder representations (not the projected embeddings) are used for
downstream classification. This is a key finding of the SimCLR paper.
"""

import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """
    g(h) = W2 * ReLU(BN(W1 * h))

    Output is L2-normalised to the unit hypersphere, which is required by
    the NT-Xent loss (cosine similarity in contrastive loss).

    Args:
        input_dim   : dimension of encoder output (2048 for ResNet50).
        hidden_dim  : hidden layer dimension (2048 recommended in SimCLR paper).
        output_dim  : contrastive embedding dimension (128 recommended).
    """

    def __init__(self, input_dim: int = 2048, hidden_dim: int = 2048, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            # No activation at output — vectors are normalised externally
        )

    def forward(self, h):
        """h: (B, input_dim) -> z: (B, output_dim), unit-normalised"""
        z = self.net(h)
        return F.normalize(z, dim=1)
