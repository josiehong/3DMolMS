import torch
import torch.nn as nn
import torch.nn.functional as F
from decimal import *
from typing import Tuple


class MolConv(nn.Module):
    def __init__(self, in_dim, out_dim, k, remove_xyz=False):
        super(MolConv, self).__init__()
        self.k = k
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.remove_xyz = remove_xyz

        self.dist_ff = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, bias=False), nn.BatchNorm2d(1), nn.Sigmoid()
        )
        self.gm2m_ff = nn.Sequential(
            nn.Conv2d(k, 1, kernel_size=1, bias=False), nn.BatchNorm2d(1), nn.Sigmoid()
        )

        if remove_xyz:
            self.update_ff = nn.Sequential(
                nn.Conv2d(in_dim - 3, out_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(negative_slope=0.02),
            )
        else:
            self.update_ff = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(negative_slope=0.02),
            )

    def forward(self, x: torch.Tensor, idx_base: torch.Tensor) -> torch.Tensor:
        dist, gm2, feat_c, feat_n = self._generate_feat(
            x, idx_base, k=self.k, remove_xyz=self.remove_xyz
        )
        """Returned features: 
		dist: torch.Size([batch_size, 1, point_num, k])
		gm2: torch.Size([batch_size, k, point_num, k]) 
		feat_c: torch.Size([batch_size, in_dim, point_num, k]) 
		feat_n: torch.Size([batch_size, in_dim, point_num, k])
		"""
        w1 = self.dist_ff(dist)
        w2 = self.gm2m_ff(gm2)

        feat = torch.mul(w1, w2) * feat_n + (1 - torch.mul(w1, w2)) * feat_c
        feat = self.update_ff(feat)
        feat = feat.mean(dim=-1, keepdim=False)
        return feat

    def _generate_feat(
        self, x: torch.Tensor, idx_base: torch.Tensor, k: int, remove_xyz: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_dims, num_points = x.size()

        # local graph (knn)
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        dist, idx = pairwise_distance.topk(k=k, dim=2)  # (batch_size, num_points, k)
        dist = -dist

        idx = idx + idx_base
        idx = idx.view(-1)

        x = x.transpose(
            2, 1
        ).contiguous()  # (batch_size, num_points, num_dims) -> (batch_size*num_points, num_dims)
        graph_feat = x.view(batch_size * num_points, -1)[idx, :]
        graph_feat = graph_feat.view(batch_size, num_points, k, num_dims)

        # gram matrix
        gm_matrix = torch.matmul(graph_feat, graph_feat.permute(0, 1, 3, 2))

        # double gram matrix
        sub_feat = gm_matrix[:, :, :, 0].unsqueeze(3)
        sub_gm_matrix = torch.matmul(sub_feat, sub_feat.permute(0, 1, 3, 2))
        sub_gm_matrix = F.normalize(sub_gm_matrix, dim=1)

        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

        if remove_xyz:
            return (
                dist.unsqueeze(3).permute(0, 3, 1, 2).contiguous(),
                sub_gm_matrix.permute(0, 3, 1, 2).contiguous(),
                x[:, :, :, 3:].permute(0, 3, 1, 2).contiguous(),
                graph_feat[:, :, :, 3:].permute(0, 3, 1, 2).contiguous(),
            )
        else:
            return (
                dist.unsqueeze(3).permute(0, 3, 1, 2).contiguous(),
                sub_gm_matrix.permute(0, 3, 1, 2).contiguous(),
                x.permute(0, 3, 1, 2).contiguous(),
                graph_feat.permute(0, 3, 1, 2).contiguous(),
            )

    def __repr__(self):
        return (
            self.__class__.__name__
            + " k = "
            + str(self.k)
            + " ("
            + str(self.in_dim)
            + " -> "
            + str(self.out_dim)
            + ")"
        )


class MolConv2(nn.Module):
    def __init__(self, in_dim, out_dim, k, remove_xyz=False):
        super(MolConv2, self).__init__()
        self.k = k
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.remove_xyz = remove_xyz

        self.dist_ff = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, bias=False), nn.BatchNorm2d(1), nn.Sigmoid()
        )

        if remove_xyz:
            self.center_ff = nn.Sequential(
                nn.Conv2d(in_dim - 3, in_dim + k - 3, kernel_size=1, bias=False),
                nn.BatchNorm2d(in_dim + k - 3),
                nn.Sigmoid(),
            )
            self.update_ff = nn.Sequential(
                nn.Conv2d(in_dim + k - 3, out_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(negative_slope=0.02),
            )
        else:
            self.center_ff = nn.Sequential(
                nn.Conv2d(in_dim, in_dim + k, kernel_size=1, bias=False),
                nn.BatchNorm2d(in_dim + k),
                nn.Sigmoid(),
            )
            self.update_ff = nn.Sequential(
                nn.Conv2d(in_dim + k, out_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(negative_slope=0.02),
            )

    def forward(self, x: torch.Tensor, idx_base: torch.Tensor) -> torch.Tensor:
        dist, gm2, feat_c, feat_n = self._generate_feat(
            x, idx_base, k=self.k, remove_xyz=self.remove_xyz
        )
        """Returned features: 
		dist: torch.Size([batch_size, 1, point_num, k])
		gm2: torch.Size([batch_size, k, point_num, k]) 
		feat_c: torch.Size([batch_size, in_dim, point_num, k]) 
		feat_n: torch.Size([batch_size, in_dim, point_num, k])
		"""
        feat_n = torch.cat(
            (feat_n, gm2), dim=1
        )  # torch.Size([batch_size, in_dim+k, point_num, k])
        feat_c = self.center_ff(feat_c)

        w = self.dist_ff(dist)

        feat = w * feat_n + feat_c
        feat = self.update_ff(feat)
        feat = feat.mean(dim=-1, keepdim=False)
        return feat

    def _generate_feat(
        self, x: torch.Tensor, idx_base: torch.Tensor, k: int, remove_xyz: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_dims, num_points = x.size()

        # local graph (knn)
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        dist, idx = pairwise_distance.topk(k=k, dim=2)  # (batch_size, num_points, k)
        dist = -dist

        idx = idx + idx_base
        idx = idx.view(-1)

        x = x.transpose(
            2, 1
        ).contiguous()  # (batch_size, num_points, num_dims) -> (batch_size*num_points, num_dims)
        # print('_double_gram_matrix (x):', torch.any(torch.isnan(x)))
        graph_feat = x.view(batch_size * num_points, -1)[idx, :]
        # print('_double_gram_matrix (graph_feat):', torch.any(torch.isnan(graph_feat)))
        graph_feat = graph_feat.view(batch_size, num_points, k, num_dims)

        # gram matrix
        gm_matrix = torch.matmul(graph_feat, graph_feat.permute(0, 1, 3, 2))
        # print('_double_gram_matrix (gm_matrix):', torch.any(torch.isnan(gm_matrix)))
        # gm_matrix = F.normalize(gm_matrix, dim=1)

        # double gram matrix
        sub_feat = gm_matrix[:, :, :, 0].unsqueeze(3)
        sub_gm_matrix = torch.matmul(sub_feat, sub_feat.permute(0, 1, 3, 2))
        sub_gm_matrix = F.normalize(sub_gm_matrix, dim=1)
        # print('_double_gram_matrix (sub_gm_matrix):', torch.any(torch.isnan(sub_gm_matrix)))

        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

        if remove_xyz:
            return (
                dist.unsqueeze(3).permute(0, 3, 1, 2).contiguous(),
                sub_gm_matrix.permute(0, 3, 1, 2).contiguous(),
                x[:, :, :, 3:].permute(0, 3, 1, 2).contiguous(),
                graph_feat[:, :, :, 3:].permute(0, 3, 1, 2).contiguous(),
            )
        else:
            return (
                dist.unsqueeze(3).permute(0, 3, 1, 2).contiguous(),
                sub_gm_matrix.permute(0, 3, 1, 2).contiguous(),
                x.permute(0, 3, 1, 2).contiguous(),
                graph_feat.permute(0, 3, 1, 2).contiguous(),
            )

    def __repr__(self):
        return (
            self.__class__.__name__
            + " k = "
            + str(self.k)
            + " ("
            + str(self.in_dim)
            + " -> "
            + str(self.out_dim)
            + ")"
        )


class MolConv3(nn.Module):
    def __init__(self, in_dim, out_dim, point_num, k, remove_xyz=False):
        super(MolConv3, self).__init__()
        self.k = k
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.remove_xyz = remove_xyz

        self.dist_ff = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, bias=False),
            nn.LayerNorm((1, point_num, k)),
            nn.Sigmoid(),
        )

        if remove_xyz:
            self.center_ff = nn.Sequential(
                nn.Conv2d(in_dim - 3, in_dim + k - 3, kernel_size=1, bias=False),
                nn.LayerNorm((in_dim + k - 3, point_num, k)),
                nn.Sigmoid(),
            )
            self.update_ff = nn.Sequential(
                nn.Conv2d(in_dim + k - 3, out_dim, kernel_size=1, bias=False),
                nn.LayerNorm((out_dim, point_num, k)),
                nn.Softplus(beta=1.0, threshold=20.0),
            )
        else:
            self.center_ff = nn.Sequential(
                nn.Conv2d(in_dim, in_dim + k, kernel_size=1, bias=False),
                nn.LayerNorm((in_dim + k, point_num, k)),
                nn.Sigmoid(),
            )
            self.update_ff = nn.Sequential(
                nn.Conv2d(in_dim + k, out_dim, kernel_size=1, bias=False),
                nn.LayerNorm((out_dim, point_num, k)),
                nn.Softplus(beta=1.0, threshold=20.0),
            )

    def forward(
        self, x: torch.Tensor, idx_base: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        # Generate features
        dist, gm2, feat_c, feat_n = self._generate_feat(
            x, idx_base, k=self.k, remove_xyz=self.remove_xyz
        )
        """Returned features:
		dist: torch.Size([batch_size, 1, point_num, k])
		gm2: torch.Size([batch_size, k, point_num, k])
		feat_c: torch.Size([batch_size, in_dim, point_num, k])
		feat_n: torch.Size([batch_size, in_dim, point_num, k])
		"""
        feat_n = torch.cat(
            (feat_n, gm2), dim=1
        )  # torch.Size([batch_size, in_dim+k, point_num, k])
        feat_c = self.center_ff(feat_c)
        w = self.dist_ff(dist)

        feat = w * feat_n + feat_c
        feat = self.update_ff(feat)

        # Average pooling along the fourth dimension
        mask_expanded = (
            mask.unsqueeze(1).unsqueeze(-1).expand_as(feat)
        )  # [batch_size, out_dim, point_num, k]
        feat = feat.masked_fill(~mask_expanded, 0.0)  # Set padding points to zero
        valid_counts = mask.sum(dim=1, keepdim=True).clamp(
            min=0.1
        )  # Avoid division by zero
        feat = feat.sum(dim=3) / valid_counts.unsqueeze(
            2
        )  # [batch_size, out_dim, point_num]
        return feat

    def _generate_feat(
        self, x: torch.Tensor, idx_base: torch.Tensor, k: int, remove_xyz: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_dims, num_points = x.size()

        # local graph (knn)
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        dist, idx = pairwise_distance.topk(k=k, dim=2)  # (batch_size, num_points, k)
        dist = -dist

        idx = idx + idx_base
        idx = idx.view(-1)

        x = x.transpose(
            2, 1
        ).contiguous()  # (batch_size, num_points, num_dims) -> (batch_size*num_points, num_dims)
        # print('_double_gram_matrix (x):', torch.any(torch.isnan(x)))
        graph_feat = x.view(batch_size * num_points, -1)[idx, :]
        # print('_double_gram_matrix (graph_feat):', torch.any(torch.isnan(graph_feat)))
        graph_feat = graph_feat.view(batch_size, num_points, k, num_dims)

        # gram matrix
        gm_matrix = torch.matmul(graph_feat, graph_feat.permute(0, 1, 3, 2))
        gm_matrix = F.normalize(gm_matrix, dim=1)
        # print('_double_gram_matrix (gm_matrix):', torch.any(torch.isnan(gm_matrix)))

        # double gram matrix
        sub_feat = gm_matrix[:, :, :, 0].unsqueeze(3)
        sub_gm_matrix = torch.matmul(sub_feat, sub_feat.permute(0, 1, 3, 2))
        sub_gm_matrix = F.normalize(sub_gm_matrix, dim=1)
        # print('_double_gram_matrix (sub_gm_matrix):', torch.any(torch.isnan(sub_gm_matrix)))

        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

        if remove_xyz:
            dist = dist.unsqueeze(3).permute(0, 3, 1, 2).contiguous()
            gm2 = sub_gm_matrix.permute(0, 3, 1, 2).contiguous()
            feat_c = x[:, :, :, 3:].permute(0, 3, 1, 2).contiguous()
            feat_n = graph_feat[:, :, :, 3:].permute(0, 3, 1, 2).contiguous()
        else:
            dist = dist.unsqueeze(3).permute(0, 3, 1, 2).contiguous()
            gm2 = sub_gm_matrix.permute(0, 3, 1, 2).contiguous()
            feat_c = x.permute(0, 3, 1, 2).contiguous()
            feat_n = graph_feat.permute(0, 3, 1, 2).contiguous()

        return dist, gm2, feat_c, feat_n

    def __repr__(self):
        return (
            self.__class__.__name__
            + " k = "
            + str(self.k)
            + " ("
            + str(self.in_dim)
            + " -> "
            + str(self.out_dim)
            + ")"
        )


class MolConv4(nn.Module):
    """SE(3)-invariant variant of MolConv3.

    The only change from MolConv3 is in the Gram matrix computation.
    When remove_xyz=True (first encoder layer, input contains raw xyz in
    dims 0–2), the Gram matrix is built from relative displacement vectors
    (xj − xi) instead of absolute position vectors.  The dot product
    <(xj−xi), (xk−xi)> encodes the angle at atom i between neighbours j
    and k — a quantity that is invariant to both translation and rotation.

    Combined with remove_xyz=True (which strips dims 0–2 from the node
    features), the first layer becomes fully SE(3)-invariant.  All
    subsequent layers receive features that were already produced by an
    SE(3)-invariant operation, so invariance propagates through the
    entire encoder without any further changes.

    For remove_xyz=False layers (all layers after the first), the input
    has no xyz — the Gram matrix falls back to the MolConv3 behaviour
    (feature-space dot products), which is rotation-invariant by
    construction since the features themselves are invariant.
    """

    def __init__(self, in_dim, out_dim, point_num, k, remove_xyz=False):
        super(MolConv4, self).__init__()
        self.k = k
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.remove_xyz = remove_xyz

        self.dist_ff = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, bias=False),
            nn.LayerNorm((1, point_num, k)),
            nn.Sigmoid(),
        )

        if remove_xyz:
            self.center_ff = nn.Sequential(
                nn.Conv2d(in_dim - 3, in_dim + k - 3, kernel_size=1, bias=False),
                nn.LayerNorm((in_dim + k - 3, point_num, k)),
                nn.Sigmoid(),
            )
            self.update_ff = nn.Sequential(
                nn.Conv2d(in_dim + k - 3, out_dim, kernel_size=1, bias=False),
                nn.LayerNorm((out_dim, point_num, k)),
                nn.Softplus(beta=1.0, threshold=20.0),
            )
        else:
            self.center_ff = nn.Sequential(
                nn.Conv2d(in_dim, in_dim + k, kernel_size=1, bias=False),
                nn.LayerNorm((in_dim + k, point_num, k)),
                nn.Sigmoid(),
            )
            self.update_ff = nn.Sequential(
                nn.Conv2d(in_dim + k, out_dim, kernel_size=1, bias=False),
                nn.LayerNorm((out_dim, point_num, k)),
                nn.Softplus(beta=1.0, threshold=20.0),
            )

    def forward(
        self, x: torch.Tensor, idx_base: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        dist, gm2, feat_c, feat_n = self._generate_feat(
            x, idx_base, mask, k=self.k, remove_xyz=self.remove_xyz
        )
        feat_n = torch.cat(
            (feat_n, gm2), dim=1
        )  # [batch_size, in_dim+k, point_num, k]
        feat_c = self.center_ff(feat_c)
        w = self.dist_ff(dist)

        feat = w * feat_n + feat_c
        feat = self.update_ff(feat)

        # Masked average pooling over neighbours
        mask_expanded = mask.unsqueeze(1).unsqueeze(-1).expand_as(feat)
        feat = feat.masked_fill(~mask_expanded, 0.0)
        valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=0.1)
        feat = feat.sum(dim=3) / valid_counts.unsqueeze(2)  # [batch_size, out_dim, point_num]
        return feat

    def _generate_feat(
        self, x: torch.Tensor, idx_base: torch.Tensor, mask: torch.Tensor,
        k: int, remove_xyz: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_dims, num_points = x.size()

        if remove_xyz:
            # Center xyz (dims 0-2) on the real-atom centroid. Distances and the
            # relative-displacement Gram matrix are subtractions of coordinates;
            # when the molecule sits far from the origin these cancel large
            # near-equal values and lose precision under translation. Centering
            # keeps them small and makes translation invariance numerically
            # exact — the centroid shifts with any input translation, so the
            # centered coordinates are identical. Padding atoms are excluded
            # from the centroid via the mask.
            m = mask.view(batch_size, 1, num_points).to(x.dtype)
            centroid = (x[:, :3, :] * m).sum(dim=2, keepdim=True) / m.sum(
                dim=2, keepdim=True
            ).clamp(min=1.0)
            x = torch.cat([x[:, :3, :] - centroid, x[:, 3:, :]], dim=1)

        # kNN graph and distances
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)

        # Exclude zero-padding atoms from REAL atoms' kNN: padding sits at the
        # coordinate origin and would otherwise pollute real neighbourhoods and
        # break translation invariance (a real→padding distance depends on the
        # real atom's absolute position). Padding *query* rows keep selecting
        # padding neighbours (distance 0), so they stay inert — all-zero and
        # constant — exactly as before masking. That matters because the Gram
        # and LayerNorm normalizations run over the atom dimension: a padding
        # row that varied under translation would leak into every real atom.
        # Gathering the unmasked distances at the chosen indices keeps -inf out
        # of the feature pipeline even when a molecule has fewer than k atoms.
        valid_col = mask.unsqueeze(1)  # [batch_size, 1, num_points] neighbour cols
        real_row  = mask.unsqueeze(2)  # [batch_size, num_points, 1] real query rows
        masked_distance = pairwise_distance.masked_fill(~valid_col, float("-inf"))
        masked_distance = torch.where(real_row, masked_distance, pairwise_distance)
        _, idx = masked_distance.topk(k=k, dim=2)  # (batch_size, num_points, k)
        dist = -torch.gather(pairwise_distance, 2, idx)

        idx = idx + idx_base
        idx = idx.view(-1)

        x = x.transpose(2, 1).contiguous()  # [batch_size*num_points, num_dims]
        graph_feat = x.view(batch_size * num_points, -1)[idx, :]
        graph_feat = graph_feat.view(batch_size, num_points, k, num_dims)

        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

        if remove_xyz:
            # SE(3)-invariant Gram matrix: relative displacement vectors (xj − xi).
            # <(xj−xi), (xk−xi)> encodes the angle ∠jik — invariant to rotation.
            rel_xyz = graph_feat[:, :, :, :3] - x[:, :, :, :3]  # [B, N, k, 3]
            gm_matrix = torch.matmul(rel_xyz, rel_xyz.permute(0, 1, 3, 2))  # [B, N, k, k]
            gm_matrix = F.normalize(gm_matrix, dim=1)
            sub_feat = gm_matrix[:, :, :, 0].unsqueeze(3)
            sub_gm_matrix = torch.matmul(sub_feat, sub_feat.permute(0, 1, 3, 2))
            sub_gm_matrix = F.normalize(sub_gm_matrix, dim=1)

            return (
                dist.unsqueeze(3).permute(0, 3, 1, 2).contiguous(),
                sub_gm_matrix.permute(0, 3, 1, 2).contiguous(),
                x[:, :, :, 3:].permute(0, 3, 1, 2).contiguous(),
                graph_feat[:, :, :, 3:].permute(0, 3, 1, 2).contiguous(),
            )
        else:
            # Subsequent layers: input has no xyz.  Features are already
            # SE(3)-invariant (produced by a prior MolConv4 with remove_xyz=True),
            # so feature-space dot products are rotation-invariant by construction.
            gm_matrix = torch.matmul(graph_feat, graph_feat.permute(0, 1, 3, 2))
            gm_matrix = F.normalize(gm_matrix, dim=1)
            sub_feat = gm_matrix[:, :, :, 0].unsqueeze(3)
            sub_gm_matrix = torch.matmul(sub_feat, sub_feat.permute(0, 1, 3, 2))
            sub_gm_matrix = F.normalize(sub_gm_matrix, dim=1)

            return (
                dist.unsqueeze(3).permute(0, 3, 1, 2).contiguous(),
                sub_gm_matrix.permute(0, 3, 1, 2).contiguous(),
                x.permute(0, 3, 1, 2).contiguous(),
                graph_feat.permute(0, 3, 1, 2).contiguous(),
            )

    def __repr__(self):
        return (
            self.__class__.__name__
            + " k = "
            + str(self.k)
            + " ("
            + str(self.in_dim)
            + " -> "
            + str(self.out_dim)
            + ")"
        )

