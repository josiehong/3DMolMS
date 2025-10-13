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
