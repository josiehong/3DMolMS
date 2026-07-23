import torch
import torch.nn as nn
import torch.nn.functional as F
from decimal import *
from typing import Tuple

# encoder layers: MolConv2 (v2, E(3)-invariant, default); MolConv1 (v1, legacy absolute-Gram).
# NB import MolConv1 explicitly for the v1 path — bare `MolConv` is aliased to MolConv2.
from .molconv import MolConv1, MolConv2


# ----------------------------------------
# >>>           encoder part           <<<
# ----------------------------------------
class Encoder(nn.Module):
    def __init__(self, in_dim, layers, emb_dim, point_num, k, chirality=False,
                 encoder_version=2):
        super(Encoder, self).__init__()
        self.emb_dim = emb_dim
        self.encoder_version = int(encoder_version)
        # chirality=True -> SE(3) (reflection-sensitive, chiral tasks); False -> E(3) (default).
        # Only the first layer sees xyz, so it carries the flag.
        if self.encoder_version == 1:
            # Legacy v1 (MolConv1): absolute-position Gram, BatchNorm, not E(3).
            self.hidden_layers = nn.ModuleList(
                [MolConv1(in_dim=in_dim, out_dim=layers[0], k=k, remove_xyz=True)]
            )
            for i in range(1, len(layers)):
                self.hidden_layers.append(
                    MolConv1(in_dim=layers[i - 1], out_dim=layers[i], k=k, remove_xyz=False)
                )
        else:
            self.hidden_layers = nn.ModuleList(
                [
                    MolConv2(
                        in_dim=in_dim,
                        out_dim=layers[0],
                        point_num=point_num,
                        k=k,
                        remove_xyz=True,
                        chirality=chirality,
                    )
                ]
            )
            for i in range(1, len(layers)):
                self.hidden_layers.append(
                    MolConv2(
                        in_dim=layers[i - 1],
                        out_dim=layers[i],
                        point_num=point_num,
                        k=k,
                        remove_xyz=False,
                    )
                )

        self.conv = nn.Sequential(
            nn.Conv1d(emb_dim, emb_dim, kernel_size=1, bias=False),
            nn.LayerNorm((emb_dim, point_num)),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(
        self,
        x: torch.Tensor,
        idx_base: torch.Tensor,
        mask: torch.Tensor,
        return_per_atom: bool = False,
    ) -> torch.Tensor:
        xs = []
        for i, hidden_layer in enumerate(self.hidden_layers):
            inp = x if i == 0 else xs[-1]
            if self.encoder_version == 1:
                tmp_x = hidden_layer(inp, idx_base)          # v1 layers take no mask
            else:
                tmp_x = hidden_layer(inp, idx_base, mask)
            xs.append(tmp_x)

        x = torch.cat(xs, dim=1)  # torch.Size([batch_size, emb_dim, point_num])
        x = self.conv(x)

        # per-atom features before pooling: [batch_size, emb_dim, point_num]
        per_atom = x

        # Apply the mask: Set padding points to a very low value for max pooling and zero for average pooling
        mask_expanded = mask.unsqueeze(1).expand_as(
            x
        )  # [batch_size, emb_dim, point_num]
        x_masked_max = x.masked_fill(
            ~mask_expanded, float("-inf")
        )  # Replace padding with -inf for max pooling
        x_masked_avg = x.masked_fill(
            ~mask_expanded, 0.0
        )  # Replace padding with 0 for average pooling

        # Max pooling along the third dimension
        max_pooled = torch.max(x_masked_max, dim=2)[0]  # [batch_size, emb_dim]

        # Average pooling along the third dimension
        # Count the valid (non-padding) points for each position
        valid_counts = mask.sum(dim=1, keepdim=True).clamp(
            min=0.1
        )  # Avoid division by zero
        avg_pooled = x_masked_avg.sum(dim=2) / valid_counts  # [batch_size, emb_dim]

        pooled = max_pooled + avg_pooled
        if return_per_atom:
            return per_atom, pooled  # [B, emb_dim, N], [B, emb_dim]
        return pooled


# ----------------------------------------
# >>>           decoder part           <<<
# ----------------------------------------
class FCResBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0) -> torch.Tensor:
        super(FCResBlock, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.linear1 = nn.Linear(in_dim, out_dim, bias=False)
        self.bn1 = nn.LayerNorm(out_dim)

        self.linear2 = nn.Linear(out_dim, out_dim, bias=False)
        self.bn2 = nn.LayerNorm(out_dim)

        self.linear3 = nn.Linear(out_dim, out_dim, bias=False)
        self.bn3 = nn.LayerNorm(out_dim)

        self.dp = nn.Dropout(dropout)

    def forward(self, x):
        identity = x

        x = self.bn1(self.linear1(x))
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.bn2(self.linear2(x))
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.bn3(self.linear3(x))

        x = x + F.interpolate(identity.unsqueeze(1), size=x.size()[1]).squeeze()

        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.dp(x)
        return x

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_dim)
            + " -> "
            + str(self.out_dim)
            + ")"
        )


class MSDecoder(nn.Module):
    def __init__(self, in_dim, layers, out_dim, dropout):
        super(MSDecoder, self).__init__()
        self.blocks = nn.ModuleList([FCResBlock(in_dim=in_dim, out_dim=layers[0])])
        for i in range(len(layers) - 1):
            if len(layers) - i > 3:
                self.blocks.append(FCResBlock(in_dim=layers[i], out_dim=layers[i + 1]))
            else:
                self.blocks.append(
                    FCResBlock(in_dim=layers[i], out_dim=layers[i + 1], dropout=dropout)
                )

        self.fc = nn.Linear(layers[-1], out_dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        # x = self.fc(x)
        # return F.glu(torch.cat((x, x), dim=1))
        return self.fc(x)


# -----------------------------------
# >>>           3DMolMS           <<<
# -----------------------------------
class MolNet_MS(nn.Module):
    def __init__(self, config):
        super(MolNet_MS, self).__init__()
        self.add_num = config["add_num"]

        self.encoder = Encoder(
            in_dim=int(config["in_dim"]),
            layers=config["encode_layers"],
            emb_dim=int(config["emb_dim"]),
            point_num=int(config["max_atom_num"]),
            k=int(config["k"]),
            chirality=bool(config.get("chirality", False)),
            encoder_version=int(config.get("encoder_version", 2)),
        )
        self.decoder = MSDecoder(
            in_dim=int(config["emb_dim"] + config["add_num"]),
            layers=config["decode_layers"],
            out_dim=int(
                Decimal(str(config["max_mz"])) // Decimal(str(config["resolution"]))
            ),
            dropout=config["dropout"],
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu"
                )
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        env: torch.Tensor = None,
        idx_base: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Input:
                x:      point set, torch.Size([batch_size, 14, atom_num])
                mask:	 mask for padding points, torch.Size([batch_size, atom_num])
                env:    experimental condiction
                idx_base:   idx for local knn
        """
        if idx_base is None:
            batch_size, num_dims, num_points = x.size()
            idx_base = (
                torch.arange(0, batch_size, device=x.device, dtype=torch.long).view(
                    -1, 1, 1
                )
                * num_points
            )
        x = self.encoder(x, idx_base, mask)  # torch.Size([batch_size, emb_dim])

        # add the encoded adduct
        if self.add_num == 1:
            x = torch.cat((x, torch.unsqueeze(env, 1)), 1)
        elif self.add_num > 1:
            x = torch.cat((x, env), 1)

        # decoder
        x = self.decoder(x)
        return x

# -------------------------------------------------------------------------
# >>>                             3DMol_Oth                             <<<
# 1) This is the model for other regression tasks, including pretrain,
# retention time prediction, collision cross-section prediction...
# 2) The difference between 3DMol_Oth and 3DMol_MS is the configuration
# parameters controlling output dimension, as well as 3DMol_Oth contain scaler.
# -------------------------------------------------------------------------
class MolNet_Oth(nn.Module):
    def __init__(self, config):
        super(MolNet_Oth, self).__init__()
        self.add_num = config["add_num"]
        self.encoder = Encoder(
            in_dim=int(config["in_dim"]),
            layers=config["encode_layers"],
            emb_dim=int(config["emb_dim"]),
            point_num=int(config["max_atom_num"]),
            k=int(config["k"]),
            chirality=bool(config.get("chirality", False)),
            encoder_version=int(config.get("encoder_version", 2)),
        )
        self.decoder = MSDecoder(
            in_dim=int(config["emb_dim"] + config["add_num"]),
            layers=config["decode_layers"],
            out_dim=1,
            dropout=config["dropout"],
        )
        self.scaler = None

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu"
                )
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def set_scaler(self, scaler):
        """Set the scaler for output normalization"""
        self.scaler = scaler

    def fit_scaler(self, targets):
        """Fit a new scaler on given targets"""
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler().fit(targets)
        return self.scaler

    def scale(self, y):
        """Scale the input using the internal scaler"""
        if self.scaler is None:
            return y

        # Convert to numpy, transform, and convert back to tensor
        y_np = y.cpu().detach().numpy().reshape(-1, 1)
        y_scaled = self.scaler.transform(y_np)
        return torch.tensor(y_scaled, dtype=torch.float).reshape(y.shape).to(y.device)

    def unscale(self, y_scaled):
        """Inverse scale the input using the internal scaler"""
        if self.scaler is None:
            return y_scaled

        # Convert to numpy, inverse transform, and convert back to tensor
        y_scaled_np = y_scaled.cpu().detach().numpy().reshape(-1, 1)
        y = self.scaler.inverse_transform(y_scaled_np)
        return (
            torch.tensor(y, dtype=torch.float)
            .reshape(y_scaled.shape)
            .to(y_scaled.device)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        env: torch.Tensor = None,
        idx_base: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Input:
                x:      	point set, torch.Size([batch_size, 14, atom_num])
                mask:	 	mask for padding points, torch.Size([batch_size, atom_num])
                env:    	experimental condiction
                idx_base:   idx for local knn
        """
        if idx_base is None:
            batch_size, num_dims, num_points = x.size()
            idx_base = (
                torch.arange(0, batch_size, device=x.device, dtype=torch.long).view(
                    -1, 1, 1
                )
                * num_points
            )
        x = self.encoder(x, idx_base, mask)  # torch.Size([batch_size, emb_dim])

        # add the encoded adduct
        if self.add_num == 1:
            x = torch.cat((x, torch.unsqueeze(env, 1)), 1)
        elif self.add_num > 1:
            x = torch.cat((x, env), 1)

        # decoder
        x = self.decoder(x)

        return x  # scaled output

    def predict(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        env: torch.Tensor = None,
        idx_base: torch.Tensor = None,
    ) -> torch.Tensor:
        """Get unscaled predictions for inference"""
        with torch.no_grad():
            # Get scaled prediction
            x = self.forward(x, mask, env, idx_base)
            # Unscale
            if self.scaler is not None:
                return self.unscale(x)
            return x  # unscaled output


# -------------------------------------------------------------------------
# >>>                        SSL Pretraining                            <<<
# Masked distance reconstruction on 3D molecular conformers:
#   Randomly mask mask_ratio of atoms (zero all 21 dims including xyz).
#   For each (masked, unmasked) atom pair, predict the Euclidean distance
#   from per-atom encoder features.  The encoder must infer the masked
#   atom's 3D location from its chemical context.
#   Valid because MolConv2 is E(3)-invariant and distances are too.
# -------------------------------------------------------------------------
class MolNet_SSL(nn.Module):
    """Encoder + distance head for masked distance reconstruction pretraining.

    SSL task — Masked distance reconstruction:
        Atoms are masked by zeroing all 21 input dims (including xyz).
        The model predicts Euclidean distances between (masked, unmasked)
        atom pairs from per-atom encoder features.
        Loss: MSELoss on raw distances (Å).
    """

    def __init__(self, config: dict) -> None:
        super(MolNet_SSL, self).__init__()
        emb_dim = int(config["emb_dim"])

        self.encoder = Encoder(
            in_dim=int(config["in_dim"]),
            layers=config["encode_layers"],
            emb_dim=emb_dim,
            point_num=int(config["max_atom_num"]),
            k=int(config["k"]),
            chirality=bool(config.get("chirality", False)),
            encoder_version=int(config.get("encoder_version", 2)),
        )
        self.distance_head = nn.Sequential(
            nn.Linear(emb_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu"
                )
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pair_indices: torch.Tensor,
        idx_base: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x:            [B, 21, N]  – mol features (masked atoms zeroed).
            mask:         [B, N]      – True for valid atoms.
            pair_indices: [B, P, 2]   – (masked_i, unmasked_j) atom index pairs.
            idx_base:     [B, 1, 1]   – kNN offset; computed internally if None.
        Returns:
            dist_pred:    [B, P]      – predicted distances (Å).
        """
        if idx_base is None:
            batch_size, _, num_points = x.size()
            idx_base = (
                torch.arange(0, batch_size, device=x.device, dtype=torch.long)
                .view(-1, 1, 1)
                * num_points
            )

        per_atom, _ = self.encoder(x, idx_base, mask, return_per_atom=True)
        per_atom_t  = per_atom.permute(0, 2, 1)  # [B, N, emb_dim]

        arange = torch.arange(per_atom_t.size(0), device=per_atom_t.device).unsqueeze(1)
        pi = per_atom_t[arange, pair_indices[:, :, 0]]  # [B, P, emb_dim]
        pj = per_atom_t[arange, pair_indices[:, :, 1]]  # [B, P, emb_dim]
        dist_pred = self.distance_head(
            torch.cat([pi, pj], dim=-1)
        ).squeeze(-1)                                    # [B, P]

        return dist_pred
