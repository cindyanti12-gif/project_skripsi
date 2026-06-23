import torch
import torch.nn as nn

from model.vgg_feature import VGGBackbone


class DETRVGGFusion(nn.Module):

    def __init__(
        self,
        num_classes=2,
        num_queries=25,
        hidden_dim=256
    ):
        super().__init__()

        self.backbone = VGGBackbone()

        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            batch_first=True
        )

        self.query_embed = nn.Embedding(
            num_queries,
            hidden_dim
        )

        self.row_embed = nn.Embedding(100, hidden_dim // 2)
        self.col_embed = nn.Embedding(100, hidden_dim // 2)

        self.class_head = nn.Linear(
            hidden_dim,
            num_classes + 1
        )

        self.bbox_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid()
        )

    def forward(self, images):

        features = self.backbone(images)

        B, C, H, W = features.shape

        # Positional Encoding 2D
        i = torch.arange(W, device=features.device)
        j = torch.arange(H, device=features.device)
        x_embed = self.col_embed(i)
        y_embed = self.row_embed(j)
        pos = torch.cat([
            x_embed.unsqueeze(0).repeat(H, 1, 1),
            y_embed.unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(B, 1, 1, 1)

        src = features.flatten(2).permute(
            0,
            2,
            1
        )

        pos_flat = pos.flatten(2).permute(
            0,
            2,
            1
        )

        query_embed = self.query_embed.weight.unsqueeze(0)

        query_embed = query_embed.repeat(
            B,
            1,
            1
        )

        hs = self.transformer(
            src=src + pos_flat,
            tgt=query_embed
        )

        pred_logits = self.class_head(hs)

        pred_boxes = self.bbox_head(hs)

        return {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes
        }