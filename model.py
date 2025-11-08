import torch
import torch.nn as nn
import torch.nn.functional as F


class ViT(nn.Module):
    def __init__(self, T1, T2, C, H, W, d_model=512, nhead=8, num_layers=4):
        super().__init__()

        self.T1 = T1  # 输入帧数
        self.T2 = T2  # 输出帧数
        self.H = H
        self.W = W

        # 1. 空间编码器 - 适配620×580输入
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(C, 64, 3, padding=1),  # [B, 64, H, W]
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 下采样2倍: 620→310, 580→290
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 再下采样2倍: 310→155, 290→145
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((64, 64)),  # 固定到16×16
            nn.Flatten(),
            nn.Linear(256 * 64 * 64, d_model),
        )

        # 2. 时序Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # 3. 输出查询（要预测的T2帧）
        self.output_queries = nn.Parameter(torch.randn(1, T2, d_model))

        # 4. 空间解码器 - 重建620×580输出
        self.frame_decoder = nn.Sequential(
            nn.Linear(d_model, 256 * 64 * 64),
            nn.ReLU(),
            nn.Unflatten(1, (256, 64, 64)),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 16→32
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 32→64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 64→128
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # 128→256
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),  # 256→512
            nn.ReLU(),
            nn.Conv2d(8, 1, 3, padding=1),  # 保持512
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((H, W))  # 调整到目标尺寸620×580
        )

    def forward(self, x):
        # x: [B, T1, C, H, W]
        B, T1, C, H, W = x.shape

        # 编码每帧图像
        encoded_frames = []
        for t in range(T1):
            frame = x[:, t]  # [B, C, H, W]
            encoded_frame = self.frame_encoder(frame)  # [B, d_model]
            encoded_frames.append(encoded_frame)

        # [B, T1, d_model]
        encoded_sequence = torch.stack(encoded_frames, dim=1)

        # 时序Transformer处理（没有位置编码）
        temporal_features = self.temporal_transformer(encoded_sequence)  # [B, T1, d_model]

        # 简单策略：用最后时刻的特征 + 查询向量
        last_frame_feature = temporal_features[:, -1:]  # [B, 1, d_model]
        output_queries = self.output_queries.repeat(B, 1, 1)  # [B, T2, d_model]

        # 简单相加融合
        future_features = last_frame_feature + output_queries  # [B, T2, d_model]

        # 解码回图像帧
        output_frames = []
        for t in range(self.T2):
            frame_feature = future_features[:, t]  # [B, d_model]
            output_frame = self.frame_decoder(frame_feature)  # [B, 1, H, W]
            output_frames.append(output_frame)

        # [B, T2, H, W]
        output = torch.stack(output_frames, dim=1).squeeze(2)

        return output