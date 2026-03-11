import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class AttentionPooling(nn.Module):
    """基于自注意力的池化层"""
    def __init__(self, in_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(in_dim, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.attn(x)
        x_pool = (x * w).sum(dim=(2, 3)) / (w.sum(dim=(2, 3)) + 1e-8)
        return x_pool

class SaliencyEncoder(nn.Module):
    """显著性特征提取器"""
    def __init__(self):
        super().__init__()
        self.enc3 = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(256, 384, 3, stride=1, padding=1), nn.ReLU()
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(384, 768, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(768, 768, 3, stride=1, padding=1), nn.ReLU()
        )

    def forward(self, x):
        f3 = self.enc3(x)
        f4 = self.enc4(f3)
        return f3, f4

class SQT_HGR_Model(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.backbone = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        self.features = self.backbone.features
        self.sal_encoder = SaliencyEncoder()

        # 纹理分支 (Level 3)
        self.tex_fusion = nn.Sequential(nn.Conv2d(384 * 2, 384, 1), nn.BatchNorm2d(384), nn.GELU())
        self.tex_pool = AttentionPooling(384)
        self.tex_head = nn.Sequential(nn.Linear(384, 128), nn.GELU(), nn.Dropout(dropout_rate), nn.Linear(128, 1))

        # 语义分支 (Level 4)
        self.sem_fusion = nn.Sequential(nn.Conv2d(768 * 2, 768, 1), nn.BatchNorm2d(768), nn.GELU())
        self.sem_pool = AttentionPooling(768)
        self.sem_head = nn.Sequential(nn.Linear(768, 256), nn.GELU(), nn.Dropout(dropout_rate), nn.Linear(256, 1))

        # 门控控制器
        self.gate_net = nn.Sequential(nn.Linear(384 + 768, 256), nn.GELU(), nn.Linear(256, 1), nn.Sigmoid())
        self.sal_pool3 = AttentionPooling(384)
        self.sal_pool4 = AttentionPooling(768)

    def forward(self, x_img, x_sal):
        x = x_img
        f_img3, f_img4 = None, None
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == 5 or i == 7:
                B, L, C = x.shape if x.ndim == 3 else (x.shape[0], x.shape[1]*x.shape[2], x.shape[3])
                H = W = int(L ** 0.5)
                cur = x.transpose(1, 2).reshape(B, C, H, W) if x.ndim == 3 else x.permute(0, 3, 1, 2)
                if i == 5: f_img3 = cur
                if i == 7: f_img4 = cur; break

        f_sal3, f_sal4 = self.sal_encoder(x_sal)
        v_tex = self.tex_pool(self.tex_fusion(torch.cat([f_img3, f_sal3], dim=1)))
        v_sem = self.sem_pool(self.sem_fusion(torch.cat([f_img4, f_sal4], dim=1)))
        
        alpha = self.gate_net(torch.cat([self.sal_pool3(f_sal3), self.sal_pool4(f_sal4)], dim=1))
        return alpha * self.tex_head(v_tex) + (1 - alpha) * self.sem_head(v_sem)

class RankLoss(nn.Module):
    def forward(self, preds, targets):
        preds, targets = preds.view(-1), targets.view(-1)
        pred_diff = preds.unsqueeze(0) - preds.unsqueeze(1)
        target_diff = targets.unsqueeze(0) - targets.unsqueeze(1)
        target_sign = (torch.sign(target_diff) + 1) / 2
        mask = (torch.abs(target_diff) > 0).float()
        loss = F.binary_cross_entropy_with_logits(pred_diff, target_sign, reduction='none')
        return (loss * mask).sum() / (mask.sum() + 1e-8)

class TotalLoss(nn.Module):
    def __init__(self, lambda_mse=1.0, lambda_rank=1.5):
        super().__init__()
        self.mse, self.rank = nn.MSELoss(), RankLoss()
        self.l_mse, self.l_rank = lambda_mse, lambda_rank
    def forward(self, p, t):
        return self.l_mse * self.mse(p, t.unsqueeze(1)) + self.l_rank * self.rank(p, t)
