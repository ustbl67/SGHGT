import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class AttentionPooling(nn.Module):
    """
    Self-Attention based pooling layer that computes weights for each spatial location.
    """
    def __init__(self, in_dim):
        super(AttentionPooling, self).__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(in_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Generate attention map
        weights = self.attn(x)
        # Weighted spatial averaging
        x_pool = (x * weights).sum(dim=(2, 3)) / (weights.sum(dim=(2, 3)) + 1e-8)
        return x_pool

class SaliencyEncoder(nn.Module):
    """
    Encoder to extract features from saliency maps at different hierarchical levels.
    """
    def __init__(self):
        super(SaliencyEncoder, self).__init__()
        # Initial layers to process saliency map down to intermediate resolution
        self.enc3 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1), nn.ReLU()
        )
        # Further downsampling to deep semantic resolution
        self.enc4 = nn.Sequential(
            nn.Conv2d(384, 768, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(768, 768, kernel_size=3, stride=1, padding=1), nn.ReLU()
        )

    def forward(self, x):
        f3 = self.enc3(x)
        f4 = self.enc4(f3)
        return f3, f4

class SQT_HGR_Model(nn.Module):
    """
    SQT-HGR: Swin-Transformer based Hierarchical Gated Regression for Image Quality Assessment.
    """
    def __init__(self, dropout_rate=0.3):
        super(SQT_HGR_Model, self).__init__()
        # Load pre-trained Swin-Transformer Tiny
        self.backbone = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        self.features = self.backbone.features
        self.sal_encoder = SaliencyEncoder()

        # Texture Branch (Level 3 features)
        self.tex_fusion = nn.Sequential(
            nn.Conv2d(384 * 2, 384, kernel_size=1),
            nn.BatchNorm2d(384),
            nn.GELU()
        )
        self.tex_pool = AttentionPooling(384)
        self.tex_head = nn.Sequential(
            nn.Linear(384, 128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )

        # Semantic Branch (Level 4 features)
        self.sem_fusion = nn.Sequential(
            nn.Conv2d(768 * 2, 768, kernel_size=1),
            nn.BatchNorm2d(768),
            nn.GELU()
        )
        self.sem_pool = AttentionPooling(768)
        self.sem_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1)
        )

        # Gating Controller to adaptively fuse Texture and Semantic scores
        self.gate_net = nn.Sequential(
            nn.Linear(384 + 768, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.sal_pool3 = AttentionPooling(384)
        self.sal_pool4 = AttentionPooling(768)

    def forward(self, x_img, x_sal):
        x = x_img
        f_img3, f_img4 = None, None
        
        # Extract hierarchical features from Swin-T
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == 5: # Level 3 stage
                if x.ndim == 3: # Handle [B, L, C] Swin output
                    B, L, C = x.shape
                    H = W = int(L ** 0.5)
                    f_img3 = x.transpose(1, 2).reshape(B, C, H, W)
                else:
                    f_img3 = x.permute(0, 3, 1, 2)
            if i == 7: # Level 4 stage
                if x.ndim == 3:
                    B, L, C = x.shape
                    H = W = int(L ** 0.5)
                    f_img4 = x.transpose(1, 2).reshape(B, C, H, W)
                else:
                    f_img4 = x.permute(0, 3, 1, 2)
                break

        # Extract saliency features
        f_sal3, f_sal4 = self.sal_encoder(x_sal)

        # Texture Score Regression
        f_tex = self.tex_fusion(torch.cat([f_img3, f_sal3], dim=1))
        s_tex = self.tex_head(self.tex_pool(f_tex))

        # Semantic Score Regression
        f_sem = self.sem_fusion(torch.cat([f_img4, f_sal4], dim=1))
        s_sem = self.sem_head(self.sem_pool(f_sem))

        # Gating mechanism
        g3 = self.sal_pool3(f_sal3)
        g4 = self.sal_pool4(f_sal4)
        alpha = self.gate_net(torch.cat([g3, g4], dim=1))

        # Final score as gated sum
        final_score = alpha * s_tex + (1 - alpha) * s_sem
        return final_score

class RankLoss(nn.Module):
    """
    Ranking loss to encourage the model to learn the relative order of image quality.
    """
    def forward(self, preds, targets):
        preds = preds.view(-1)
        targets = targets.view(-1)
        
        # Calculate pairwise differences
        pred_diff = preds.unsqueeze(0) - preds.unsqueeze(1)
        target_diff = targets.unsqueeze(0) - targets.unsqueeze(1)
        
        # Get ground truth relationship (-1, 0, or 1) converted to probability (0 to 1)
        target_sign = (torch.sign(target_diff) + 1) / 2
        
        # Mask to ignore pairs with the same ground truth score
        mask = (torch.abs(target_diff) > 0).float()
        
        # Binary Cross Entropy over pairwise differences
        loss = F.binary_cross_entropy_with_logits(pred_diff, target_sign, reduction='none')
        return (loss * mask).sum() / (mask.sum() + 1e-8)

class TotalLoss(nn.Module):
    """
    Combined loss function: MSE + Lambda * RankLoss
    """
    def __init__(self, lambda_mse=1.0, lambda_rank=1.0):
        super(TotalLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.rank = RankLoss()
        self.lambda_mse = lambda_mse
        self.lambda_rank = lambda_rank

    def forward(self, preds, targets):
        loss_mse = self.mse(preds, targets.unsqueeze(1))
        loss_rank = self.rank(preds, targets)
        return self.lambda_mse * loss_mse + self.lambda_rank * loss_rank
