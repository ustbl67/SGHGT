import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, t
import torchvision.transforms as transforms
import torchvision.transforms.functional as T_F
from PIL import Image
from tqdm import tqdm
import os

def validate_ten_crop_detailed(model, dataset, device, desc="Validation"):
    model.eval()
    predictions, ground_truth, image_paths = [], [], []
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc=desc, leave=False):
            p = dataset.paths[i]
            img = Image.open(p).convert('RGB').resize((dataset.base_size, dataset.base_size), Image.BILINEAR)
            img_t = transforms.ToTensor()(img)
            sal_t = torch.load(dataset.sal_cache[p]).squeeze(0) if torch.load(dataset.sal_cache[p]).ndim == 4 else torch.load(dataset.sal_cache[p])
            
            crops_img = T_F.five_crop(img_t, (dataset.input_size, dataset.input_size))
            crops_sal = T_F.five_crop(sal_t, (dataset.input_size, dataset.input_size))
            
            batch_img = torch.stack([norm(c) for c in list(crops_img) + [T_F.hflip(c) for c in crops_img]]).to(device)
            batch_sal = torch.stack([c for c in list(crops_sal) + [T_F.hflip(c) for c in crops_sal]]).to(device)

            pred_score = model(batch_img, batch_sal).mean().item()
            predictions.append(pred_score)
            ground_truth.append(dataset.scores[i].item())
            image_paths.append(p)

    srcc = spearmanr(ground_truth, predictions)[0] if len(ground_truth) >= 2 else 0.0
    plcc = pearsonr(ground_truth, predictions)[0] if len(ground_truth) >= 2 else 0.0
    return srcc, plcc, predictions, ground_truth, image_paths


def plot_scatter(preds, gts, dataset_name, fold_idx, output_dir):
    plt.figure(figsize=(8, 6))
    plt.scatter(gts, preds, alpha=0.5)
    plt.plot([min(gts), max(gts)], [min(gts), max(gts)], 'r--')
    plt.xlabel('Ground Truth'); plt.ylabel('Predictions')
    plt.title(f'{dataset_name} Fold {fold_idx+1}')
    plt.savefig(os.path.join(output_dir, f'scatter_{fold_idx+1}.png'))
    plt.close()
