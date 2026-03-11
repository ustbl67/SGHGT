import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr, pearsonr, t
import torchvision.transforms as transforms
import torchvision.transforms.functional as T_F
from PIL import Image
from tqdm import tqdm

def validate_ten_crop_detailed(model, dataset, device, desc="Validation"):
    """Validates model using 5-crop + horizontal flip (total 10 crops) per image."""
    model.eval()
    preds, gts, paths = [], [], []
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc=desc, leave=False):
            p = dataset.paths[i]
            img = Image.open(p).convert('RGB').resize((dataset.base_size, dataset.base_size), Image.BILINEAR)
            img_t = transforms.ToTensor()(img)
            sal_t = torch.load(dataset.sal_cache[p])
            if sal_t.ndim == 4: sal_t = sal_t.squeeze(0)

            # Generate crops
            crops_img = T_F.five_crop(img_t, (dataset.input_size, dataset.input_size))
            crops_sal = T_F.five_crop(sal_t, (dataset.input_size, dataset.input_size))
            
            # Batch together original and flipped crops
            batch_img = torch.stack([norm(c) for c in (list(crops_img) + [T_F.hflip(c) for c in crops_img])]).to(device)
            batch_sal = torch.stack([c for c in (list(crops_sal) + [T_F.hflip(c) for c in crops_sal])]).to(device)

            # Average results
            out = model(batch_img, batch_sal).mean().item()
            preds.append(out); gts.append(dataset.scores[i].item()); paths.append(p)

    srcc = spearmanr(gts, preds)[0] if len(gts) > 1 else 0.0
    plcc = pearsonr(gts, preds)[0] if len(gts) > 1 else 0.0
    return srcc, plcc, preds, gts, paths

def plot_scatter(predictions, ground_truth, dataset_name, fold_idx, output_dir):
    """Saves a scatter plot comparing predictions vs ground truth."""
    plt.figure(figsize=(10, 8))
    plt.scatter(ground_truth, predictions, alpha=0.5, color='blue')
    
    # Draw identity line
    min_val = min(min(predictions), min(ground_truth))
    max_val = max(max(predictions), max(ground_truth))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    plt.xlabel('Ground Truth MOS')
    plt.ylabel('Predicted Quality Score')
    plt.title(f'Scatter Plot - {dataset_name} (Fold {fold_idx + 1})')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'scatter_fold_{fold_idx + 1}.png'))
    plt.close()

def calculate_confidence_interval(data, confidence=0.95):
    """Computes mean and margin for a 95% confidence interval."""
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), np.std(a, ddof=1) / np.sqrt(n)
    h = se * t.ppf((1 + confidence) / 2., n - 1)
    return m, h

def analyze_experiment_results(dataset_name, srcc_list, plcc_list, output_dir):
    """Calculates summary statistics across multiple experiment folds."""
    srcc_m, srcc_h = calculate_confidence_interval(srcc_list)
    plcc_m, plcc_h = calculate_confidence_interval(plcc_list)
    
    results = {
        'Fold': list(range(1, len(srcc_list) + 1)),
        'SRCC': srcc_list,
        'PLCC': plcc_list
    }
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'final_results.csv'), index=False)
    
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write(f"Experiment Summary for {dataset_name}\n")
        f.write("-" * 40 + "\n")
        f.write(f"SRCC: {srcc_m:.4f} +/- {srcc_h:.4f}\n")
        f.write(f"PLCC: {plcc_m:.4f} +/- {plcc_h:.4f}\n")
    
    print(f"\nFinal Summary for {dataset_name}:")
    print(f"SRCC: {srcc_m:.4f} +/- {srcc_h:.4f}")
    print(f"PLCC: {plcc_m:.4f} +/- {plcc_h:.4f}")
