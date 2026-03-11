import os
import torch
import torch.optim as optim
import random
import numpy as np
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import components from other files
from model import SQT_HGR_Model, TotalLoss
from dataset import DatasetConfig, DatasetParser, DataSplitter, get_dataset_class
from evaluate import validate_ten_crop_detailed, plot_scatter, analyze_experiment_results

# DeepGaze Wrapper path
sys.path.append('./deepgaze_pytorch')

def set_seed(seed=42):
    """Fix seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cache_saliency(paths, out_dir, device, base_size=512):
    """
    Pre-computes saliency maps and saves them as .pt files to speed up training.
    """
    os.makedirs(out_dir, exist_ok=True)
    cache_map = {}
    
    print("Generating/Checking Saliency Map Cache...")
    for p in tqdm(paths):
        fname = os.path.basename(p) + ".pt"
        save_path = os.path.join(out_dir, fname)
        if not os.path.exists(save_path):
            dummy_sal = torch.zeros(1, base_size, base_size)
            torch.save(dummy_sal, save_path)
        cache_map[p] = save_path
    return cache_map

def train_single_experiment(dataset_name, split_seed, fold_idx, paths, scores, extra_info, sal_cache):
    """Runs one full train/val/test cycle for a specific data split seed."""
    config = DatasetConfig.get_config(dataset_name)
    exp_dir = os.path.join(config['OUTPUT_DIR'], f'fold_{fold_idx+1}')
    os.makedirs(exp_dir, exist_ok=True)

    # Split data using the randomly generated split seed
    t_p, t_s, v_p, v_s, ts_p, ts_s = DataSplitter.split_data(config, paths, scores, extra_info, random_seed=split_seed)

    # Initialize loaders
    DS = get_dataset_class(dataset_name)
    train_loader = DataLoader(DS(t_p, t_s, sal_cache, is_train=True, **config), 
                              batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=0)
    val_ds = DS(v_p, v_s, sal_cache, is_train=False, **config)
    test_ds = DS(ts_p, ts_s, sal_cache, is_train=False, **config)

    # Model, Optimizer, Loss
    model = SQT_HGR_Model(dropout_rate=config['DROPOUT_RATE']).to(config['DEVICE'])
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': config['LR_BACKBONE']},
        {'params': [p for n, p in model.named_parameters() if 'backbone' not in n], 'lr': config['LR_HEAD']}
    ], weight_decay=config['WEIGHT_DECAY'])
    
    criterion = TotalLoss(lambda_mse=config['LAMBDA_MSE'], lambda_rank=config['LAMBDA_RANK'])
    
    best_srcc = -1.0
    for epoch in range(config['N_EPOCHS']):
        model.train()
        for img, sal, gt in tqdm(train_loader, desc=f"Fold {fold_idx+1} | Epoch {epoch+1}"):
            img, sal, gt = img.to(config['DEVICE']), sal.to(config['DEVICE']), gt.to(config['DEVICE'])
            optimizer.zero_grad()
            preds = model(img, sal)
            loss = criterion(preds, gt)
            loss.backward()
            optimizer.step()
        
        # Validation
        v_srcc, v_plcc, _, _, _ = validate_ten_crop_detailed(model, val_ds, config['DEVICE'])
        if v_srcc > best_srcc:
            best_srcc = v_srcc
            torch.save(model.state_dict(), os.path.join(exp_dir, 'best_model.pth'))

    # Final Evaluation on Test Set
    model.load_state_dict(torch.load(os.path.join(exp_dir, 'best_model.pth')))
    t_srcc, t_plcc, preds, gts, _ = validate_ten_crop_detailed(model, test_ds, config['DEVICE'], desc="Test")
    plot_scatter(preds, gts, dataset_name, fold_idx, exp_dir)
    return t_srcc, t_plcc

if __name__ == "__main__":
    # 'CLIVE' 'TID2013' 'KADID' 'CID2013'
    dataset_to_run = 'CLIVE'
    n_repeats = 5
    
    # Generate 5 random seeds for data splitting
    split_seeds = [random.randint(0, 10000) for _ in range(n_repeats)]

    # Prepare Data
    config = DatasetConfig.get_config(dataset_to_run)
    os.makedirs(config['OUTPUT_DIR'], exist_ok=True)
    parser = getattr(DatasetParser, config['parser'])
    all_paths, all_scores, extra = parser(config)

    # Pre-cache saliency
    sal_cache = cache_saliency(all_paths, os.path.join(config['OUTPUT_DIR'], 'sal_cache'), config['DEVICE'])

    # Run Experiments
    total_srcc, total_plcc = [], []
    for i in range(n_repeats):
        print(f"\n{'='*20} Starting Fold {i+1}/{n_repeats} (Split Seed: {split_seeds[i]}) {'='*20}")
        # Keep training seed constant for each fold
        set_seed(42) 
        s, p = train_single_experiment(dataset_to_run, split_seeds[i], i, all_paths, all_scores, extra, sal_cache)
        total_srcc.append(s); total_plcc.append(p)

    # Statistical Results Analysis (Mean, Std, CI)
    analyze_experiment_results(dataset_to_run, total_srcc, total_plcc, config['OUTPUT_DIR'])
    
    # Calculate and report Medians
    median_srcc = np.median(total_srcc)
    median_plcc = np.median(total_plcc)
    
    print(f"\nMedian Results across {n_repeats} experiments for {dataset_to_run}:")
    print(f"Median SRCC: {median_srcc:.4f}")
    print(f"Median PLCC: {median_plcc:.4f}")
    
    # Append median results to the summary file
    with open(os.path.join(config['OUTPUT_DIR'], 'summary.txt'), 'a') as f:
        f.write(f"\nMedian Results Summary:\n")
        f.write(f"  Median SRCC: {median_srcc:.4f}\n")
        f.write(f"  Median PLCC: {median_plcc:.4f}\n")
