import os
import torch
import random
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader
from sqt_hgr import SQT_HGR_Model, TotalLoss
from evaluate import validate_ten_crop_detailed, calculate_confidence_interval, plot_scatter
from dataset import DatasetConfig, DatasetParser, DataSplitter, get_dataset_class

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def run_experiment(dataset_name, n_repeats=5, manual_seeds=[180, 120, 140, 160, 100]):
    config = DatasetConfig.get_config(dataset_name)
    parser = getattr(DatasetParser, config['parser'])
    paths, scores, extra_info = parser(config)
    
    # 这里省略了 cache_saliency 调用，建议保留原文件中的定义
    sal_cache = {} # 应调用 cache_saliency 获取

    all_srcc, all_plcc = [], []
    training_seed = 142

    for i in range(n_repeats):
        set_seed(training_seed)
        split_seed = manual_seeds[i]
        
        # 数据划分与 Loader
        tr_p, tr_s, val_p, val_s, te_p, te_s = DataSplitter.split_data(config, paths, scores, extra_info, split_seed)
        ds_cls = get_dataset_class(dataset_name)
        train_loader = DataLoader(ds_cls(tr_p, tr_s, sal_cache, is_train=True, **config), batch_size=config['BATCH_SIZE'], shuffle=True)
        val_ds = ds_cls(val_p, val_s, sal_cache, is_train=False, **config)
        test_ds = ds_cls(te_p, te_s, sal_cache, is_train=False, **config)

        model = SQT_HGR_Model(dropout_rate=config['DROPOUT_RATE']).to(config['DEVICE'])
        optimizer = optim.AdamW([
            {'params': model.backbone.parameters(), 'lr': config['LR_BACKBONE']},
            {'params': [p for n, p in model.named_parameters() if 'backbone' not in n], 'lr': config['LR_HEAD']}
        ], weight_decay=config['WEIGHT_DECAY'])
        
        criterion = TotalLoss(config['LAMBDA_MSE'], config['LAMBDA_RANK'])
        best_srcc = -1

        for ep in range(config['N_EPOCHS']):
            model.train()
            for img, sal, sc in train_loader:
                img, sal, sc = img.to(config['DEVICE']), sal.to(config['DEVICE']), sc.to(config['DEVICE'])
                optimizer.zero_grad()
                loss = criterion(model(img, sal), sc)
                loss.backward()
                optimizer.step()
            
            val_srcc, _, _, _, _ = validate_ten_crop_detailed(model, val_ds, config['DEVICE'])
            if val_srcc > best_srcc:
                best_srcc = val_srcc
                torch.save(model.state_dict(), 'best.pth')

        # 测试阶段
        model.load_state_dict(torch.load('best.pth'))
        srcc, plcc, preds, gts, _ = validate_ten_crop_detailed(model, test_ds, config['DEVICE'])
        all_srcc.append(srcc); all_plcc.append(plcc)
        plot_scatter(preds, gts, dataset_name, i, config['OUTPUT_DIR'])

    # 打印最终统计
    m_s, l_s, u_s = calculate_confidence_interval(all_srcc)
    print(f"Final SRCC: {m_s:.4f} ± {u_s-m_s:.4f}")

if __name__ == "__main__":
    run_experiment('CLIVE')
