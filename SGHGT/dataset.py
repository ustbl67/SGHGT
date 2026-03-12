import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as T_F
from sklearn.model_selection import train_test_split

class DatasetConfig:
    """Unified configuration class for various IQA datasets."""
    @staticmethod
    def get_config(dataset_name):
        # Default parameters for single-stage training
        single_stage_params = {
            'N_EPOCHS': 40,
            'BATCH_SIZE': 16,
            'IMG_SIZE': 224,
            'LR_HEAD': 3e-4,
            'LR_BACKBONE': 2e-5,
            'BASE_SIZE': 512,
            'LAMBDA_MSE': 1.0,
            'LAMBDA_RANK': 1.0
        }

        configs = {
            'CID2013': {
                'BASE_DIR': r"C:\Users\Administrator\Desktop\IQA DATA\CID2013",
                'MOS_FILE_PATH': os.path.join(r"C:\Users\Administrator\Desktop\IQA DATA\CID2013", "source_id_iq_score_pairs.txt"),
                'IS_DIRS': [os.path.join(r"C:\Users\Administrator\Desktop\IQA DATA\CID2013", f"IS{i}") for i in range(1, 7)],
                'OUTPUT_DIR': "./cid2013_results",
                'parser': 'parse_cid2013_v2',
                'split_method': 'by_folder',
                'model_type': 'HGR',
                'training_stages': 'single'
            },
            'TID2013': {
                'BASE_DIR': r"C:\Users\Administrator\Desktop\IQA DATA\tid2013",
                'MOS_FILE_PATH': os.path.join(r"C:\Users\Administrator\Desktop\IQA DATA\tid2013", "mos_with_names.txt"),
                'DISTORTED_IMAGES_DIR': os.path.join(r"C:\Users\Administrator\Desktop\IQA DATA\tid2013", "distorted_images"),
                'OUTPUT_DIR': "./tid2013_results",
                'parser': 'parse_tid2013',
                'split_method': 'by_reference',
                'model_type': 'HGR',
                'training_stages': 'single'
            },
            'CLIVE': {
                'BASE_DIR': r"C:\Users\Administrator\Desktop\IQA DATA\CLIVE",
                'IMAGES_DIR': os.path.join(r"C:\Users\Administrator\Desktop\IQA DATA\CLIVE", "Images"),
                'MOS_FILE_PATH': os.path.join(r"C:\Users\Administrator\Desktop\IQA DATA\CLIVE", "Data", "image_mos_mapping.txt"),
                'OUTPUT_DIR': "./clive_results",
                'parser': 'parse_clive',
                'split_method': 'random',
                'model_type': 'HGR',
                'training_stages': 'single'
            },
            'KADID': {
                'BASE_DIR': r"C:\Users\Administrator\Desktop\IQA DATA\kadid10k",
                'MOS_FILE_PATH': os.path.join(r"C:\Users\Administrator\Desktop\IQA DATA\kadid10k", "exported_data.txt"),
                'IMAGES_DIR': os.path.join(r"C:\Users\Administrator\Desktop\IQA DATA\kadid10k", "images"),
                'OUTPUT_DIR': "./kadid10k_results",
                'parser': 'parse_kadid',
                'split_method': 'by_reference',
                'model_type': 'HGR',
                'training_stages': 'two_stage',
                'N_EPOCHS': 40,
                'LR_HEAD': 3e-4,
                'LR_BACKBONE': 8e-6,
                'BATCH_SIZE': 8,
                'BASE_SIZE': 512,
                'INPUT_SIZE': 384,
                'IMG_SIZE': 384,
                'LAMBDA_MSE': 1.0,
                'LAMBDA_RANK': 1.0
            }
        }

        common_params = {
            'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
            'WEIGHT_DECAY': 1e-4,
            'DROPOUT_RATE': 0.3
        }

        config = configs.get(dataset_name)
        if config:
            if config.get('training_stages') == 'single':
                config.update(single_stage_params)
            config.update(common_params)
            return config
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

class BaseIQADataset(Dataset):
    """General IQA dataset class handling images and pre-computed saliency maps."""
    def __init__(self, paths, scores, sal_cache, is_train=True, img_size=224, base_size=512, input_size=None):
        self.paths = paths
        self.scores = torch.tensor(scores, dtype=torch.float32)
        self.sal_cache = sal_cache
        self.is_train = is_train
        self.img_size = img_size
        self.base_size = base_size
        self.input_size = input_size if input_size is not None else img_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert('RGB').resize((self.base_size, self.base_size), Image.BILINEAR)
            img_t = transforms.ToTensor()(img)
            
            sal_path = self.sal_cache[self.paths[idx]]
            sal = torch.load(sal_path)
            if sal.ndim == 4: sal = sal.squeeze(0)

            crop_size = self.input_size
            if self.is_train:
                i, j, h, w = transforms.RandomCrop.get_params(img_t, (crop_size, crop_size))
                img_out = T_F.crop(img_t, i, j, h, w)
                sal_out = T_F.crop(sal, i, j, h, w)
                if random.random() > 0.5:
                    img_out = T_F.hflip(img_out)
                    sal_out = T_F.hflip(sal_out)
            else:
                img_out = T_F.center_crop(img_t, (crop_size, crop_size))
                sal_out = T_F.center_crop(sal, (crop_size, crop_size))

            norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            return norm(img_out), sal_out, self.scores[idx]
        except Exception as e:
            return torch.zeros(3, self.input_size, self.input_size), torch.zeros(1, self.input_size, self.input_size), torch.tensor(0.0)

# Dataset Subclasses
class CID2013Dataset(BaseIQADataset): pass
class TID2013Dataset(BaseIQADataset): pass
class CLIVEDataset(BaseIQADataset): pass
class KADIDDataset(BaseIQADataset): pass

class DatasetParser:
    """Helper to find files and parse MOS text files."""
    @staticmethod
    def find_all_image_files(base_dirs):
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        all_images = []
        for base_dir in base_dirs:
            if not os.path.exists(base_dir): continue
            for root, _, files in os.walk(base_dir):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        all_images.append(os.path.join(root, file))
        return all_images

    @staticmethod
    def parse_cid2013_v2(config):
        mos_dict = {}
        with open(config['MOS_FILE_PATH'], 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2: mos_dict[parts[0]] = float(parts[1])
        all_image_paths = DatasetParser.find_all_image_files(config['IS_DIRS'])
        paths, scores, folders = [], [], []
        for p in all_image_paths:
            fname = os.path.splitext(os.path.basename(p))[0]
            if fname in mos_dict:
                paths.append(p); scores.append(mos_dict[fname])
                folders.append(os.path.basename(os.path.dirname(os.path.dirname(p))))
        return paths, scores, folders

    @staticmethod
    def parse_tid2013(config):
        paths, scores, refs = [], [], []
        with open(config['MOS_FILE_PATH'], 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    score, img_name = float(parts[0]), parts[1]
                    img_path = os.path.join(config['DISTORTED_IMAGES_DIR'], img_name)
                    if os.path.exists(img_path):
                        paths.append(img_path); scores.append(score)
                        refs.append(img_name.split('_')[0])
        return paths, scores, refs

    @staticmethod
    def parse_clive(config):
        mos_dict = {}
        with open(config['MOS_FILE_PATH'], 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2: mos_dict[parts[0]] = float(parts[1])
        image_files = DatasetParser.find_all_image_files([config['IMAGES_DIR']])
        paths, scores = [], []
        for p in image_files:
            name = os.path.basename(p)
            if name in mos_dict:
                paths.append(p); scores.append(mos_dict[name])
        return paths, scores, None

    @staticmethod
    def parse_kadid(config):
        paths, scores, refs = [], [], []
        with open(config['MOS_FILE_PATH'], 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    img_path = os.path.join(config['IMAGES_DIR'], parts[0])
                    if os.path.exists(img_path):
                        paths.append(img_path); scores.append(float(parts[2])); refs.append(parts[1])
        return paths, scores, refs

class DataSplitter:
    """Splits data based on folder, reference image, or random indexing."""
    @staticmethod
    def split_data(config, paths, scores, extra_info=None, random_seed=42):
        method = config.get('split_method', 'random')
        if method == 'by_folder' and extra_info:
            folders = list(set(extra_info))
            folders.sort()
            random.seed(random_seed)
            random.shuffle(folders)
            train_f, val_f, test_f = folders[:3], folders[3:4], folders[4:6]
            t_p, t_s, v_p, v_s, ts_p, ts_s = [], [], [], [], [], []
            for p, s, f in zip(paths, scores, extra_info):
                if f in train_f: t_p.append(p); t_s.append(s)
                elif f in val_f: v_p.append(p); v_s.append(s)
                elif f in test_f: ts_p.append(p); ts_s.append(s)
            return t_p, t_s, v_p, v_s, ts_p, ts_s
        elif method == 'by_reference' and extra_info:
            urefs = np.unique(extra_info)
            tr_r, tmp_r = train_test_split(urefs, test_size=0.4, random_state=random_seed)
            v_r, ts_r = train_test_split(tmp_r, test_size=0.5, random_state=random_seed)
            t_p = [paths[i] for i, r in enumerate(extra_info) if r in tr_r]
            t_s = [scores[i] for i, r in enumerate(extra_info) if r in tr_r]
            v_p = [paths[i] for i, r in enumerate(extra_info) if r in v_r]
            v_s = [scores[i] for i, r in enumerate(extra_info) if r in v_r]
            ts_p = [paths[i] for i, r in enumerate(extra_info) if r in ts_r]
            ts_s = [scores[i] for i, r in enumerate(extra_info) if r in ts_r]
            return t_p, t_s, v_p, v_s, ts_p, ts_s
        else:
            idx = list(range(len(paths)))
            tr_i, tmp_i = train_test_split(idx, test_size=0.4, random_state=random_seed)
            v_i, ts_i = train_test_split(tmp_i, test_size=0.5, random_state=random_seed)
            return [paths[i] for i in tr_i], [scores[i] for i in tr_i], \
                   [paths[i] for i in v_i], [scores[i] for i in v_i], \
                   [paths[i] for i in ts_i], [scores[i] for i in ts_i]

def get_dataset_class(name):
    cls_map = {'CID2013': CID2013Dataset, 'TID2013': TID2013Dataset, 'CLIVE': CLIVEDataset, 'KADID': KADIDDataset}
    return cls_map.get(name, BaseIQADataset)
