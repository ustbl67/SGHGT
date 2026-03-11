import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import torchvision.transforms.functional as T_F
import torchvision.transforms as transforms


class DatasetConfig:
    """Unified configuration class for IQA datasets."""

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
                'MOS_FILE_PATH': os.path.join(r"C:\Users\Administrator\Desktop\IQA DATA\CID2013",
                                              "source_id_iq_score_pairs.txt"),
                'IS_DIRS': [os.path.join(r"C:\Users\Administrator\Desktop\IQA DATA\CID2013", f"IS{i}") for i in
                            range(1, 7)],  # IS directory structure
                'OUTPUT_DIR': "./cid2013_results_hgr_final",
                'parser': 'parse_cid2013_v2',
                'split_method': 'by_folder',  # Split by folder for CID2013
                'model_type': 'HGR',
                'training_stages': 'single'
            },
            'TID2013': {
                'BASE_DIR': r"C:\Users\Administrator\Desktop\IQA DATA\tid2013",
                'MOS_FILE_PATH': os.path.join(r"C:\Users\Administrator\Desktop\IQA DATA\tid2013", "mos_with_names.txt"),
                'DISTORTED_IMAGES_DIR': os.path.join(r"C:\Users\Administrator\Desktop\IQA DATA\tid2013",
                                                     "distorted_images"),
                'REFERENCE_IMAGES_DIR': os.path.join(r"C:\Users\Administrator\Desktop\IQA DATA\tid2013",
                                                     "reference_images"),
                'OUTPUT_DIR': "./tid2013_results_hgr_final",
                'parser': 'parse_tid2013',
                'split_method': 'by_reference',
                'model_type': 'HGR',
                'training_stages': 'single'
            },
            'CLIVE': {
                'BASE_DIR': r"C:\Users\Administrator\Desktop\IQA DATA\CLIVE",
                'IMAGES_DIR': os.path.join(r"C:\Users\Administrator\Desktop\IQA DATA\CLIVE", "Images"),
                'MOS_FILE_PATH': os.path.join(r"C:\Users\Administrator\Desktop\IQA DATA\CLIVE", "Data",
                                              "image_mos_mapping.txt"),
                'OUTPUT_DIR': "./clive_results_hgr_final",
                'parser': 'parse_clive',
                'split_method': 'random',
                'model_type': 'HGR',
                'training_stages': 'single'
            },
            'KADID': {
                'BASE_DIR': r"C:\Users\Administrator\Desktop\IQA DATA\kadid10k",
                'MOS_FILE_PATH': os.path.join(r"C:\Users\Administrator\Desktop\IQA DATA\kadid10k", "exported_data.txt"),
                'IMAGES_DIR': os.path.join(r"C:\Users\Administrator\Desktop\IQA DATA\kadid10k", "images"),
                'OUTPUT_DIR': "./kadid10k_results_final",
                'parser': 'parse_kadid',
                'split_method': 'by_reference',
                'model_type': 'HGR',
                'training_stages': 'two_stage',
                # KADID specific parameters
                'N_EPOCHS_WARMUP': 5,
                'N_EPOCHS_FINETUNE': 40,
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

        # Common training parameters
        common_params = {
            'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
            'WEIGHT_DECAY': 1e-4,
            'DROPOUT_RATE': 0.3,
            'DEEPGAZE_PATH': './deepgaze_pytorch'
        }

        config = configs.get(dataset_name)
        if config:
            # 1. Apply default single_stage_params if training_stages is 'single'
            if config.get('training_stages') == 'single':
                config.update(single_stage_params)

            # 2. Add common parameters
            config.update(common_params)

            return config
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")


class BaseIQADataset(Dataset):
    """Base class for Image Quality Assessment datasets."""

    def __init__(self, paths, scores, sal_cache, is_train=True, img_size=224, base_size=512, input_size=None):
        self.paths = paths
        self.scores = torch.tensor(scores, dtype=torch.float32)
        self.sal_cache = sal_cache
        self.is_train = is_train
        self.img_size = img_size
        self.base_size = base_size
        # Use input_size for cropping; defaults to img_size if not provided
        self.input_size = input_size if input_size is not None else img_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i]).convert('RGB')
            sal_path = self.sal_cache[self.paths[i]]

            try:
                sal = torch.load(sal_path)
            except:
                sal = torch.zeros(1, self.base_size, self.base_size)

            # 1. Uniformly resize to BASE_SIZE
            img = img.resize((self.base_size, self.base_size), Image.BILINEAR)
            img_t = transforms.ToTensor()(img)

            if sal.ndim == 4:
                sal = sal.squeeze(0)

            # 2. Transform (strictly aligned cropping for image and saliency map)
            crop_size = self.input_size
            if self.is_train:
                i_crop, j_crop, h, w = transforms.RandomCrop.get_params(
                    img_t, output_size=(crop_size, crop_size)
                )
                img_out = T_F.crop(img_t, i_crop, j_crop, h, w)
                sal_out = T_F.crop(sal, i_crop, j_crop, h, w)

                if random.random() > 0.5:
                    img_out = T_F.hflip(img_out)
                    sal_out = T_F.hflip(sal_out)

                if random.random() > 0.7:
                    angle = random.randint(-10, 10)
                    img_out = T_F.rotate(img_out, angle)
                    sal_out = T_F.rotate(sal_out, angle)
            else:
                img_out = T_F.center_crop(img_t, (crop_size, crop_size))
                sal_out = T_F.center_crop(sal, (crop_size, crop_size))

            # 3. Normalize
            norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            img_out = norm(img_out)

            return img_out, sal_out, self.scores[i]
        except Exception as e:
            print(f"Error loading image {self.paths[i]}: {e}")
            dummy_img = torch.zeros(3, crop_size, crop_size)
            dummy_sal = torch.zeros(1, crop_size, crop_size)
            return dummy_img, dummy_sal, torch.tensor(0.0)


class CID2013Dataset(BaseIQADataset):
    """CID2013 Dataset implementation."""
    pass


class TID2013Dataset(BaseIQADataset):
    """TID2013 Dataset implementation."""
    pass


class CLIVEDataset(BaseIQADataset):
    """CLIVE Dataset implementation."""
    pass


class KADIDDataset(BaseIQADataset):
    """KADID Dataset implementation."""
    pass


class DatasetParser:
    """Utility class for parsing various IQA datasets."""

    @staticmethod
    def find_all_image_files(base_dirs):
        """Recursively find all image files in specified directories."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        all_images = []

        for base_dir in base_dirs:
            if not os.path.exists(base_dir):
                print(f"Warning: Directory {base_dir} does not exist")
                continue

            print(f"Searching for images in {base_dir}...")
            for root, dirs, files in os.walk(base_dir):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        img_path = os.path.join(root, file)
                        all_images.append(img_path)

        print(f"Found {len(all_images)} images in total")
        return all_images

    @staticmethod
    def parse_cid2013_v2(config):
        """Parse CID2013 dataset using the IS folder structure."""
        if not os.path.exists(config['MOS_FILE_PATH']):
            print(f"MOS file not found: {config['MOS_FILE_PATH']}")
            return [], [], []

        mos_dict = {}
        with open(config['MOS_FILE_PATH'], 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split()
                if len(parts) < 2: continue
                try:
                    source_id = parts[0]
                    score = float(parts[1])
                    mos_dict[source_id] = score
                except (ValueError, IndexError) as e:
                    continue

        print(f"Loaded {len(mos_dict)} MOS entries")
        all_image_paths = DatasetParser.find_all_image_files(config['IS_DIRS'])

        paths, scores, is_folders = [], [], []
        unmatched_count = 0
        for img_path in all_image_paths:
            filename = os.path.basename(img_path)
            filename_no_ext = os.path.splitext(filename)[0]
            matched = False
            for source_id, score in mos_dict.items():
                if filename_no_ext == source_id or source_id in filename:
                    paths.append(img_path)
                    scores.append(score)
                    is_folder = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
                    is_folders.append(is_folder)
                    matched = True
                    break
            if not matched:
                base_name_parts = filename_no_ext.split('_')[0]
                if base_name_parts in mos_dict:
                    paths.append(img_path)
                    scores.append(mos_dict[base_name_parts])
                    is_folder = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
                    is_folders.append(is_folder)
                    matched = True
                else:
                    unmatched_count += 1

        print(f"Successfully matched {len(paths)} images with MOS scores")
        return paths, scores, is_folders

    @staticmethod
    def parse_tid2013(config):
        """Parse TID2013 dataset."""
        if not os.path.exists(config['MOS_FILE_PATH']):
            print(f"MOS file not found: {config['MOS_FILE_PATH']}")
            return [], [], []

        paths, scores, refs = [], [], []
        with open(config['MOS_FILE_PATH'], 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split()
                if len(parts) < 2: continue
                try:
                    score = float(parts[0])
                    img_name = parts[1]
                    img_path = os.path.join(config['DISTORTED_IMAGES_DIR'], img_name)
                    if os.path.exists(img_path):
                        paths.append(img_path)
                        scores.append(score)
                        ref_name = img_name.split('_')[0]
                        refs.append(ref_name)
                except (ValueError, IndexError):
                    continue

        print(f"Successfully parsed {len(paths)} images from TID2013")
        return paths, scores, refs

    @staticmethod
    def parse_clive(config):
        """Parse LIVE Challenge (CLIVE) dataset."""
        if not os.path.exists(config['MOS_FILE_PATH']):
            print(f"MOS file not found: {config['MOS_FILE_PATH']}")
            return [], [], None

        image_extensions = {'.bmp', '.jpg', '.jpeg', '.png'}
        image_files = []
        for root, _, files in os.walk(config['IMAGES_DIR']):
            for file in files:
                if os.path.splitext(file)[1].lower() in image_extensions:
                    image_files.append(os.path.join(root, file))

        mos_dict = {}
        with open(config['MOS_FILE_PATH'], 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    mos_dict[parts[0]] = float(parts[1])

        paths, scores = [], []
        for image_path in image_files:
            image_name = os.path.basename(image_path)
            if image_name in mos_dict:
                paths.append(image_path)
                scores.append(mos_dict[image_name])

        print(f"Successfully matched {len(paths)} images with MOS scores")
        return paths, scores, None

    @staticmethod
    def parse_kadid(config):
        """Parse KADID-10k dataset."""
        if not os.path.exists(config['MOS_FILE_PATH']):
            print(f"MOS file not found: {config['MOS_FILE_PATH']}")
            return [], [], []

        paths, scores, refs = [], [], []
        try:
            with open(config['MOS_FILE_PATH'], 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    parts = line.split()
                    if len(parts) < 3: continue
                    try:
                        dist_img_name, ref_img_name, dmos_score = parts[0], parts[1], float(parts[2])
                        img_path = os.path.join(config['IMAGES_DIR'], dist_img_name)
                        if os.path.exists(img_path):
                            paths.append(img_path)
                            scores.append(dmos_score)
                            refs.append(ref_img_name)
                    except (ValueError, IndexError):
                        continue
        except Exception as e:
            print(f"Error reading MOS file: {e}")
            return [], [], []

        print(f"Successfully parsed {len(paths)} images from KADID-10k")
        return paths, scores, refs


class DataSplitter:
    """Handles dataset splitting into train, validation, and test sets."""

    @staticmethod
    def split_data(config, paths, scores, extra_info=None, random_seed=42):
        """Split data based on the configured method (by folder, by reference, or random)."""
        split_method = config.get('split_method', 'random')

        if split_method == 'by_folder' and extra_info:  # CID2013: Split by IS folders
            is_folders = extra_info
            unique_is_folders = sorted(list(set(is_folders)))
            random.seed(random_seed)
            shuffled_is_folders = unique_is_folders.copy()
            random.shuffle(shuffled_is_folders)

            # Allocation: 3 Train, 1 Val, 2 Test
            train_is, val_is, test_is = shuffled_is_folders[:3], shuffled_is_folders[3:4], shuffled_is_folders[4:6]

            train_paths, train_scores = [], []
            val_paths, val_scores = [], []
            test_paths, test_scores = [], []

            for path, score, is_folder in zip(paths, scores, is_folders):
                if is_folder in train_is:
                    train_paths.append(path); train_scores.append(score)
                elif is_folder in val_is:
                    val_paths.append(path); val_scores.append(score)
                elif is_folder in test_is:
                    test_paths.append(path); test_scores.append(score)

            return train_paths, train_scores, val_paths, val_scores, test_paths, test_scores

        elif split_method == 'by_reference' and extra_info:  # TID2013, KADID: Split by reference images
            refs = extra_info
            urefs = np.unique(refs)
            train_refs, temp_refs = train_test_split(urefs, test_size=0.4, random_state=random_seed)
            val_refs, test_refs = train_test_split(temp_refs, test_size=0.5, random_state=random_seed)

            train_idx = [i for i, r in enumerate(refs) if r in train_refs]
            val_idx = [i for i, r in enumerate(refs) if r in val_refs]
            test_idx = [i for i, r in enumerate(refs) if r in test_refs]

            return ([paths[i] for i in train_idx], [scores[i] for i in train_idx],
                    [paths[i] for i in val_idx], [scores[i] for i in val_idx],
                    [paths[i] for i in test_idx], [scores[i] for i in test_idx])

        else:  # CLIVE: Random split
            train_idx, temp_idx = train_test_split(range(len(paths)), test_size=0.4, random_state=random_seed, shuffle=True)
            val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=random_seed, shuffle=True)

            return ([paths[i] for i in train_idx], [scores[i] for i in train_idx],
                    [paths[i] for i in val_idx], [scores[i] for i in val_idx],
                    [paths[i] for i in test_idx], [scores[i] for i in test_idx])


def get_dataset_class(dataset_name):
    """Retrieve the corresponding dataset class based on name."""
    dataset_classes = {
        'CID2013': CID2013Dataset,
        'TID2013': TID2013Dataset,
        'CLIVE': CLIVEDataset,
        'KADID': KADIDDataset
    }
    return dataset_classes.get(dataset_name, BaseIQADataset)
