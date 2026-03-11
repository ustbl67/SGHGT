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
    """统一的数据集配置类"""

    @staticmethod
    def get_config(dataset_name):
        # 针对单阶段训练的默认参数
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
                            range(1, 7)],  # 修改为IS目录结构
                'OUTPUT_DIR': "./cid2013_results_hgr_final",
                'parser': 'parse_cid2013_v2',
                'split_method': 'by_folder',  # 修改为按文件夹划分
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
            # --- START MODIFICATION for KADID (Two Stage) ---
            'KADID': {
                'BASE_DIR': r"C:\Users\Administrator\Desktop\IQA DATA\kadid10k",
                'MOS_FILE_PATH': os.path.join(r"C:\Users\Administrator\Desktop\IQA DATA\kadid10k", "exported_data.txt"),
                'IMAGES_DIR': os.path.join(r"C:\Users\Administrator\Desktop\IQA DATA\kadid10k", "images"),
                'OUTPUT_DIR': "./kadid10k_results_final",
                'parser': 'parse_kadid',
                'split_method': 'by_reference',
                'model_type': 'HGR',
                'training_stages': 'two_stage',  # 回退为两阶段
                # KADID特定参数
                'N_EPOCHS_WARMUP': 5,  # Warmup 轮数
                'N_EPOCHS_FINETUNE': 40,  # Finetune 轮数
                'N_EPOCHS': 40,  # 主循环的 Epochs 数 (与 Finetune 保持一致)
                'LR_HEAD': 3e-4,
                'LR_BACKBONE': 8e-6,
                'BATCH_SIZE': 8,
                'BASE_SIZE': 512,
                'INPUT_SIZE': 384,
                'IMG_SIZE': 384,
                'LAMBDA_MSE': 1.0,
                'LAMBDA_RANK': 1.0
            },
            # --- END MODIFICATION for KADID ---
            # --- START MODIFICATION for KONIQ (Two Stage) ---
            'KONIQ': {
                'BASE_DIR': r"C:\Users\Administrator\Desktop\IQA DATA\koniq",
                'IMAGES_DIR': os.path.join(r"C:\Users\Administrator\Desktop\IQA DATA\koniq", "koniq10k"),
                'MOS_FILE_PATH': os.path.join(r"C:\Users\Administrator\Desktop\IQA DATA\koniq", "image_mos_pairs.txt"),
                'OUTPUT_DIR': "./koniq_results_final",
                'parser': 'parse_koniq',
                'split_method': 'random',
                'model_type': 'HGR',
                'training_stages': 'two_stage',  # 回退为两阶段
                # KONIQ特定参数
                'N_EPOCHS_WARMUP': 5,  # Warmup 轮数
                'N_EPOCHS_FINETUNE': 50,  # Finetune 轮数
                'N_EPOCHS': 50,  # 主循环的 Epochs 数 (与 Finetune 保持一致)
                'LR_HEAD': 3e-4,
                'LR_BACKBONE': 8e-6,
                'BATCH_SIZE': 8,
                'BASE_SIZE': 512,
                'INPUT_SIZE': 384,
                'IMG_SIZE': 384,
                'LAMBDA_MSE': 1.0,
                'LAMBDA_RANK': 1.0
            }
            # --- END MODIFICATION for KONIQ ---
        }

        # 通用训练参数
        common_params = {
            'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
            'WEIGHT_DECAY': 1e-4,
            'DROPOUT_RATE': 0.3,
            'DEEPGAZE_PATH': './deepgaze_pytorch'
        }

        config = configs.get(dataset_name)
        if config:
            # 1. 如果是单阶段且未定义关键参数，则使用默认的 single_stage_params
            if config.get('training_stages') == 'single':
                # 对于 CID2013, TID2013, CLIVE，应用默认参数
                config.update(single_stage_params)

            # 2. 添加通用参数
            config.update(common_params)

            return config
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")


class BaseIQADataset(Dataset):
    """基础IQA数据集类"""

    def __init__(self, paths, scores, sal_cache, is_train=True, img_size=224, base_size=512, input_size=None):
        self.paths = paths
        self.scores = torch.tensor(scores, dtype=torch.float32)
        self.sal_cache = sal_cache
        self.is_train = is_train
        self.img_size = img_size
        self.base_size = base_size
        # input_size 应该用于裁剪，如果未提供则默认为 img_size
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

            # 1. 统一 Resize 到 BASE_SIZE
            img = img.resize((self.base_size, self.base_size), Image.BILINEAR)
            img_t = transforms.ToTensor()(img)

            if sal.ndim == 4:
                sal = sal.squeeze(0)

            # 2. Transform (严格对齐的裁剪)
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
    """CID2013数据集"""
    pass


class TID2013Dataset(BaseIQADataset):
    """TID2013数据集"""
    pass


class CLIVEDataset(BaseIQADataset):
    """CLIVE数据集"""
    pass


class KADIDDataset(BaseIQADataset):
    """KADID数据集"""
    pass


class KONIQDataset(BaseIQADataset):
    """KONIQ数据集"""
    pass


class DatasetParser:
    """数据集解析器"""

    @staticmethod
    def find_all_image_files(base_dirs):
        """递归查找所有图片文件"""
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
        """解析CID2013数据集（新版本：IS文件夹结构）"""
        if not os.path.exists(config['MOS_FILE_PATH']):
            print(f"MOS file not found: {config['MOS_FILE_PATH']}")
            return [], [], []

        # 读取MOS文件
        mos_dict = {}
        with open(config['MOS_FILE_PATH'], 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 2:
                    continue

                try:
                    source_id = parts[0]
                    score = float(parts[1])
                    mos_dict[source_id] = score
                except (ValueError, IndexError) as e:
                    print(f"Error parsing line: {line}, error: {e}")
                    continue

        print(f"Loaded {len(mos_dict)} MOS entries")

        # 查找所有图片文件
        all_image_paths = DatasetParser.find_all_image_files(config['IS_DIRS'])

        # 匹配图片路径和MOS分数
        paths, scores, is_folders = [], [], []

        unmatched_count = 0
        for img_path in all_image_paths:
            filename = os.path.basename(img_path)
            filename_no_ext = os.path.splitext(filename)[0]

            # 尝试多种匹配方式
            matched = False
            for source_id, score in mos_dict.items():
                # 1. 直接匹配文件名（无扩展名）
                if filename_no_ext == source_id:
                    paths.append(img_path)
                    scores.append(score)
                    # 提取IS文件夹信息
                    is_folder = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
                    is_folders.append(is_folder)
                    matched = True
                    break
                # 2. 检查source_id是否包含在文件名中
                elif source_id in filename:
                    paths.append(img_path)
                    scores.append(score)
                    is_folder = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
                    is_folders.append(is_folder)
                    matched = True
                    break

            if not matched:
                # 尝试更宽松的匹配：移除可能的后缀
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
        if unmatched_count > 0:
            print(f"Warning: {unmatched_count} images could not be matched with MOS entries")

        # 统计每个IS文件夹的图片数量
        is_counts = {}
        for is_folder in is_folders:
            is_counts[is_folder] = is_counts.get(is_folder, 0) + 1
        print(f"Images per IS folder: {is_counts}")

        return paths, scores, is_folders

    @staticmethod
    def parse_tid2013(config):
        """解析TID2013数据集"""
        if not os.path.exists(config['MOS_FILE_PATH']):
            print(f"MOS file not found: {config['MOS_FILE_PATH']}")
            return [], [], []

        paths, scores, refs = [], [], []
        with open(config['MOS_FILE_PATH'], 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 2:
                    continue

                try:
                    score = float(parts[0])
                    img_name = parts[1]
                    img_path = os.path.join(config['DISTORTED_IMAGES_DIR'], img_name)

                    if os.path.exists(img_path):
                        paths.append(img_path)
                        scores.append(score)
                        ref_name = img_name.split('_')[0]
                        refs.append(ref_name)
                    else:
                        print(f"Image not found: {img_path}")
                except (ValueError, IndexError) as e:
                    continue

        print(f"Successfully parsed {len(paths)} images from TID2013")
        return paths, scores, refs

    @staticmethod
    def parse_clive(config):
        """解析LIVE Challenge数据集"""
        if not os.path.exists(config['MOS_FILE_PATH']):
            print(f"MOS file not found: {config['MOS_FILE_PATH']}")
            return [], [], None

        # 读取所有图像文件
        image_extensions = {'.bmp', '.jpg', '.jpeg', '.png'}
        image_files = []

        for root, dirs, files in os.walk(config['IMAGES_DIR']):
            for file in files:
                if os.path.splitext(file)[1].lower() in image_extensions:
                    image_files.append(os.path.join(root, file))

        print(f"Found {len(image_files)} images in {config['IMAGES_DIR']}")

        # 读取MOS映射文件
        mos_dict = {}
        with open(config['MOS_FILE_PATH'], 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    image_name = parts[0]
                    mos_score = float(parts[1])
                    mos_dict[image_name] = mos_score

        print(f"Loaded {len(mos_dict)} MOS scores from mapping file")

        # 匹配图像路径和MOS分数
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
        """解析KADID-10k数据集"""
        if not os.path.exists(config['MOS_FILE_PATH']):
            print(f"MOS file not found: {config['MOS_FILE_PATH']}")
            return [], [], []

        # 读取DMOS文件
        paths, scores, refs = [], [], []

        try:
            # 读取导出的txt文件，格式为：dist_img ref_img dmos
            with open(config['MOS_FILE_PATH'], 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    # 解析格式: "dist_img ref_img dmos"
                    parts = line.split()
                    if len(parts) < 3:
                        continue

                    try:
                        dist_img_name = parts[0]
                        ref_img_name = parts[1]
                        dmos_score = float(parts[2])

                        # 构建失真图片完整路径
                        img_path = os.path.join(config['IMAGES_DIR'], dist_img_name)

                        if os.path.exists(img_path):
                            paths.append(img_path)
                            scores.append(dmos_score)
                            refs.append(ref_img_name)  # 参考图片名称，如 "I01"
                        else:
                            print(f"Image not found: {img_path}")

                    except (ValueError, IndexError) as e:
                        print(f"Error parsing line: {line}, error: {e}")
                        continue

        except Exception as e:
            print(f"Error reading MOS file: {e}")
            return [], [], []

        print(f"Successfully parsed {len(paths)} images from KADID-10k")
        return paths, scores, refs

    @staticmethod
    def parse_koniq(config):
        """解析KONIQ数据集"""
        if not os.path.exists(config['MOS_FILE_PATH']):
            print(f"MOS file not found: {config['MOS_FILE_PATH']}")
            return [], [], None

        # 读取MOS文件
        paths, scores = [], []
        with open(config['MOS_FILE_PATH'], 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # 解析格式: "图片名称 分数"
                parts = line.split()
                if len(parts) < 2:
                    continue

                try:
                    img_name = parts[0]
                    score = float(parts[1])

                    # 构建图片完整路径
                    img_path = os.path.join(config['IMAGES_DIR'], img_name)

                    if os.path.exists(img_path):
                        paths.append(img_path)
                        scores.append(score)
                    else:
                        print(f"Image not found: {img_path}")

                except (ValueError, IndexError) as e:
                    print(f"Error parsing line: {line}, error: {e}")
                    continue

        print(f"Successfully parsed {len(paths)} images from KONIQ")
        return paths, scores, None


class DataSplitter:
    """数据分割器"""

    @staticmethod
    def split_data(config, paths, scores, extra_info=None, random_seed=42): # <--- 改变 1: 接受 random_seed
        """根据配置分割数据"""
        split_method = config.get('split_method', 'random')

        if split_method == 'by_folder' and extra_info:  # CID2013 (按IS文件夹划分)
            is_folders = extra_info
            unique_is_folders = list(set(is_folders))
            unique_is_folders.sort()  # 排序确保一致性
            print(f"Available IS folders: {unique_is_folders}")
            print(f"Total IS folders: {len(unique_is_folders)}")

            # 修改为：3个训练集，1个验证集，2个测试集
            random.seed(random_seed) # <--- 改变 2: 使用传入的 random_seed

            # 随机打乱IS文件夹列表
            shuffled_is_folders = unique_is_folders.copy()
            random.shuffle(shuffled_is_folders)

            # 分配：3个训练，1个验证，2个测试
            train_is = shuffled_is_folders[:3]
            val_is = shuffled_is_folders[3:4]
            test_is = shuffled_is_folders[4:6]

            # 确保分配正确
            if len(train_is) != 3 or len(val_is) != 1 or len(test_is) != 2:
                raise ValueError(f"分配错误: 训练集={len(train_is)}, 验证集={len(val_is)}, 测试集={len(test_is)}")

            print(f"Training IS folders (3): {train_is}")
            print(f"Validation IS folders (1): {val_is}")
            print(f"Testing IS folders (2): {test_is}")

            train_paths, train_scores = [], []
            val_paths, val_scores = [], []
            test_paths, test_scores = [], []

            for path, score, is_folder in zip(paths, scores, is_folders):
                if is_folder in train_is:
                    train_paths.append(path)
                    train_scores.append(score)
                elif is_folder in val_is:
                    val_paths.append(path)
                    val_scores.append(score)
                elif is_folder in test_is:
                    test_paths.append(path)
                    test_scores.append(score)
                else:
                    print(f"Warning: Image {path} from IS folder {is_folder} not assigned to any split")

            print(
                f"Train: {len(train_paths)} images | Validation: {len(val_paths)} images | Test: {len(test_paths)} images")

            # 检查是否有数据丢失
            total_assigned = len(train_paths) + len(val_paths) + len(test_paths)
            if total_assigned != len(paths):
                print(f"Warning: Data loss detected! Total assigned: {total_assigned}, Original: {len(paths)}")

            return train_paths, train_scores, val_paths, val_scores, test_paths, test_scores

        elif split_method == 'by_reference' and extra_info:  # TID2013, KADID (按参考图片划分)
            refs = extra_info
            urefs = np.unique(refs)
            print(f"Total unique reference images: {len(urefs)}")

            # 按照6:2:2的比例划分参考图片
            # 首先划分训练集和临时集 (6:4)
            train_refs, temp_refs = train_test_split(
                urefs, test_size=0.4, random_state=random_seed # <--- 改变 3: 使用传入的 random_seed
            )

            # 再从临时集中划分验证集和测试集 (2:2)
            val_refs, test_refs = train_test_split(
                temp_refs, test_size=0.5, random_state=random_seed # <--- 改变 3: 使用传入的 random_seed
            )

            print(f"Training reference images (60%): {len(train_refs)}")
            print(f"Validation reference images (20%): {len(val_refs)}")
            print(f"Testing reference images (20%): {len(test_refs)}")

            # 根据参考图片归属分配图片
            train_idx = [i for i, r in enumerate(refs) if r in train_refs]
            val_idx = [i for i, r in enumerate(refs) if r in val_refs]
            test_idx = [i for i, r in enumerate(refs) if r in test_refs]

            train_paths = [paths[i] for i in train_idx]
            train_scores = [scores[i] for i in train_idx]
            val_paths = [paths[i] for i in val_idx]
            val_scores = [scores[i] for i in val_idx]
            test_paths = [paths[i] for i in test_idx]
            test_scores = [scores[i] for i in test_idx]

            print(f"Train: {len(train_paths)} images ({len(train_paths) / len(paths) * 100:.1f}%)")
            print(f"Validation: {len(val_paths)} images ({len(val_paths) / len(paths) * 100:.1f}%)")
            print(f"Test: {len(test_paths)} images ({len(test_paths) / len(paths) * 100:.1f}%)")

            return train_paths, train_scores, val_paths, val_scores, test_paths, test_scores

        else:  # CLIVE, KONIQ - random split
            print(f"Total images for random split: {len(paths)}")

            # 按照6:2:2的比例随机划分
            # 首先划分训练集和临时集 (6:4)
            train_idx, temp_idx = train_test_split(
                range(len(paths)), test_size=0.4, random_state=random_seed, shuffle=True # <--- 改变 4: 使用传入的 random_seed
            )

            # 再从临时集中划分验证集和测试集 (2:2)
            val_idx, test_idx = train_test_split(
                temp_idx, test_size=0.5, random_state=random_seed, shuffle=True # <--- 改变 4: 使用传入的 random_seed
            )

            train_paths = [paths[i] for i in train_idx]
            train_scores = [scores[i] for i in train_idx]
            val_paths = [paths[i] for i in val_idx]
            val_scores = [scores[i] for i in val_idx]
            test_paths = [paths[i] for i in test_idx]
            test_scores = [scores[i] for i in test_idx]

            print(f"Train: {len(train_paths)} images ({len(train_paths) / len(paths) * 100:.1f}%)")
            print(f"Validation: {len(val_paths)} images ({len(val_paths) / len(paths) * 100:.1f}%)")
            print(f"Test: {len(test_paths)} images ({len(test_paths) / len(paths) * 100:.1f}%)")

            return train_paths, train_scores, val_paths, val_scores, test_paths, test_scores


def get_dataset_class(dataset_name):
    """获取数据集类"""
    dataset_classes = {
        'CID2013': CID2013Dataset,
        'TID2013': TID2013Dataset,
        'CLIVE': CLIVEDataset,
        'KADID': KADIDDataset,
        'KONIQ': KONIQDataset
    }
    return dataset_classes.get(dataset_name, BaseIQADataset)
