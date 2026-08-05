"""
NEU-DET 数据集安装脚本
======================
功能：
1. 备份现有 ship 数据集
2. 解压 NEU-DET.zip
3. XML(VOC格式) → YOLO 格式转换
4. 划分 train/val/test
5. 生成 data.yaml 配置文件

NEU-DET 包含 6 类钢材表面缺陷：
  crazing(裂纹), inclusion(夹杂), patches(斑块),
  pitted_surface(麻点), rolled-in_scale(轧制氧化皮), scratches(划痕)
"""

import os
import sys
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime

# ===================== 配置 =====================
BASE_DIR = Path(__file__).parent  # dataset/
ZIP_FILE = BASE_DIR / "NEU-DET.zip"
TEMP_EXTRACT = BASE_DIR / ".neu_det_temp"

# NEU-DET 类别（6类钢材表面缺陷）
CLASSES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

# 数据集划分比例
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ===================== 备份现有数据 =====================
def backup_existing():
    """备份现有的 ship 数据集"""
    images_dir = BASE_DIR / "images"
    labels_dir = BASE_DIR / "labels"
    data_yaml = BASE_DIR / "data.yaml"

    if not any(images_dir.iterdir()) and not any(labels_dir.iterdir()):
        print("[SKIP] images/ 和 labels/ 目录为空，跳过备份")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = BASE_DIR / f"backup_ship_{timestamp}"
    backup_dir.mkdir(exist_ok=True)

    for src_dir in [images_dir, labels_dir]:
        if src_dir.exists():
            dst = backup_dir / src_dir.name
            shutil.move(str(src_dir), str(dst))
            print(f"[BACKUP] {src_dir.name}/ → {backup_dir.name}/")

    if data_yaml.exists():
        shutil.copy(str(data_yaml), str(backup_dir / "data.yaml"))
        print(f"[BACKUP] data.yaml → {backup_dir.name}/")

    print(f"[BACKUP] 原 ship 数据集已备份到 {backup_dir.name}/\n")


# ===================== 解压 =====================
def extract_dataset():
    """解压 NEU-DET.zip"""
    if TEMP_EXTRACT.exists():
        shutil.rmtree(str(TEMP_EXTRACT))
    TEMP_EXTRACT.mkdir(exist_ok=True)

    print(f"[EXTRACT] 正在解压 {ZIP_FILE.name}...")
    shutil.unpack_archive(str(ZIP_FILE), str(TEMP_EXTRACT), 'zip')
    print(f"[EXTRACT] 解压完成 → {TEMP_EXTRACT}/NEU-DET/\n")


# ===================== XML → YOLO 转换 =====================
def convert_xml_to_yolo(xml_path, img_w=200, img_h=200):
    """
    将 VOC 格式 XML 转换为 YOLO 格式标注
    返回: list of "class_id cx cy w h"  (归一化坐标)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 尝试从 XML 获取真实宽高
    size = root.find('size')
    if size is not None:
        w = float(size.find('width').text)
        h = float(size.find('height').text)
    else:
        w, h = img_w, img_h

    annotations = []
    for obj in root.iter('object'):
        cls_name = obj.find('name').text.strip()
        if cls_name not in CLASSES:
            print(f"  [WARN] 未知类别 '{cls_name}' in {xml_path}")
            continue
        cls_id = CLASSES.index(cls_name)

        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)

        # YOLO 格式: <class> <x_center> <y_center> <width> <height> (归一化)
        cx = ((xmin + xmax) / 2.0) / w
        cy = ((ymin + ymax) / 2.0) / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h

        # 裁剪到 [0, 1]
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        bw = max(0.0, min(1.0, bw))
        bh = max(0.0, min(1.0, bh))

        annotations.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    return annotations


# ===================== 收集所有数据 =====================
def collect_and_convert():
    """收集所有 NEU-DET 图像和标注，转换为 YOLO 格式"""
    neu_det_dir = TEMP_EXTRACT / "NEU-DET"
    data_pairs = []  # [(image_path, yolo_label_text)]

    for subset in ['train', 'validation']:
        img_base = neu_det_dir / subset / "images"
        ann_dir = neu_det_dir / subset / "annotations"

        if not ann_dir.exists():
            print(f"[WARN] 标注目录不存在: {ann_dir}")
            continue

        xml_files = list(ann_dir.glob("*.xml"))
        print(f"[CONVERT] {subset} 集: {len(xml_files)} 个标注文件")

        for xml_path in xml_files:
            # XML 文件名如 crazing_1.xml
            base_name = xml_path.stem  # e.g., crazing_1

            # 图像在 <class>/<base_name>.jpg 下
            # base_name 格式为 <class>_<number>，例如 crazing_1
            # 从文件名提取类别（去掉尾部数字和下划线）
            # e.g., crazing_1 → crazing, patches_23 → patches
            parts = base_name.rsplit('_', 1)
            if len(parts) == 2:
                class_name = parts[0]
            else:
                class_name = base_name

            # 查找图像文件
            img_path = img_base / class_name / f"{base_name}.jpg"
            if not img_path.exists():
                # 尝试其他扩展名
                for ext in ['.jpg', '.png', '.bmp', '.JPG', '.PNG']:
                    alt_path = img_base / class_name / f"{base_name}{ext}"
                    if alt_path.exists():
                        img_path = alt_path
                        break
                else:
                    print(f"  [WARN] 找不到图像: {base_name}.jpg in {img_base / class_name}")
                    continue

            # 转换标注
            yolo_anns = convert_xml_to_yolo(xml_path)
            if not yolo_anns:
                continue

            data_pairs.append((img_path, '\n'.join(yolo_anns)))

    print(f"[CONVERT] 共计 {len(data_pairs)} 个有效图像-标注对\n")
    return data_pairs


# ===================== 划分数据集 =====================
def split_and_save(data_pairs):
    """划分 train/val/test 并保存到 images/ 和 labels/ 目录"""
    import random
    random.seed(42)
    random.shuffle(data_pairs)

    n = len(data_pairs)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_data = data_pairs[:n_train]
    val_data = data_pairs[n_train:n_train + n_val]
    test_data = data_pairs[n_train + n_val:]

    print(f"[SPLIT] Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # 创建目录结构
    subsets = {
        'train': train_data,
        'val': val_data,
        'test': test_data,
    }

    for subset_name, pairs in subsets.items():
        img_dir = BASE_DIR / "images" / subset_name
        lbl_dir = BASE_DIR / "labels" / subset_name

        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[SAVE] {subset_name}/ ({len(pairs)} 个)")

        for img_path, yolo_text in pairs:
            # 使用原始文件名（如 crazing_1.jpg）
            filename = img_path.stem

            # 复制图像到目标
            dst_img = img_dir / f"{filename}.jpg"
            shutil.copy2(str(img_path), str(dst_img))

            # 写入 YOLO 标注
            dst_lbl = lbl_dir / f"{filename}.txt"
            with open(dst_lbl, 'w', encoding='utf-8') as f:
                f.write(yolo_text + '\n')

    print("\n[SAVE] 所有文件保存完毕\n")


# ===================== 生成 data.yaml =====================
def create_yaml():
    """生成 YOLO 格式的 data.yaml"""
    yaml_content = f"""# NEU-DET 钢材表面缺陷检测数据集
# 自动生成于: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

# 数据集路径 (相对于 train.py 运行目录)
path: .
train: ./dataset/images/train
val: ./dataset/images/val
test: ./dataset/images/test

# 类别数量
nc: {len(CLASSES)}

# 类别名称
names: {CLASSES}
"""
    yaml_path = BASE_DIR / "data.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    print(f"[YAML] data.yaml 已生成:")
    print(yaml_content)


# ===================== 清理 =====================
def cleanup():
    """删除临时解压文件"""
    if TEMP_EXTRACT.exists():
        shutil.rmtree(str(TEMP_EXTRACT))
        print(f"[CLEAN] 已删除临时目录 {TEMP_EXTRACT.name}/")


# ===================== 主流程 =====================
def main():
    print("=" * 60)
    print("  NEU-DET 数据集安装脚本")
    print("  钢材表面缺陷检测 - 6 类 1800 张图像")
    print("=" * 60)
    print()

    # 检查 zip 文件
    if not ZIP_FILE.exists():
        print(f"[ERROR] 找不到 {ZIP_FILE}")
        print("请先下载 NEU-DET.zip 到 dataset/ 目录")
        sys.exit(1)

    # Step 1: 备份现有数据
    backup_existing()

    # Step 2: 解压
    extract_dataset()

    # Step 3: 转换
    data_pairs = collect_and_convert()

    if len(data_pairs) == 0:
        print("[ERROR] 没有找到有效的图像-标注对")
        cleanup()
        sys.exit(1)

    # Step 4: 划分并保存
    split_and_save(data_pairs)

    # Step 5: 生成 yaml
    create_yaml()

    # Step 6: 清理
    cleanup()

    print("=" * 60)
    print("  安装完成!")
    print(f"  训练集: {int(len(data_pairs) * TRAIN_RATIO)} 张")
    print(f"  验证集: {int(len(data_pairs) * VAL_RATIO)} 张")
    print(f"  测试集: {len(data_pairs) - int(len(data_pairs) * TRAIN_RATIO) - int(len(data_pairs) * VAL_RATIO)} 张")
    print(f"  类别数: {len(CLASSES)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
