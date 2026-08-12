"""
GC10-DET 数据集转换脚本
将 VOC 格式（XML 标注 + 按类别分文件夹的图像）转为 YOLO 格式
同时进行 train/val/test 数据集划分

GC10-DET 类别:
  0: chongkong   (冲孔 - punching hole)
  1: hanfeng     (焊缝 - weld line)
  2: yueyawan    (月牙弯 - crescent gap)
  3: shuiban     (水斑 - water spot)
  4: youban      (油斑 - oil spot)
  5: siban       (丝斑 - silk spot)
  6: yiwu        (异物 - inclusion)
  7: yahen       (压痕 - rolled pit)
  8: zhehen      (折痕 - crease)
  9: yaozhe      (腰折 - waist folding)
"""

import os
import xml.etree.ElementTree as ET
import shutil
from pathlib import Path
import random

random.seed(42)

# ========== 配置 ==========
# GC10-DET 源数据路径（Kaggle 下载位置）
SRC_DIR = os.path.expanduser("~/.cache/kagglehub/datasets/alex000kim/gc10det/versions/1")
# 目标路径
DST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GC10-DET")

# 类别映射 (XML name -> 类别ID)
CLASS_MAP = {
    '1_chongkong':  0,
    '2_hanfeng':    1,
    '3_yueyawan':   2,
    '4_shuiban':    3,
    '5_youban':     4,
    '6_siban':      5,
    '7_yiwu':       6,
    '8_yahen':      7,
    '9_zhehen':     8,
    '10_yaozhe':    9,
    '10_yaozhed':   9,   # 标注笔误修正
    # 'd' 标签跳过（仅1个标注错误）
}

CLASS_NAMES = [
    'chongkong', 'hanfeng', 'yueyawan', 'shuiban', 'youban',
    'siban', 'yiwu', 'yahen', 'zhehen', 'yaozhe'
]

# 划分比例
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# TEST_RATIO = 0.1

# ========== 处理 ==========
print("=== GC10-DET: VOC -> YOLO 格式转换 ===\n")

# 创建目录结构
for split in ['train', 'val', 'test']:
    Path(DST_DIR, 'images', split).mkdir(parents=True, exist_ok=True)
    Path(DST_DIR, 'labels', split).mkdir(parents=True, exist_ok=True)

# Step 1: 收集所有有效样本
valid_samples = []
skipped = {'d': 0, 'no_image': 0, 'no_xml': 0, 'other': 0}

xml_dir = Path(SRC_DIR, 'lable')
xml_files = list(xml_dir.glob('*.xml'))
print(f"XML 标注文件总数: {len(xml_files)}")

for xml_path in xml_files:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find('filename').text
    img_w = int(root.find('size/width').text)
    img_h = int(root.find('size/height').text)

    # 查找图片（可能在任何类别文件夹中）
    img_path = None
    for class_id in range(1, 11):
        candidate = Path(SRC_DIR, str(class_id), filename)
        if candidate.exists():
            img_path = candidate
            break

    if img_path is None:
        skipped['no_image'] += 1
        continue

    # 解析所有目标框
    yolo_lines = []
    valid_objects = 0

    for obj in root.findall('object'):
        name = obj.find('name').text

        # 跳过错误标签 'd'
        if name == 'd':
            skipped['d'] += 1
            continue

        if name not in CLASS_MAP:
            skipped['other'] += 1
            print(f"  未知类别: {name} in {xml_path.name}")
            continue

        cls_id = CLASS_MAP[name]
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)

        # 转换为 YOLO 格式 (cx, cy, w, h 归一化)
        cx = ((xmin + xmax) / 2) / img_w
        cy = ((ymin + ymax) / 2) / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h

        # 边界检查
        cx = max(0, min(1, cx))
        cy = max(0, min(1, cy))
        w = max(0, min(1, w))
        h = max(0, min(1, h))

        if w <= 0 or h <= 0:
            continue

        yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        valid_objects += 1

    if valid_objects > 0:
        valid_samples.append((img_path, yolo_lines, filename))

print(f"有效样本: {len(valid_samples)}")
print(f"跳过: d标签={skipped['d']}, 无图片={skipped['no_image']}, 无XML={skipped['no_xml']}, 其他={skipped['other']}")

# Step 2: 划分数据集
random.shuffle(valid_samples)
n_total = len(valid_samples)
n_train = int(n_total * TRAIN_RATIO)
n_val = int(n_total * VAL_RATIO)

splits = {
    'train': valid_samples[:n_train],
    'val': valid_samples[n_train:n_train + n_val],
    'test': valid_samples[n_train + n_val:],
}

# Step 3: 复制图片并写入标签
for split_name, samples in splits.items():
    for img_path, yolo_lines, filename in samples:
        # 复制图片
        dst_img = Path(DST_DIR, 'images', split_name, filename)
        shutil.copy2(img_path, dst_img)

        # 写入 YOLO 标签
        label_name = os.path.splitext(filename)[0] + '.txt'
        dst_label = Path(DST_DIR, 'labels', split_name, label_name)
        with open(dst_label, 'w') as f:
            f.write('\n'.join(yolo_lines))

for split_name, samples in splits.items():
    print(f"  {split_name}: {len(samples)} 张")

# Step 4: 创建 data.yaml
yaml_path = Path(DST_DIR, 'data.yaml')
yaml_content = f"""# GC10-DET Dataset (10 classes, YOLO format)
train: images/train
val: images/val
test: images/test

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""

with open(yaml_path, 'w') as f:
    f.write(yaml_content)

print(f"\n=== GC10-DET 数据集准备完成 ===")
print(f"输出目录: {DST_DIR}")
print(f"总样本: {n_total} (train={n_train}, val={n_val}, test={n_total - n_train - n_val})")
print(f"类别数: {len(CLASS_NAMES)}")
print(f"YAML: {yaml_path}")
