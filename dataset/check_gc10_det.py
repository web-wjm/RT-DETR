from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent / "GC10-DET"
SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
NUM_CLASSES = 10


def main():
    per_split = {}
    split_stems = {}
    class_counts = defaultdict(Counter)
    image_errors = []
    invalid_lines = []
    empty_labels = []
    duplicate_label_rows = []
    size_counts = Counter()
    content_hashes = defaultdict(list)

    for split in SPLITS:
        image_dir = ROOT / "images" / split
        label_dir = ROOT / "labels" / split
        images = sorted(p for p in image_dir.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        labels = sorted(label_dir.glob("*.txt"))
        image_stems = {p.stem for p in images}
        label_stems = {p.stem for p in labels}
        split_stems[split] = image_stems
        per_split[split] = {
            "images": len(images),
            "labels": len(labels),
            "missing_labels": len(image_stems - label_stems),
            "orphan_labels": len(label_stems - image_stems),
        }

        for image_path in images:
            try:
                with Image.open(image_path) as image:
                    size_counts[image.size] += 1
                    image.verify()
                content_hashes[sha256(image_path.read_bytes()).hexdigest()].append((split, image_path.name))
            except Exception as exc:
                image_errors.append((str(image_path), repr(exc)))

        for label_path in labels:
            text = label_path.read_text(encoding="utf-8", errors="strict").strip()
            if not text:
                empty_labels.append(str(label_path))
                continue

            seen_rows = set()
            for line_no, line in enumerate(text.splitlines(), 1):
                parts = line.split()
                if len(parts) != 5:
                    invalid_lines.append((str(label_path), line_no, "field_count", line))
                    continue
                try:
                    cls = int(parts[0])
                    cx, cy, width, height = (float(value) for value in parts[1:])
                except ValueError:
                    invalid_lines.append((str(label_path), line_no, "parse", line))
                    continue

                class_counts[split][cls] += 1
                values = (cx, cy, width, height)
                if not 0 <= cls < NUM_CLASSES:
                    invalid_lines.append((str(label_path), line_no, "class_range", line))
                elif any(value < 0 or value > 1 for value in values) or width <= 0 or height <= 0:
                    invalid_lines.append((str(label_path), line_no, "value_range", line))
                elif cx - width / 2 < -1e-6 or cx + width / 2 > 1 + 1e-6:
                    invalid_lines.append((str(label_path), line_no, "x_boundary", line))
                elif cy - height / 2 < -1e-6 or cy + height / 2 > 1 + 1e-6:
                    invalid_lines.append((str(label_path), line_no, "y_boundary", line))

                normalized_row = (cls, cx, cy, width, height)
                if normalized_row in seen_rows:
                    duplicate_label_rows.append((str(label_path), line_no, line))
                seen_rows.add(normalized_row)

    split_overlap = {}
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1:]:
            split_overlap[f"{left}-{right}"] = sorted(split_stems[left] & split_stems[right])

    duplicate_content = [items for items in content_hashes.values() if len(items) > 1]

    print("root", ROOT)
    print("per_split", per_split)
    print("total_images", sum(item["images"] for item in per_split.values()))
    print("total_labels", sum(item["labels"] for item in per_split.values()))
    print("class_counts", {split: dict(sorted(counts.items())) for split, counts in class_counts.items()})
    print("split_stem_overlap", {key: len(value) for key, value in split_overlap.items()})
    print("duplicate_image_content", len(duplicate_content), duplicate_content[:5])
    print("empty_labels", len(empty_labels), empty_labels[:5])
    print("invalid_lines", len(invalid_lines), invalid_lines[:10])
    print("duplicate_label_rows", len(duplicate_label_rows), duplicate_label_rows[:5])
    print("image_errors", len(image_errors), image_errors[:5])
    print("image_size_counts", size_counts.most_common())


if __name__ == "__main__":
    main()
