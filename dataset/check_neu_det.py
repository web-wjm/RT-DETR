from collections import Counter
from pathlib import Path
import zipfile

from PIL import Image


ROOT = Path(__file__).resolve().parent
ZIP_PATH = ROOT / "NEU-DET.zip"
SPLITS = ("train", "val", "test")


def main():
    print("zip_exists", ZIP_PATH.exists(), "zip_size", ZIP_PATH.stat().st_size if ZIP_PATH.exists() else None)
    if ZIP_PATH.exists():
        with zipfile.ZipFile(ZIP_PATH) as zf:
            names = zf.namelist()
            ext_counts = Counter(Path(name).suffix.lower() for name in names if Path(name).suffix)
            print("zip_bad_file", zf.testzip())
            print("zip_ext_counts", dict(sorted(ext_counts.items())))
            print("zip_total_entries", len(names))

    all_img_stems = set()
    all_lbl_stems = set()
    class_counts = Counter()
    invalid_lines = []
    empty_labels = []
    image_errors = []
    size_counts = Counter()
    per_split = {}

    for split in SPLITS:
        img_dir = ROOT / "images" / split
        lbl_dir = ROOT / "labels" / split
        imgs = sorted(p for p in img_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
        lbls = sorted(lbl_dir.glob("*.txt"))
        per_split[split] = (len(imgs), len(lbls))
        all_img_stems.update(p.stem for p in imgs)
        all_lbl_stems.update(p.stem for p in lbls)

        for image_path in imgs:
            try:
                with Image.open(image_path) as image:
                    size_counts[image.size] += 1
                    image.verify()
            except Exception as exc:
                image_errors.append((str(image_path), repr(exc)))

        for label_path in lbls:
            text = label_path.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                empty_labels.append(str(label_path))
                continue
            for line_no, line in enumerate(text.splitlines(), 1):
                parts = line.split()
                if len(parts) != 5:
                    invalid_lines.append((str(label_path), line_no, line, "field_count"))
                    continue
                try:
                    cls = int(float(parts[0]))
                    values = [float(x) for x in parts[1:]]
                except ValueError:
                    invalid_lines.append((str(label_path), line_no, line, "parse"))
                    continue
                class_counts[cls] += 1
                if cls < 0 or cls > 5 or any(v < 0 or v > 1 for v in values) or values[2] <= 0 or values[3] <= 0:
                    invalid_lines.append((str(label_path), line_no, line, "range"))

    missing_labels = sorted(all_img_stems - all_lbl_stems)
    orphan_labels = sorted(all_lbl_stems - all_img_stems)
    print("per_split_images_labels", per_split)
    print("total_images", sum(v[0] for v in per_split.values()))
    print("total_labels", sum(v[1] for v in per_split.values()))
    print("missing_labels_for_images", len(missing_labels), missing_labels[:10])
    print("orphan_labels_without_images", len(orphan_labels), orphan_labels[:10])
    print("class_counts", dict(sorted(class_counts.items())))
    print("empty_labels", len(empty_labels), empty_labels[:5])
    print("invalid_lines", len(invalid_lines), invalid_lines[:5])
    print("image_errors", len(image_errors), image_errors[:3])
    print("image_size_counts_top", size_counts.most_common(10))


if __name__ == "__main__":
    main()
