from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ET
import zipfile


ROOT = Path(__file__).resolve().parent
ZIP_PATH = ROOT / "NEU-DET.zip"
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
MISSING_STEM = "crazing_240"


def convert_xml_to_yolo(xml_text):
    root = ET.fromstring(xml_text)
    size = root.find("size")
    width = float(size.findtext("width", default="200")) if size is not None else 200.0
    height = float(size.findtext("height", default="200")) if size is not None else 200.0
    lines = []
    for obj in root.iter("object"):
        class_name = obj.findtext("name", default="").strip()
        if class_name not in CLASSES:
            continue
        class_id = CLASSES.index(class_name)
        box = obj.find("bndbox")
        xmin = float(box.findtext("xmin"))
        ymin = float(box.findtext("ymin"))
        xmax = float(box.findtext("xmax"))
        ymax = float(box.findtext("ymax"))
        cx = ((xmin + xmax) / 2.0) / width
        cy = ((ymin + ymax) / 2.0) / height
        bw = (xmax - xmin) / width
        bh = (ymax - ymin) / height
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        bw = max(0.0, min(1.0, bw))
        bh = max(0.0, min(1.0, bh))
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return "\n".join(lines) + "\n"


def main():
    out_img = ROOT / "images" / "val" / f"{MISSING_STEM}.jpg"
    out_lbl = ROOT / "labels" / "val" / f"{MISSING_STEM}.txt"
    out_img.parent.mkdir(parents=True, exist_ok=True)
    out_lbl.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH) as zf:
        image_name = next(name for name in zf.namelist() if name.endswith(f"{MISSING_STEM}.jpg"))
        xml_name = next(name for name in zf.namelist() if name.endswith(f"{MISSING_STEM}.xml"))

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / f"{MISSING_STEM}.jpg"
            with zf.open(image_name) as src, open(tmp_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            shutil.copy2(tmp_path, out_img)

        xml_text = zf.read(xml_name).decode("utf-8", errors="ignore")
        out_lbl.write_text(convert_xml_to_yolo(xml_text), encoding="utf-8")

    print(f"fixed {out_img}")
    print(f"fixed {out_lbl}")


if __name__ == "__main__":
    main()
