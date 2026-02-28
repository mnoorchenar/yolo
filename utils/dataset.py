# Zip extraction, class parsing, data.yaml builder
import os, zipfile, shutil, json, yaml
from pathlib import Path


def extract_labelstudio_zip(zip_path: str, dest_dir: str) -> dict:
    """
    Extract a Label Studio YOLO-format export zip.

    Expected zip structure (Label Studio default):
        images/train/*.jpg | *.jpeg | *.png
        labels/train/*.txt
        classes.txt          ← required
        images/val/          ← optional
        labels/val/          ← optional
    """
    dest = Path(dest_dir)
    if dest.exists():
        shutil.rmtree(dest)          # replace previous dataset
    dest.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest)

    classes   = _find_classes(dest)
    img_train = _count_images(dest / "images" / "train") + _count_images(dest / "images")
    img_val   = _count_images(dest / "images" / "val")

    return {
        "classes":     classes,
        "num_classes": len(classes),
        "image_count": img_train + img_val,
        "train_count": img_train,
        "val_count":   img_val,
        "root":        str(dest),
    }


def build_data_yaml(dataset_root: str, classes: list, output_path: str) -> str:
    root      = Path(dataset_root)
    train_dir = root / "images" / "train"
    val_dir   = root / "images" / "val"

    # Fall back to images/ for both splits if no train/ sub-folder
    if not train_dir.exists():
        train_dir = root / "images"
    if not val_dir.exists():
        val_dir = train_dir     # reuse train as val for demo

    data = {
        "path":  str(root.resolve()),
        "train": str(train_dir.resolve()),
        "val":   str(val_dir.resolve()),
        "nc":    len(classes),
        "names": classes,
    }
    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    return output_path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_classes(root: Path) -> list:
    """Try classes.txt → notes.json → return empty list."""
    for f in root.rglob("classes.txt"):
        lines = [l.strip() for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        if lines:
            return lines

    for f in root.rglob("notes.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(d, list):
                return [item.get("name", str(i)) for i, item in enumerate(d)]
            if "categories" in d:
                return [c["name"] for c in d["categories"]]
        except Exception:
            continue

    return []


def _count_images(path: Path) -> int:
    if not path.exists():
        return 0
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sum(1 for f in path.rglob("*") if f.suffix.lower() in exts)