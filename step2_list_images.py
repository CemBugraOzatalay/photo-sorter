from pathlib import Path
import yaml

# 1) Config oku
with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

input_dirs = cfg["input_dirs"]
exts = [e.lower() for e in cfg["image_extensions"]]

# 2) Resimleri topla
images = []

for root in input_dirs:
    root_path = Path(root)

    # Klasör var mı?
    if not root_path.exists():
        print("Klasör bulunamadı:", root_path)
        continue

    # Rekürsif tarama: alt klasörler dahil
    for p in root_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            images.append(p)

print("Toplam resim:", len(images))
print("İlk 10 dosya:")
for p in images[:10]:
    print(" -", p)
