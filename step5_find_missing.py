from pathlib import Path
import yaml
import json
import hashlib

# --- config oku ---
with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

input_dirs = cfg["input_dirs"]
exts = {e.lower() for e in cfg["image_extensions"]}

db_dir = Path(cfg["db_dir"])
index_path = db_dir / "index.jsonl"

# --- index'ten path set'i çıkar ---
indexed_paths = set()
with open(index_path, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        indexed_paths.add(obj["path"])

# --- DCIM'i tekrar tara ve eksikleri bul ---
all_paths = []
for root in input_dirs:
    rootp = Path(root)
    for p in rootp.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            all_paths.append(str(p))

missing = [p for p in all_paths if p not in indexed_paths]

print("DCIM toplam:", len(all_paths))
print("Index toplam:", len(indexed_paths))
print("Eksik (index'e yazılmamış):", len(missing))

print("\nİlk 30 eksik dosya:")
for p in missing[:30]:
    print(" -", p)

# İstersen dosyaya da yazalım:
out_txt = db_dir / "missing_paths.txt"
with open(out_txt, "w", encoding="utf-8") as f:
    for p in missing:
        f.write(p + "\n")

print("\nEksikler dosyaya yazıldı:", out_txt)
