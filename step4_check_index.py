from pathlib import Path
import yaml

with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

index_path = Path(cfg["db_dir"]) / "index.jsonl"

if not index_path.exists():
    print("index.jsonl bulunamadı:", index_path)
    raise SystemExit

count = 0
with open(index_path, "r", encoding="utf-8") as f:
    for _ in f:
        count += 1

print("index.jsonl bulundu ✅")
print("Toplam kayıt (satır):", count)
print("Dosya yolu:", index_path)
