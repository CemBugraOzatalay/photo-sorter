from pathlib import Path
import yaml
import json
import hashlib

# 1) Config oku: input klasörleri + uzantılar burada
with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

input_dirs = cfg["input_dirs"]
exts = {e.lower() for e in cfg["image_extensions"]}

# 2) db klasörü ve index dosyası yolu
db_dir = Path(cfg["db_dir"])
db_dir.mkdir(parents=True, exist_ok=True)   # yoksa oluştur
index_path = db_dir / "index.jsonl"

# 3) Daha önce kaydedilmiş md5'leri oku (cache)
#    Amaç: aynı dosyayı tekrar kaydetmeyelim
seen_md5 = set()
if index_path.exists():
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                seen_md5.add(obj["md5"])
            except:
                pass

def md5_of_file(p: Path) -> str:
    """Dosyanın içeriğinden MD5 parmak izi üretir."""
    h = hashlib.md5()
    with open(p, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)  # 1MB parça parça oku (RAM şişmesin)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def iter_images():
    """input_dirs içindeki tüm görselleri rekürsif gezer."""
    for root in input_dirs:
        root_path = Path(root)
        if not root_path.exists():
            print("Klasör bulunamadı:", root_path)
            continue

        for p in root_path.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                yield p

# 4) Index'e yeni kayıt ekle
new_count = 0

with open(index_path, "a", encoding="utf-8") as out:
    for p in iter_images():
        try:
            md5 = md5_of_file(p)
        except:
            continue

        # Daha önce yazıldıysa atla
        if md5 in seen_md5:
            continue

        rec = {
            "path": str(p),
            "md5": md5,
            "size": p.stat().st_size
        }

        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        seen_md5.add(md5)
        new_count += 1

print("Index oluşturuldu.")
print("Yeni eklenen kayıt:", new_count)
print("Index yolu:", index_path)
