from pathlib import Path
import json
import shutil
from collections import Counter
import yaml

# 1) config
with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

db_dir = Path(cfg["db_dir"])
clusters_path = db_dir / "face_clusters.jsonl"

out_root = Path("output")
people_dir = out_root / "people"
unknown_dir = people_dir / "unknown_faces"

people_dir.mkdir(parents=True, exist_ok=True)
unknown_dir.mkdir(parents=True, exist_ok=True)

# 2) önce cluster sayımlarını al (sadece yüzlü ve cluster_id >=0)
cluster_counts = Counter()
rows = []

with open(clusters_path, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        rows.append(obj)
        if obj.get("has_face") and obj.get("cluster_id") is not None and obj["cluster_id"] >= 0:
            cluster_counts[obj["cluster_id"]] += 1

sorted_clusters = [cid for cid, _ in cluster_counts.most_common()]
cid_to_person = {cid: f"Person_{i+1:04d}" for i, cid in enumerate(sorted_clusters)}

faces_total = sum(1 for r in rows if r.get("has_face"))
unknown_total = sum(1 for r in rows if r.get("has_face") and r.get("cluster_id") == -1)

print("Toplam yüzlü foto:", faces_total)
print("Toplam cluster (>=0):", len(sorted_clusters))
print("Unknown faces (-1):", unknown_total)

# 3) güvenli kopyalama (isim çakışırsa _1 _2)
def safe_copy(src, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    src_p = Path(src)
    dst = dst_dir / src_p.name

    if dst.exists():
        stem = src_p.stem
        suf = src_p.suffix
        k = 1
        while True:
            cand = dst_dir / f"{stem}_{k}{suf}"
            if not cand.exists():
                dst = cand
                break
            k += 1

    shutil.copy2(src_p, dst)

# 4) sadece yüzlüleri kopyala
copied = 0
failed = 0

for obj in rows:
    if not obj.get("has_face"):
        continue  # yüzsüzleri ŞİMDİLİK dokunmuyoruz

    p = obj["path"]
    cid = obj.get("cluster_id")

    try:
        if cid is None or cid == -1:
            safe_copy(p, unknown_dir)
        else:
            person_name = cid_to_person.get(cid, "Person_9999")
            safe_copy(p, people_dir / person_name)

        copied += 1
        if copied % 500 == 0:
            print("Kopyalanan:", copied)

    except Exception:
        failed += 1

print("Bitti ✅")
print("Toplam kopyalanan (sadece yüzlü):", copied)
print("Kopyalanamayan:", failed)
print("Output:", out_root.resolve())
