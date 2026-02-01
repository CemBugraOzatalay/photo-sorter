from pathlib import Path
import json
import yaml

with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

db_dir = Path(cfg["db_dir"])
meta_path = db_dir / "face_meta.jsonl"
fixed_path = db_dir / "face_meta_fixed.jsonl"

face_counter = 0
total = 0

with open(meta_path, "r", encoding="utf-8") as fin, open(fixed_path, "w", encoding="utf-8") as fout:
    for line in fin:
        total += 1
        obj = json.loads(line)

        if obj.get("has_face"):
            obj["emb_index"] = face_counter
            face_counter += 1
        else:
            obj["emb_index"] = None

        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

print("Toplam satır:", total)
print("Yüzlü satır:", face_counter)
print("Yeni meta:", fixed_path)
