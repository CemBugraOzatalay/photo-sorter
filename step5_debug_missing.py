from pathlib import Path
import yaml
import hashlib

# Config oku
with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

db_dir = Path(cfg["db_dir"])
missing_txt = db_dir / "missing_paths.txt"
log_path = db_dir / "missing_debug_log.txt"

def md5_of_file(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

missing_paths = []
with open(missing_txt, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            missing_paths.append(Path(line))

ok = 0
fail = 0

with open(log_path, "w", encoding="utf-8") as out:
    for p in missing_paths:
        try:
            if not p.exists():
                out.write(f"[NOT_FOUND] {p}\n")
                fail += 1
                continue

            # Boyut kontrol (çok küçük/0 mı?)
            size = p.stat().st_size
            if size == 0:
                out.write(f"[ZERO_SIZE] {p}\n")
                fail += 1
                continue

            md5 = md5_of_file(p)
            out.write(f"[OK] size={size} md5={md5} {p}\n")
            ok += 1

        except PermissionError as e:
            out.write(f"[PERMISSION] {p} | {e}\n")
            fail += 1
        except Exception as e:
            out.write(f"[ERROR] {p} | {type(e).__name__}: {e}\n")
            fail += 1

print("Debug bitti.")
print("OK:", ok)
print("FAIL:", fail)
print("Log:", log_path)
