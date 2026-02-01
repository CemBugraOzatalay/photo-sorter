import yaml

with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

print("CONFIG OK")
print("input_dirs =", cfg["input_dirs"])
print("extensions =", cfg["image_extensions"])
