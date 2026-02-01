import numpy as np
import cv2

p = r"input"

def imread_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img

img = imread_unicode(p)
print("OK" if img is not None else "FAIL", "shape:", None if img is None else img.shape)
