# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional
import shutil
import cv2
import numpy as np

def copy_files_to_dir(files: Iterable[str], dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    c = 0
    for f in files:
        src = Path(f)
        if src.exists():
            shutil.copy2(src, dest / src.name)
            c += 1
    return c

def encode_jpeg(img: np.ndarray, quality: int = 90) -> Optional[bytes]:
    if img is None or img.size == 0:
        return None
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else None

def write_jpeg(img: np.ndarray, path: Path, quality: int = 92) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    path.write_bytes(buf.tobytes())
    return path
