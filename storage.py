# -*- coding: utf-8 -*-
# yolovizi_core/storage.py
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Iterable
from .config import PRODUCTS_DB, DATASETS, DATA_DIR, TEMPLATE_DIR
from .utils.image_io import copy_files_to_dir, write_jpeg
import numpy as np

def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

def _load_db() -> Dict:
    if not PRODUCTS_DB.exists():
        return {"seq": 0, "items": []}
    try:
        return json.loads(PRODUCTS_DB.read_text(encoding="utf-8"))
    except Exception:
        return {"seq": 0, "items": []}

def _save_db(db: Dict):
    PRODUCTS_DB.parent.mkdir(parents=True, exist_ok=True)
    PRODUCTS_DB.write_text(json.dumps(db, ensure_ascii=False, indent=2), encoding="utf-8")

class ProductStore:
    def list(self) -> List[Dict]:
        return list(_load_db().get("items", []))

    def create(self, name: str) -> Dict:
        db = _load_db()
        seq = int(db.get("seq", 0)) + 1
        db["seq"] = seq
        item = {"id": seq, "name": name, "created_at": _now_iso()}
        items = db.get("items", [])
        items.append(item)
        db["items"] = items
        _save_db(db)
        return item

    def add_images(self, product_id: int, files: Iterable[str]) -> int:
        dest = self.images_dir(product_id)
        return copy_files_to_dir(files, dest)

    def images_dir(self, product_id: int) -> Path:
        return DATASETS / f"product_{product_id}" / "images"

    def labels_dir(self, product_id: int) -> Path:
        return DATASETS / f"product_{product_id}" / "labels"

    def save_captured(self, product_id: int, img_bgr: np.ndarray) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out = self.images_dir(product_id) / f"cap_{ts}.jpg"
        return write_jpeg(img_bgr, out, quality=95)

    def get_product_image(self, product_id: int) -> str:
        return str(TEMPLATE_DIR / f"product_{product_id}" / "template.png")

def calibration_path() -> Path:
    return DATA_DIR / "calibration.jpg"
