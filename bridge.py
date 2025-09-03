# -*- coding: utf-8 -*-
# yolovizi_core/bridge.py
from __future__ import annotations
from typing import List, Dict
from .storage import ProductStore, calibration_path
from .training import Trainer
from .tracking import Tracker
from .utils.pylon_cam import PylonGrabber
from .utils.image_io import write_jpeg
from .config import MODELS_DIR
from .labeling import label_images_in_folder
from .synth import synth_init as _synth_init_poly, synth_make as _synth_make, synth_init_rect as _synth_init_rect

_store  = ProductStore()
_train  = Trainer()
_track  = Tracker()

def list_products() -> List[Dict]:
    items = _store.list()
    out = []
    for p in items:
        pid = int(p["id"])
        q = dict(p)
        q["images_dir"] = str(_store.images_dir(pid))
        q["labels_dir"] = str(_store.labels_dir(pid))
        mp = (MODELS_DIR / f"product_{pid}.pt")
        q["model_path"] = str(mp) if mp.exists() else None
        q["image_path"] = str(_store.get_product_image(pid)) 
        out.append(q)
    return out

def create_product(name: str) -> Dict:
    return _store.create(name)

def get_product_image(product_id: int) -> str:
    return str(_store.get_product_image(product_id))

def train(product_id: int, dataset_dir: str) -> Dict:
    model_path = _train.train(product_id, dataset_dir)
    return {"status": "done", "model_path": model_path}

def track_start(product_id: int):
    _track.start(product_id)

def track_stop():
    _track.stop()

def track_status() -> Dict:
    return _track.status()

def get_frame() -> bytes | None:
    return _track.get_jpeg()

def capture_background() -> str:
    with PylonGrabber() as g:
        img = g.grab_one()
        if img is None:
            raise RuntimeError("Kamera goruntusu alinamadi (background).")
        out = calibration_path()
        write_jpeg(img, out, quality=95)
        return str(out)

def capture_product_photo(product_id: int) -> str:
    with PylonGrabber() as g:
        img = g.grab_one()
        if img is None:
            raise RuntimeError("Kamera goruntusu alinamadi (product).")
        out = _store.save_captured(product_id, img)
        return str(out)

def label_images(product_id: int, images_dir: str | None = None):
    return label_images_in_folder(product_id, images_dir)

def synth_init_poly(product_id: int, image_path: str | None = None) -> dict:
    return _synth_init_poly(product_id, image_path)

def synth_init_rect(product_id: int, image_path: str | None = None) -> dict:
    return _synth_init_rect(product_id, image_path)


def synth_make_data(product_id: int, count: int = 1200, bg_dir: str | None = None,
                    out_w: int = 1280, out_h: int = 720) -> dict:
    return _synth_make(product_id,
                       count=count, 
                       bg_dir=bg_dir,
                       out_w=out_w,
                       out_h=out_h)

def sync_camera() -> bool:
    from .utils.pylon_cam import PylonGrabber
    try:
        with PylonGrabber() as g:
            return g.synchronize()
    except Exception as e:
        print(f"[sync_camera] error: {e}")
        return False