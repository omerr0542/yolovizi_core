# yolovizi_core/config.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
from pathlib import Path

try:
    import winreg
except ImportError:
    winreg = None  

REG_PATH = r"Software\VisionSuite"

def _reg_get(name: str, default: str = "") -> str:
    if not winreg:
        return default
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, REG_PATH) as key:
            val, _ = winreg.QueryValueEx(key, name)
            return str(val)
    except FileNotFoundError:
        return default
    except Exception:
        return default


BASE_DIR = Path(
    os.environ.get("VISIONSUITE_HOME",
    Path(os.getenv("LOCALAPPDATA", str(Path.home()))) / "VisionSuite")
).resolve()

DATA_DIR     = BASE_DIR / "data"
PRODUCTS_DB  = DATA_DIR / "products.json"
DATASETS     = DATA_DIR / "datasets"
MODELS_DIR   = DATA_DIR / "models"
TMP_DIR      = DATA_DIR / "tmp"
TEMPLATE_DIR = DATA_DIR / "templates"

PYLON_SERIAL = os.environ.get("VISIONSUITE_PYLON_SERIAL",
                _reg_get("PYLON_SERIAL", ""))

FRAME_WIDTH  = int(os.environ.get("VISIONSUITE_FRAME_WIDTH",
                _reg_get("FRAME_WIDTH", "1280")))

FRAME_HEIGHT = int(os.environ.get("VISIONSUITE_FRAME_HEIGHT",
                _reg_get("FRAME_HEIGHT", "720")))

TARGET_FPS   = int(os.environ.get("VISIONSUITE_TARGET_FPS",
                _reg_get("TARGET_FPS", "30")))

PIXEL_FORMAT = os.environ.get("VISIONSUITE_PIXEL_FORMAT",
                _reg_get("PIXEL_FORMAT", "BGR8"))

HORIZ_FLIP   = bool(int(os.environ.get("VISIONSUITE_FLIP_H",
                _reg_get("FLIP_H", "0"))))

EPOCHS: int = int(os.environ.get("VISIONSUITE_EPOCHS",
             _reg_get("EPOCHS", "30")))

IMGSZ:  int = int(os.environ.get("VISIONSUITE_IMGSZ",
             _reg_get("IMGSZ", "640")))

MODEL:  str = os.environ.get("VISIONSUITE_MODEL",
             _reg_get("MODEL", "yolov8s"))

MODEL_SEG = MODEL + "-seg.pt"
MODEL_POSE = MODEL + "-pose.pt"
MODEL = MODEL + ".pt"

WORLD_MAX_X: int = int(os.environ.get("VISIONSUITE_WORLD_MAX_X",
                 _reg_get("WORLD_MAX_X", str(FRAME_WIDTH))))

WORLD_MAX_Y: int = int(os.environ.get("VISIONSUITE_WORLD_MAX_Y",
                 _reg_get("WORLD_MAX_Y", str(FRAME_HEIGHT))))

CONF_THRESHOLD = float(os.getenv("VISIONSUITE_CONF_THR", "0.80"))
