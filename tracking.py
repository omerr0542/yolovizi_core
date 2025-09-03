# -*- coding: utf-8 -*-
# yolovizi_core/tracking.py
from __future__ import annotations

from typing import Optional, Dict, Tuple
import math, json
import numpy as np
import cv2

from .utils.pylon_cam import PylonGrabber
from .config import MODELS_DIR, DATA_DIR, WORLD_MAX_X, WORLD_MAX_Y, CONF_THRESHOLD
from .utils.image_io import encode_jpeg

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None  # type: ignore


def _wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0

def _to360(a: float) -> float:
    return (a + 360.0) % 360.0

def _angdiff(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


class AngleEMA:
    def __init__(self, alpha: float = 0.20):
        self.alpha = float(alpha)
        self.s: Optional[float] = None
        self.c: Optional[float] = None

    def reset(self):
        self.s = self.c = None

    def update(self, a_deg: float) -> float:
        rad = math.radians(a_deg)
        s, c = math.sin(rad), math.cos(rad)
        if self.s is None or self.c is None:
            self.s, self.c = s, c
        else:
            a = self.alpha
            self.s = (1.0 - a) * self.s + a * s
            self.c = (1.0 - a) * self.c + a * c
        out = math.degrees(math.atan2(self.s, self.c))
        return _to360(out)

def _scale_xy(x: float, y: float, img_w: int, img_h: int) -> tuple[float,float]:
    sx = (WORLD_MAX_X / float(img_w)) if img_w > 0 else 1.0
    sy = (WORLD_MAX_Y / float(img_h)) if img_h > 0 else 1.0
    return (x * sx, y * sy)

class PointTracker:
    def __init__(self, alpha: float = 0.4, win: int = 28, patch: int = 16):
        self.alpha = float(alpha)
        self.win = int(win)
        self.patch = int(patch)
        self.prev: Optional[Tuple[int,int]] = None
        self.tmpl: Optional[np.ndarray] = None  # gray

    def set_template(self, gray: np.ndarray, pt_xy: Tuple[int,int]):
        x, y = int(pt_xy[0]), int(pt_xy[1])
        h, w = gray.shape[:2]
        s = self.patch
        x0, y0 = max(0, x - s), max(0, y - s)
        x1, y1 = min(w, x + s), min(h, y + s)
        crop = gray[y0:y1, x0:x1]
        if crop.size == 0: return
        self.tmpl = crop.copy()
        self.prev = (x, y)

    def update(self, gray: np.ndarray) -> Optional[Tuple[int,int]]:
        if self.tmpl is None or self.prev is None:
            return None
        x, y = self.prev
        h, w = gray.shape[:2]
        x0, y0 = max(0, x - self.win), max(0, y - self.win)
        x1, y1 = min(w, x + self.win), min(h, y + self.win)
        roi = gray[y0:y1, x0:x1]
        th, tw = self.tmpl.shape[:2]
        if roi.shape[0] < th or roi.shape[1] < tw:
            return self.prev
        res = cv2.matchTemplate(roi, self.tmpl, cv2.TM_CCOEFF_NORMED)
        _, _, _, maxLoc = cv2.minMaxLoc(res)
        tx = x0 + maxLoc[0] + tw // 2
        ty = y0 + maxLoc[1] + th // 2
        self.prev = (tx, ty)
        return self.prev

class DirectionDisambiguator:
    def __init__(self):
        self.front_tmpl: Optional[np.ndarray] = None  

    def load_from_template_files(self, product_id: int) -> None:
        tdir = DATA_DIR / "templates" / f"product_{product_id}"
        meta = tdir / "template.json"
        png  = tdir / "template.png"
        if not meta.exists() or not png.exists():
            self.front_tmpl = None
            return
        try:
            m = json.loads(meta.read_text(encoding="utf-8"))
            gp = m.get("grip_point", None)
            size = m.get("template_size", None)
            if not gp or not size:
                self.front_tmpl = None
                return
            img = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
            if img is None:
                self.front_tmpl = None
                return
            if img.shape[2] == 4:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            else:
                rgb = img
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            w, h = int(size[0]), int(size[1])
            cx, cy = w / 2.0, h / 2.0
            gx, gy = float(gp[0]), float(gp[1])
            vx, vy = gx - cx, gy - cy
            L = max(1.0, math.hypot(vx, vy))
            ux, uy = vx / L, vy / L
            r = 24
            px = int(round(cx + ux * r))
            py = int(round(cy + uy * r))
            s = 14
            x0, y0 = max(0, px - s), max(0, py - s)
            x1, y1 = min(gray.shape[1], px + s), min(gray.shape[0], py + s)
            roi = gray[y0:y1, x0:x1]
            self.front_tmpl = roi.copy() if roi.size else None
        except Exception:
            self.front_tmpl = None

    @staticmethod
    def _ncc(a: np.ndarray, t: np.ndarray) -> float:
        if a is None or t is None or a.size == 0 or t.size == 0: return -1.0
        if a.shape[0] < t.shape[0] or a.shape[1] < t.shape[1]:   return -1.0
        res = cv2.matchTemplate(a, t, cv2.TM_CCOEFF_NORMED)
        return float(res.max()) if res.size else -1.0

    def choose(self, frame_gray: np.ndarray, cx: float, cy: float, theta_deg: float) -> float:
        if self.front_tmpl is None:
            return _to360(theta_deg)
        H, W = frame_gray.shape[:2]

        def sample(phi_deg: float) -> Optional[np.ndarray]:
            rad = math.radians(phi_deg)
            r = 28
            px = int(round(cx + r * math.cos(rad)))
            py = int(round(cy - r * math.sin(rad)))
            s = 16
            x0, y0 = max(0, px - s), max(0, py - s)
            x1, y1 = min(W, px + s), min(H, py + s)
            roi = frame_gray[y0:y1, x0:x1]
            return roi if roi.size else None

        a0 = _to360(theta_deg)
        a1 = _to360(a0 + 180.0)
        r0 = sample(a0)
        r1 = sample(a1)
        s0 = self._ncc(r0, self.front_tmpl) if r0 is not None else -1.0
        s1 = self._ncc(r1, self.front_tmpl) if r1 is not None else -1.0
        return a0 if s0 >= s1 else a1


class Tracker:
    _SMOOTH_ALPHA_DEFAULT: float = 0.20

    def __init__(self) -> None:
        self._grab = PylonGrabber()
        self._model = None
        self._product_id: Optional[int] = None

        self._ang_ema = AngleEMA(self._SMOOTH_ALPHA_DEFAULT)
        self._angle_offset: float = 0.0

        self._pt = PointTracker(alpha=0.4, win=28, patch=16)
        self._had_reliable_grip: bool = False

        self._disamb = DirectionDisambiguator()

    def _load_model(self, product_id: int) -> None:
        if YOLO is None:
            self._model = None
            return
        ckpt = MODELS_DIR / f"product_{product_id}.pt"
        if not ckpt.exists():
            self._model = None
            return
        m = YOLO(str(ckpt))
        t = getattr(m, "task", None)
        if t != "pose":
            raise RuntimeError(f"Wrong checkpoint for pose: task={t}, ckpt={ckpt}")
        self._model = m

    def start(self, product_id: int) -> None:
        self._product_id = product_id
        self._load_model(product_id)
        self._grab.start()
        self._ang_ema.reset()
        self._pt.prev = None
        self._pt.tmpl = None
        self._had_reliable_grip = False
        self._disamb.load_from_template_files(product_id)

    def stop(self) -> None:
        self._grab.stop()
        self._grab.close()
        self._model = None
        self._product_id = None
        self._ang_ema.reset()
        self._pt.prev = None
        self._pt.tmpl = None
        self._had_reliable_grip = False

    def _smooth_angle(self, a_meas: float, trusted: bool) -> float:
        a = _wrap180(a_meas + self._angle_offset)
        return _to360(a)
   
    def status(self) -> Dict:
        
        def _scale_xy(x: float, y: float, img_w: int, img_h: int) -> tuple[float, float]:
            if img_w <= 0 or img_h <= 0 or WORLD_MAX_X <= 0 or WORLD_MAX_Y <= 0:
                return (x, y)
            return (x * (WORLD_MAX_X / float(img_w)),
                    y * (WORLD_MAX_Y / float(img_h)))

        frame = self._grab.get_last()
        if frame is None:
            return self._empty_status()

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detected = False
        conf = 0.0
        cx = cy = 0.0
        x1 = y1 = bw = bh = 0.0
        grip_x: Optional[float] = None
        grip_y: Optional[float] = None
        ang_meas: Optional[float] = None
        src_from_grip: bool = False

        if self._model is not None:
            try:
                res = self._model(frame, verbose=False)[0]

                best_i: Optional[int] = None
                if getattr(res, "boxes", None) is not None and len(res.boxes) > 0:
                    scores = res.boxes.conf.detach().cpu().numpy()
                    best_i = int(np.argmax(scores))
                    b = res.boxes[best_i]
                    xyxy = b.xyxy[0].detach().cpu().numpy().tolist()
                    conf = float(b.conf[0].item())
                    if conf < CONF_THRESHOLD:
                        detected = False
                        best_i = None
                    else:
                        x1i, y1i, x2i, y2i = map(int, xyxy)
                        bw, bh = float(x2i - x1i), float(y2i - y1i)
                        cx, cy = float(x1i + bw / 2.0), float(y1i + bh / 2.0)
                        x1, y1 = float(x1i), float(y1i)
                        detected = True

                if detected and best_i is not None and getattr(res, "keypoints", None) is not None:
                    try:
                        kxy = res.keypoints.xy[best_i].detach().cpu().numpy()  # (K,2)
                        if isinstance(kxy, np.ndarray) and kxy.ndim == 2 and kxy.shape[1] >= 2:
                            k1x, k1y = float(kxy[0, 0]), float(kxy[0, 1])  # grip
                            grip_x, grip_y = k1x, k1y

                            min_vec = max(4.0, 0.03 * max(bw, bh))
                            src_from_grip = False

                            if kxy.shape[0] >= 2:
                                k2x, k2y = float(kxy[1, 0]), float(kxy[1, 1])
                                vx, vy = (k2x - k1x), (k2y - k1y)
                                if (vx * vx + vy * vy) >= (min_vec * min_vec):
                                    ang_meas = math.degrees(math.atan2(vy, vx))  # k1->k2
                                    src_from_grip = True
                                else:
                                    dx, dy = (k1x - cx), (k1y - cy)
                                    if (dx * dx + dy * dy) >= (min_vec * min_vec):
                                        ang_meas = math.degrees(math.atan2(dy, dx))
                                        src_from_grip = True
                            else:
                                dx, dy = (k1x - cx), (k1y - cy)
                                if (dx * dx + dy * dy) >= (min_vec * min_vec):
                                    ang_meas = math.degrees(math.atan2(dy, dx))
                                    src_from_grip = True

                            if src_from_grip and not self._had_reliable_grip:
                                self._pt.set_template(gray, (int(k1x), int(k1y)))
                                self._had_reliable_grip = True
                        else:
                            grip_x = grip_y = None
                    except Exception:
                        pass

                if detected and grip_x is None and self._had_reliable_grip:
                    p = self._pt.update(gray)
                    if p is not None:
                        gx, gy = p
                        grip_x, grip_y = float(gx), float(gy)
                        dx, dy = grip_x - cx, grip_y - cy
                        min_vec = max(4.0, 0.03 * max(bw, bh))
                        if (dx * dx + dy * dy) >= (min_vec * min_vec):
                            ang_meas = math.degrees(math.atan2(dy, dx))
                            src_from_grip = True

                if detected and ang_meas is None:
                    if best_i is not None and getattr(res, "masks", None) is not None:
                        try:
                            if hasattr(res.masks, "xy") and res.masks.xy is not None:
                                mask = np.zeros((h, w), np.uint8)
                                segs = res.masks.xy[best_i]
                                if isinstance(segs, (list, tuple)) and len(segs) > 0:
                                    for seg in segs:
                                        pts = np.asarray(seg, dtype=np.int32).reshape(-1, 1, 2)
                                        cv2.fillPoly(mask, [pts], 255)
                                    ang_meas = self._angle_from_mask(mask)
                            else:
                                m = res.masks.data[best_i].detach().cpu().numpy().astype(np.uint8) * 255
                                if m.ndim > 2: m = m[0]
                                if m.shape[:2] != (h, w):
                                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                                ang_meas = self._angle_from_mask(m)
                        except Exception:
                            ang_meas = None

                    if ang_meas is None and bw > 1.0 and bh > 1.0:
                        pad = 0.15
                        xx1 = max(0, int(x1 - pad * bw))
                        yy1 = max(0, int(y1 - pad * bh))
                        xx2 = min(w, int(x1 + bw + pad * bw))
                        yy2 = min(h, int(y1 + bh + pad * bh))
                        roi = frame[yy1:yy2, xx1:xx2]
                        ang_meas = self._angle_from_roi(roi)

                if detected and ang_meas is not None and not src_from_grip:
                    ang_meas = self._disamb.choose(gray, cx, cy, ang_meas)

            except Exception:
                detected = False
                conf = 0.0
                cx = cy = 0.0
                x1 = y1 = bw = bh = 0.0
                grip_x = grip_y = None
                ang_meas = None
                src_from_grip = False

        angle_out = 0.0
        if detected and ang_meas is not None:
            angle_out = self._smooth_angle(ang_meas, trusted=src_from_grip)

        x2, y2 = (x1 + bw, y1 + bh)
        corners_px = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        corners_w = [ _scale_xy(float(px), float(py), w, h) for (px, py) in corners_px ]

        cx_w, cy_w = _scale_xy(cx, cy, w, h)
        gw = None
        if grip_x is not None and grip_y is not None:
            gw = _scale_xy(grip_x, grip_y, w, h)

        return {
            "product_id": self._product_id or -1,
            "detected": bool(detected),
            "confidence": float(conf),

            "center_x": float(cx), "center_y": float(cy),
            "center_wx": float(cx_w), "center_wy": float(cy_w),
            "angle_deg": float(_to360(angle_out)),

            "bbox_x": float(x1), "bbox_y": float(y1),
            "bbox_w": float(bw), "bbox_h": float(bh),

            "corners_px": [(float(a), float(b)) for (a, b) in corners_px],
            "corners_w":  [(float(a), float(b)) for (a, b) in corners_w],

            "grip_x": None if grip_x is None else float(grip_x),
            "grip_y": None if grip_y is None else float(grip_y),
            "grip_wx": None if gw is None else float(gw[0]),
            "grip_wy": None if gw is None else float(gw[1]),
        }

    def get_jpeg(self) -> Optional[bytes]:
        frame = self._grab.get_last()
        return encode_jpeg(frame) if frame is not None else None

    @staticmethod
    def _angle_from_mask(mask: np.ndarray) -> float:
        if mask is None or mask.size == 0:
            return 0.0
        m8 = mask.astype("uint8")
        m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
        M = cv2.moments(m8, binaryImage=True)
        mu20, mu02, mu11 = M.get("mu20", 0.0), M.get("mu02", 0.0), M.get("mu11", 0.0)
        if (mu20 + mu02) <= 1e-6:
            return 0.0
        theta = 0.5 * math.degrees(math.atan2(2.0 * mu11, (mu20 - mu02)))
        return _to360(theta)

    @staticmethod
    def _angle_from_roi(roi_bgr: np.ndarray) -> float:
        if roi_bgr is None or roi_bgr.size == 0:
            return 0.0
        g = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (5, 5), 0)
        _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = cv2.Canny(th, 0, 255)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return 0.0
        c = max(cnts, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        ang = rect[2]
        if rect[1][0] < rect[1][1]:
            ang += 90.0
        return _to360(ang)

    @staticmethod
    def _empty_status() -> Dict:
        return {
            "product_id": -1,
            "detected": False,
            "confidence": 0.0,
            "center_x": 0.0, "center_y": 0.0,
            "angle_deg": 0.0,
            "bbox_x": 0.0, "bbox_y": 0.0,
            "bbox_w": 0.0, "bbox_h": 0.0,
            "grip_x": None, "grip_y": None,
        }

