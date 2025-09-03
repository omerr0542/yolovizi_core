# -*- coding: utf-8 -*-
# yolovizi_core/synth.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional
import json, random
import cv2
import numpy as np
import math

from .storage import ProductStore, calibration_path
from .config import DATA_DIR
from .utils.image_io import write_jpeg

# ----------------------------
# Helpers (angle & PCA axis)
# ----------------------------
def _wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0

def _principal_axis_deg_from_alpha(alpha: np.ndarray) -> float:
    """PCA major axis (deg, [-180,180)) from alpha>0 pixels."""
    ys, xs = np.where(alpha > 0)
    if len(xs) < 10:
        return 0.0
    pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    mean, evecs = cv2.PCACompute(pts, mean=None, maxComponents=2)
    vx, vy = float(evecs[0, 0]), float(evecs[0, 1])
    return _wrap_deg(np.degrees(np.arctan2(vy, vx)))

# -----------------
# UI: polygon selection
# -----------------
def _select_polygon(img: np.ndarray) -> Optional[List[Tuple[int,int]]]:
    pts: List[Tuple[int,int]] = []
    disp = img.copy()
    win = "draw polygon (L:point, R:close, S/Enter:save, C:clear, Q:quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow(win, 1280, 720)

    def refresh():
        nonlocal disp
        disp = img.copy()
        for i, p in enumerate(pts):
            cv2.circle(disp, p, 3, (0,165,255), -1)
            if i > 0:
                cv2.line(disp, pts[i-1], p, (0,165,255), 2)
        if len(pts) >= 3:
            cv2.polylines(disp, [np.array(pts,np.int32)], True, (0,255,0), 2)
        txt = "[L]point  [R]close  [S/Enter]save  [C]clear  [Q]quit"
        cv2.putText(disp, txt, (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        cv2.putText(disp, txt, (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    def on_mouse(e,x,y,flags,param):
        if e == cv2.EVENT_LBUTTONDOWN:
            pts.append((int(x),int(y))); refresh()
        elif e == cv2.EVENT_RBUTTONDOWN:
            if len(pts) >= 3 and pts[0] != pts[-1]:
                pts.append(pts[0]); refresh()

    refresh()
    cv2.setMouseCallback(win, on_mouse)

    while True:
        cv2.imshow(win, disp)
        k = cv2.waitKey(10) & 0xFF
        if k in (ord('q'), 27):
            cv2.destroyWindow(win); return None
        if k == ord('c'):
            pts.clear(); refresh()
        if k in (ord('s'), 13):  # S or Enter
            if len(pts) >= 3 and pts[0] != pts[-1]:
                pts.append(pts[0])
            cv2.destroyWindow(win)
            return pts if len(pts) >= 4 else None

# -----------------
# UI: rectangle selection by DRAG
# -----------------
def _select_rect_drag(img: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
    win = "draw rectangle (Drag LMB, release to confirm; C:clear, Q:quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow(win, 1280, 720)

    p0: Optional[Tuple[int,int]] = None
    p1: Optional[Tuple[int,int]] = None
    dragging = False

    def refresh(show_live=False):
        d = img.copy()
        if p0 is not None:
            cv2.circle(d, p0, 4, (0,165,255), -1)
        if p0 is not None and (show_live and p1 is not None):
            x0,y0 = p0; x1,y1 = p1
            cv2.rectangle(d, (min(x0,x1),min(y0,y1)), (max(x0,x1),max(y0,y1)), (0,255,0), 2)
        txt = "Drag to draw; release to confirm. [C]clear  [Q]quit"
        cv2.putText(d, txt, (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        cv2.putText(d, txt, (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        return d

    def on_mouse(e,x,y,flags,param):
        nonlocal p0,p1,dragging
        if e == cv2.EVENT_LBUTTONDOWN:
            p0 = (int(x),int(y)); p1 = p0; dragging = True
        elif e == cv2.EVENT_MOUSEMOVE and dragging:
            p1 = (int(x),int(y))
        elif e == cv2.EVENT_LBUTTONUP and dragging:
            p1 = (int(x),int(y)); dragging = False
            cv2.destroyWindow(win)

    cv2.setMouseCallback(win, on_mouse)

    while True:
        show_live = dragging
        disp = refresh(show_live)
        if p0 is not None and p1 is not None and not dragging:
            x0,y0 = p0; x1,y1 = p1
            x0,x1 = sorted((max(0,x0), max(0,x1)))
            y0,y1 = sorted((max(0,y0), max(0,y1)))
            return (x0,y0,x1,y1) if (x1>x0 and y1>y0) else None

        cv2.imshow(win, disp)
        k = cv2.waitKey(10) & 0xFF
        if k in (ord('q'), 27):
            cv2.destroyWindow(win); return None
        if k == ord('c'):
            p0 = p1 = None; dragging = False

# ----------------------------
# Geometry helpers
# ----------------------------
def _minarect_angle_deg(poly: np.ndarray) -> float:
    rect = cv2.minAreaRect(poly.astype(np.float32))
    ang = rect[2]
    if rect[1][0] < rect[1][1]:
        ang += 90.0
    return (ang + 360.0) % 360.0

def _rotate_image_and_mask(img_bgr: np.ndarray, mask: np.ndarray, angle_deg: float):
    (h,w) = img_bgr.shape[:2]
    c = (w/2.0, h/2.0)
    M = cv2.getRotationMatrix2D(c, angle_deg, 1.0)
    rot = cv2.warpAffine(img_bgr, M, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    rmask = cv2.warpAffine(mask, M, (w,h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    return rot, rmask, M

def _crop_to_content(img_bgr: np.ndarray, mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0: return img_bgr, mask, (0,0)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return img_bgr[y0:y1+1, x0:x1+1], mask[y0:y1+1, x0:x1+1], (x0,y0)

def _poly_to_alpha_template(img_bgr: np.ndarray, polygon: List[Tuple[int,int]], make_upright=True):
    poly = np.array(polygon[:-1], np.int32) if polygon[0]==polygon[-1] else np.array(polygon, np.int32)
    mask = np.zeros(img_bgr.shape[:2], np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    cut = img_bgr.copy(); cut[mask==0] = 0

    angle = _minarect_angle_deg(poly) if make_upright else 0.0
    rot, rmask, M = _rotate_image_and_mask(cut, mask, -angle)
    crop, cmask, (x0,y0) = _crop_to_content(rot, rmask)

    alpha = (cmask>0).astype(np.uint8)*255
    rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    rgba[:,:,3] = alpha

    poly_h = np.hstack([poly.astype(np.float32), np.ones((len(poly),1), np.float32)])
    rot_poly = (M @ poly_h.T).T
    rot_poly = rot_poly - np.array([[x0,y0]], dtype=np.float32)
    return rgba, rot_poly, angle, M, (x0, y0)

def _select_two_points(img: np.ndarray,
                       hint: str = "Click GRIP then DIRECTION  [L:pick  C:clear  R:swap  S/Enter:save  Q:quit]"):
    """
    Single window: first click = GRIP, second click = DIRECTION.
    Keys:
      C: clear both
      R: swap points
      S/Enter: save (requires both points)
      Q/Esc: cancel (return None)
    Returns: ((gx,gy), (dx,dy)) or None
    """
    p1 = None  # GRIP
    p2 = None  # DIRECTION
    win = "pick two points"
    disp = img.copy()

    def draw_overlay():
        d = img.copy()
        cv2.putText(d, hint, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 2)
        cv2.putText(d, hint, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 1)
        if p1 is not None:
            cv2.circle(d, p1, 6, (0,255,255), -1)
            cv2.putText(d, "GRIP", (p1[0]+8, p1[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
            cv2.putText(d, "GRIP", (p1[0]+8, p1[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
        if p2 is not None:
            cv2.circle(d, p2, 6, (0,0,255), -1)
            cv2.putText(d, "DIR", (p2[0]+8, p2[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
            cv2.putText(d, "DIR", (p2[0]+8, p2[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
        if p1 is not None and p2 is not None:
            cv2.line(d, p1, p2, (255,255,0), 2)
        return d

    def on_mouse(e, x, y, flags, param):
        nonlocal p1, p2, disp
        if e == cv2.EVENT_LBUTTONDOWN:
            if p1 is None:
                p1 = (int(x), int(y))
            elif p2 is None:
                p2 = (int(x), int(y))
            else:
                p2 = (int(x), int(y))
            disp = draw_overlay()

    cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow(win, 1280, 720)
    cv2.setMouseCallback(win, on_mouse)
    disp = draw_overlay()

    while True:
        cv2.imshow(win, disp)
        k = cv2.waitKey(10) & 0xFF
        if k in (ord('q'), 27):
            cv2.destroyWindow(win); return None
        elif k == ord('c'):
            p1 = None; p2 = None; disp = draw_overlay()
        elif k == ord('r'):
            if p1 is not None and p2 is not None:
                p1, p2 = p2, p1; disp = draw_overlay()
        elif k in (ord('s'), 13):  # S or Enter
            if p1 is not None and p2 is not None:
                cv2.destroyWindow(win)
                return (p1, p2)

# -----------------
# UI: single point
# -----------------
def _select_point(img: np.ndarray, hint: str = "select point (L:pick, S/Enter:save, Q:quit)") -> Optional[Tuple[int,int]]:
    p: Optional[Tuple[int,int]] = None
    disp = img.copy()
    win = "point"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow(win, 1280, 720)

    def refresh():
        nonlocal disp
        disp = img.copy()
        txt = hint
        cv2.putText(disp, txt, (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        cv2.putText(disp, txt, (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        if p is not None:
            cv2.circle(disp, p, 5, (0,255,255), -1)

    def on_mouse(e,x,y,flags,param):
        nonlocal p
        if e == cv2.EVENT_LBUTTONDOWN:
            p = (int(x), int(y))
            refresh()

    refresh()
    cv2.setMouseCallback(win, on_mouse)
    while True:
        cv2.imshow(win, disp)
        k = cv2.waitKey(10) & 0xFF
        if k in (ord('q'), 27):
            cv2.destroyWindow(win); return None
        if k in (13, ord('s')):   # Enter or S
            cv2.destroyWindow(win)
            return p

# -----------------------------------
# Rect-mode template (drag) + points
# -----------------------------------
def synth_init_rect(product_id: int, image_path: str | None = None) -> dict:
    store = ProductStore()
    if image_path:
        src = Path(image_path)
    else:
        imgs = [p for p in store.images_dir(product_id).glob("*.*")
                if p.suffix.lower() in (".jpg",".jpeg",".png",".bmp")]
        if not imgs:
            raise FileNotFoundError("No image found; please capture/add a product photo first.")
        src = imgs[0]

    img = cv2.imread(str(src))
    if img is None:
        raise RuntimeError(f"Could not read image: {src}")

    # 1) rectangle by drag
    rect = _select_rect_drag(img)
    if rect is None:
        raise RuntimeError("No rectangle was selected.")
    x0, y0, x1, y1 = rect
    x0 = max(0, min(x0, img.shape[1]-1))
    x1 = max(0, min(x1, img.shape[1]-1))
    y0 = max(0, min(y0, img.shape[0]-1))
    y1 = max(0, min(y1, img.shape[0]-1))
    if x1 <= x0 or y1 <= y0:
        raise RuntimeError("Invalid rectangle.")

    # 2) pick GRIP then DIRECTION on same window
    picked = _select_two_points(img, "Click GRIP then DIRECTION  [L:pick  C:clear  R:swap  S:save  Q:quit]")
    if picked is None:
        raise RuntimeError("Point picking cancelled.")
    gp, dp = picked

    # build rectangular alpha template (opaque)
    crop = img[y0:y1+1, x0:x1+1].copy()
    h, w = crop.shape[:2]
    rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA); rgba[:, :, 3] = 255

    # to template coords
    gp_tmpl = (float(gp[0] - x0), float(gp[1] - y0))
    dp_tmpl = (float(dp[0] - x0), float(dp[1] - y0))

    # axis for robustness + align sign to grip->dir
    alpha = rgba[:, :, 3]
    axis_deg = _principal_axis_deg_from_alpha(alpha)
    v_gd = np.array([dp_tmpl[0] - gp_tmpl[0], dp_tmpl[1] - gp_tmpl[1]], np.float32)
    v_ax = np.array([math.cos(math.radians(axis_deg)), math.sin(math.radians(axis_deg))], np.float32)
    if v_ax.dot(v_gd) < 0:
        axis_deg = _wrap_deg(axis_deg + 180.0)

    # save
    tdir = DATA_DIR / "templates" / f"product_{product_id}"
    tdir.mkdir(parents=True, exist_ok=True)
    png = tdir / "template.png"
    meta = tdir / "template.json"
    cv2.imwrite(str(png), rgba, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])

    meta.write_text(json.dumps({
        "product_id": product_id,
        "angle_ref_deg": 0.0,
        "template_size": [int(w), int(h)],
        "polygon": [[0.0,0.0],[w,0.0],[w,h],[0.0,h]],
        "grip_point": [float(gp_tmpl[0]), float(gp_tmpl[1])],
        "dir_point":  [float(dp_tmpl[0]), float(dp_tmpl[1])],
        "axis_deg": float(axis_deg),
        "ref_off_deg": 0.0
    }, ensure_ascii=True, indent=2), encoding="utf-8")

    out = {
        "template_png": str(png),
        "template_json": str(meta),
        "angle_ref_deg": 0.0,
        "grip_point": [float(gp_tmpl[0]), float(gp_tmpl[1])],
        "dir_point":  [float(dp_tmpl[0]), float(dp_tmpl[1])],
        "mode": "rect"
    }
    return out

# -----------------------------------
# Polygon-mode template + points (first photo == 0 deg reference)
# -----------------------------------
def synth_init(product_id: int, image_path: str | None = None) -> dict:
    store = ProductStore()
    if image_path:
        src = Path(image_path)
    else:
        imgs = [p for p in store.images_dir(product_id).glob("*.*")
                if p.suffix.lower() in (".jpg",".jpeg",".png",".bmp")]
        if not imgs:
            raise FileNotFoundError("No image found; please capture/add a product photo first.")
        src = imgs[0]

    img = cv2.imread(str(src))
    if img is None:
        raise RuntimeError(f"Could not read image: {src}")

    # 1) polygon
    poly = _select_polygon(img)
    if not poly:
        raise RuntimeError("No polygon was selected.")

    rgba, canon_poly, angle_ref, M, (x0,y0) = _poly_to_alpha_template(img, poly, make_upright=False)

    # 2) pick GRIP then DIRECTION on same window
    picked = _select_two_points(img, "Click GRIP then DIRECTION  [L:pick  C:clear  R:swap  S:save  Q:quit]")
    if picked is None:
        raise RuntimeError("Point picking cancelled.")
    gp, dp = picked

    # to template coords using same transform M
    gp_h = np.array([[gp[0], gp[1], 1.0]], dtype=np.float32).T
    dp_h = np.array([[dp[0], dp[1], 1.0]], dtype=np.float32).T
    gp_rot = (M @ gp_h).T[0]; dp_rot = (M @ dp_h).T[0]
    gp_tmpl = (float(gp_rot[0] - x0), float(gp_rot[1] - y0))
    dp_tmpl = (float(dp_rot[0] - x0), float(dp_rot[1] - y0))

    # axis + align sign to grip->dir
    alpha = rgba[:, :, 3]
    axis_deg = _principal_axis_deg_from_alpha(alpha)
    v_gd = np.array([dp_tmpl[0] - gp_tmpl[0], dp_tmpl[1] - gp_tmpl[1]], np.float32)
    v_ax = np.array([math.cos(math.radians(axis_deg)), math.sin(math.radians(axis_deg))], np.float32)
    if v_ax.dot(v_gd) < 0:
        axis_deg = _wrap_deg(axis_deg + 180.0)

    # save
    tdir = DATA_DIR / "templates" / f"product_{product_id}"
    tdir.mkdir(parents=True, exist_ok=True)
    png = tdir / "template.png"
    meta = tdir / "template.json"
    cv2.imwrite(str(png), rgba, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])

    meta.write_text(json.dumps({
        "product_id": product_id,
        "angle_ref_deg": 0.0,
        "template_size": [int(rgba.shape[1]), int(rgba.shape[0])],
        "polygon": np.asarray(canon_poly, dtype=float).tolist(),
        "grip_point": [float(gp_tmpl[0]), float(gp_tmpl[1])],
        "dir_point":  [float(dp_tmpl[0]), float(dp_tmpl[1])],
        "axis_deg": float(axis_deg),
        "ref_off_deg": 0.0
    }, ensure_ascii=True, indent=2), encoding="utf-8")

    return {
        "template_png": str(png),
        "template_json": str(meta),
        "angle_ref_deg": 0.0,
        "grip_point": [float(gp_tmpl[0]), float(gp_tmpl[1])],
        "dir_point":  [float(dp_tmpl[0]), float(dp_tmpl[1])],
        "mode": "polygon"
    }


# -----------------------------------
# Synthetic dataset generator
# -----------------------------------
def synth_make(product_id: int, count: int = 1200, bg_dir: str | None = None,
               out_w: int = 1280, out_h: int = 720) -> dict:
    store = ProductStore()
    tdir = DATA_DIR / "templates" / f"product_{product_id}"
    png  = tdir / "template.png"
    meta = tdir / "template.json"
    if not png.exists() or not meta.exists():
        raise FileNotFoundError("Template files not found; run synth_init first.")

    M = json.loads(meta.read_text(encoding="utf-8"))
    grip = M.get("grip_point", [None, None])
    dirp = M.get("dir_point", None)
    axis_deg = float(M.get("axis_deg", 0.0))
    if grip is None or grip[0] is None or grip[1] is None:
        raise RuntimeError("Missing grip_point in template.json")

    tmpl_rgba = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
    if tmpl_rgba is None:
        raise RuntimeError(f"Could not read template: {png}")

    # backgrounds
    bgs: List[Path] = []
    if bg_dir:
        p = Path(bg_dir)
        if p.is_file():
            bgs = [p]
        elif p.is_dir():
            bgs = [x for x in p.glob("*.*") if x.suffix.lower() in (".jpg",".jpeg",".png",".bmp")]
    if not bgs:
        cal = calibration_path()
        if cal.exists():
            bgs = [cal]

    images_dir = store.images_dir(product_id)
    labels_dir = store.labels_dir(product_id)
    (images_dir / "train").mkdir(parents=True, exist_ok=True)
    (labels_dir / "train").mkdir(parents=True, exist_ok=True)
    (images_dir / "val").mkdir(parents=True, exist_ok=True)
    (labels_dir / "val").mkdir(parents=True, exist_ok=True)

    def paste_rgba(bg_bgr, fg_rgba, cx, cy, angle, scale):
        h, w = fg_rgba.shape[:2]

        # scale
        sw = max(1, int(round(w * scale)))
        sh = max(1, int(round(h * scale)))
        fg = cv2.resize(fg_rgba, (sw, sh), interpolation=cv2.INTER_LINEAR)

        # rotation matrix + new fit size (no crop)
        c = (sw/2.0, sh/2.0)
        Mrot = cv2.getRotationMatrix2D(c, angle, 1.0)
        abs_cos = abs(Mrot[0,0]); abs_sin = abs(Mrot[0,1])
        rw = int(sh * abs_sin + sw * abs_cos)
        rh = int(sh * abs_cos + sw * abs_sin)
        Mrot[0,2] += (rw/2.0 - c[0])
        Mrot[1,2] += (rh/2.0 - c[1])

        # rotate with transparent border
        rot = cv2.warpAffine(
            fg, Mrot, (rw, rh),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )

        # place on background
        out = bg_bgr.copy()
        x0 = int(round(cx - rw/2.0)); y0 = int(round(cy - rh/2.0))
        x1 = x0 + rw;                  y1 = y0 + rh

        # intersection with canvas
        bx0, by0, bx1, by1 = 0, 0, out.shape[1], out.shape[0]
        rx0, ry0 = max(x0, bx0), max(y0, by0)
        rx1, ry1 = min(x1, bx1), min(y1, by1)
        if rx1 <= rx0 or ry1 <= ry0:
            return out, Mrot, (x0,y0,x1,y1), None

        sub = out[ry0:ry1, rx0:rx1]
        u0, v0 = rx0 - x0, ry0 - y0
        u1, v1 = u0 + (rx1 - rx0), v0 + (ry1 - ry0)
        crop = rot[v0:v1, u0:u1]

        # alpha blend
        if crop.shape[2] == 4:
            alpha = crop[:,:,3:4].astype(np.float32) / 255.0
            sub[:] = (1.0 - alpha) * sub.astype(np.float32) + alpha * crop[:,:,:3].astype(np.float32)
            sub[:] = np.clip(sub, 0, 255).astype(np.uint8)
        else:
            sub[:] = crop

        # tight bbox from alpha
        tight_rect = None
        if crop.shape[2] == 4:
            a = crop[:,:,3]
            ys, xs = np.where(a > 0)
            if len(xs) > 0:
                gx0bb = rx0 + int(xs.min())
                gy0bb = ry0 + int(ys.min())
                gx1bb = rx0 + int(xs.max())
                gy1bb = ry0 + int(ys.max())
                tight_rect = (gx0bb, gy0bb, gx1bb, gy1bb)

        return out, Mrot, (x0,y0,x1,y1), tight_rect

    created_train = created_val = 0
    for i in range(count):
        bgp = random.choice(bgs) if bgs else None
        if bgp is not None:
            bg = cv2.imread(str(bgp))
            if bg is None:
                bg = np.zeros((out_h, out_w, 3), np.uint8)
            else:
                bg = cv2.resize(bg, (out_w, out_h), interpolation=cv2.INTER_AREA)
        else:
            bg = np.zeros((out_h, out_w, 3), np.uint8)

        # random pose
        cx = random.randint(int(0.25*out_w), int(0.75*out_w))
        cy = random.randint(int(0.25*out_h), int(0.75*out_h))
        angle = random.uniform(-180, 180)
        scale = random.uniform(0.98, 1.02)

        # rotate & place
        comp, Mrot, env_rect, tight_rect = paste_rgba(bg, tmpl_rgba, cx, cy, angle, scale)

        # bbox: tight preferred
        if tight_rect is not None:
            bx0, by0, bx1, by1 = tight_rect
        else:
            bx0, by0, bx1, by1 = env_rect

        # yolo bbox normalized
        nx = ((bx0 + bx1) / 2.0) / float(out_w)
        ny = ((by0 + by1) / 2.0) / float(out_h)
        nw = (bx1 - bx0) / float(out_w)
        nh = (by1 - by0) / float(out_h)

        # keypoint 1 (grip)
        gx0, gy0 = float(grip[0]), float(grip[1])
        gx_s, gy_s = gx0 * scale, gy0 * scale
        gx_rot = Mrot[0,0]*gx_s + Mrot[0,1]*gy_s + Mrot[0,2]
        gy_rot = Mrot[1,0]*gx_s + Mrot[1,1]*gy_s + Mrot[1,2]
        x0_env, y0_env, _, _ = env_rect
        gabs_x = x0_env + gx_rot
        gabs_y = y0_env + gy_rot
        k1x = min(1.0, max(0.0, gabs_x / float(out_w)))
        k1y = min(1.0, max(0.0, gabs_y / float(out_h)))

        # keypoint 2 (direction) if exists
        if dirp is not None and len(dirp) == 2:
            dx0, dy0 = float(dirp[0]), float(dirp[1])
            dx_s, dy_s = dx0 * scale, dy0 * scale
            dx_rot = Mrot[0,0]*dx_s + Mrot[0,1]*dy_s + Mrot[0,2]
            dy_rot = Mrot[1,0]*dx_s + Mrot[1,1]*dy_s + Mrot[1,2]
            dabs_x = x0_env + dx_rot
            dabs_y = y0_env + dy_rot
            k2x = min(1.0, max(0.0, dabs_x / float(out_w)))
            k2y = min(1.0, max(0.0, dabs_y / float(out_h)))

            line = f"0 {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f} {k1x:.6f} {k1y:.6f} 2 {k2x:.6f} {k2y:.6f} 2\n"
        else:
            line = f"0 {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f} {k1x:.6f} {k1y:.6f} 2\n"

        # clamp
        nx = min(1.0, max(0.0, nx)); ny = min(1.0, max(0.0, ny))
        nw = min(1.0, max(0.0, nw)); nh = min(1.0, max(0.0, nh))

        # output
        split = "train" if i < int(count*0.9) else "val"
        ip = images_dir / split / f"img_{i:06d}.jpg"
        lp = labels_dir / split / f"img_{i:06d}.txt"
        ip.parent.mkdir(parents=True, exist_ok=True)
        lp.parent.mkdir(parents=True, exist_ok=True)
        assert isinstance(comp, np.ndarray), type(comp)
        write_jpeg(comp, ip, quality=95)
        lp.write_text(line, encoding="utf-8")

        if split == "train": created_train += 1
        else: created_val += 1

    return {"created_train": created_train, "created_val": created_val}
