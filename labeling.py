# -*- coding: utf-8 -*-
# yolovizi_core/labeling.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np

from .storage import ProductStore

def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))

def box_to_yolo(x: int, y: int, w: int, h: int, img_w: int, img_h: int, class_id: int = 0) -> str:
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    ww = w / img_w
    hh = h / img_h
    return f"{class_id} {clamp01(cx):.6f} {clamp01(cy):.6f} {clamp01(ww):.6f} {clamp01(hh):.6f}"

def yolo_to_box(line: str, img_w: int, img_h: int) -> Tuple[int,int,int,int,int]:
    # returns (class_id, x, y, w, h) in pixels
    parts = line.strip().split()
    if len(parts) != 5:
        raise ValueError("Bad YOLO line")
    c, cx, cy, ww, hh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    w = int(round(ww * img_w))
    h = int(round(hh * img_h))
    x = int(round(cx * img_w - w / 2.0))
    y = int(round(cy * img_h - h / 2.0))
    return c, x, y, w, h

class Labeler:
    # """
    # - Görüntü baþýna çoklu kutu (ayný class_id=0)
    # - Modlar:
    #     * Sürükle-Býrak kutu (LBUTTON down->move->up)
    #     * Çoklu nokta (sol týk noktalar; sað týk kapat ? bbox)
    #   'M' ile mod deðiþtir.
    # - Kýsayollar:
    #     S: Kaydet & sonlandýr (mevcut görüntü)
    #     N: Kaydet & sonraki görüntü (folder modunda)
    #     U: Son kutuyu geri al
    #     C: Tüm kutularý temizle
    #     M: Mod deðiþtir (drag / poly)
    #     ESC/Q: Ýptal/çýk
    # """
    def __init__(self, class_id: int = 0):
        self.class_id = class_id
        self.mode = "drag"   # "drag" | "poly"
        self.boxes: List[Tuple[int,int,int,int]] = []
        self._drawing = False
        self._start = (0,0)
        self._poly_pts: List[Tuple[int,int]] = []
        self._img = None
        self._img_disp = None

    def _refresh(self):
        disp = self._img.copy()
        for (x,y,w,h) in self.boxes:
            cv2.rectangle(disp, (x,y), (x+w,y+h), (0,255,0), 2)
        if self.mode == "poly" and self._poly_pts:
            for i,p in enumerate(self._poly_pts):
                cv2.circle(disp, p, 3, (0,165,255), -1)
                if i>0:
                    cv2.line(disp, self._poly_pts[i-1], p, (0,165,255), 1)
        txt = f"[{self.mode}]  S:save  N:save+next  U:undo  C:clear  M:mode  Q:quit"
        cv2.putText(disp, txt, (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,50,50), 2)
        cv2.putText(disp, txt, (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        self._img_disp = disp

    def _on_mouse(self, event, x, y, flags, param):
        if self.mode == "drag":
            if event == cv2.EVENT_LBUTTONDOWN:
                self._drawing = True
                self._start = (x,y)
            elif event == cv2.EVENT_MOUSEMOVE and self._drawing:
                self._refresh()
                x0,y0 = self._start
                cv2.rectangle(self._img_disp, (x0,y0), (x,y), (0,0,255), 2)
            elif event == cv2.EVENT_LBUTTONUP and self._drawing:
                self._drawing = False
                x0,y0 = self._start
                x1,y1 = x,y
                x_min, y_min = min(x0,x1), min(y0,y1)
                w,h = abs(x1-x0), abs(y1-y0)
                if w>3 and h>3:
                    self.boxes.append((x_min,y_min,w,h))
                self._refresh()
        else:
            if event == cv2.EVENT_LBUTTONDOWN:
                self._poly_pts.append((x,y))
                self._refresh()
            elif event == cv2.EVENT_RBUTTONDOWN:
                if len(self._poly_pts) >= 2:
                    xs = [p[0] for p in self._poly_pts]
                    ys = [p[1] for p in self._poly_pts]
                    x_min,y_min = min(xs),min(ys)
                    x_max,y_max = max(xs),max(ys)
                    w,h = x_max-x_min, y_max-y_min
                    if w>3 and h>3:
                        self.boxes.append((x_min,y_min,w,h))
                self._poly_pts.clear()
                self._refresh()

    def annotate(self, img: np.ndarray, win_name="label") -> Optional[List[Tuple[int,int,int,int]]]:
        self._img = img.copy()
        self.boxes.clear()
        self._poly_pts.clear()
        self._drawing = False
        self._refresh()

        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
        cv2.resizeWindow(win_name, 1280, 720)
        cv2.setMouseCallback(win_name, self._on_mouse)

        while True:
            cv2.imshow(win_name, self._img_disp)
            k = cv2.waitKey(10) & 0xFF
            if k == ord('q') or k == 27:  # ESC/Q
                cv2.destroyWindow(win_name)
                return None
            if k == ord('u'):
                if self.boxes: self.boxes.pop()
                self._refresh()
            if k == ord('c'):
                self.boxes.clear(); self._poly_pts.clear(); self._refresh()
            if k == ord('m'):
                self.mode = "poly" if self.mode=="drag" else "drag"
                self._poly_pts.clear()
                self._refresh()
            if k == ord('s') or k == ord('n'):
                cv2.destroyWindow(win_name)
                return self.boxes

def label_images_in_folder(product_id: int, images_dir: Optional[str] = None):
    store = ProductStore()
    img_dir = Path(images_dir) if images_dir else store.images_dir(product_id)
    lbl_dir = store.labels_dir(product_id)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in img_dir.glob("*.*") if p.suffix.lower() in (".jpg",".jpeg",".png",".bmp")])
    if not images:
        print(f"[label] not Found: {img_dir}")
        return

    lab = Labeler(class_id=0)
    idx = 0
    while 0 <= idx < len(images):
        img_path = images[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[skip] not read: {img_path.name}")
            idx += 1
            continue

        lbl_path = lbl_dir / (img_path.stem + ".txt")
        pre_boxes: List[Tuple[int,int,int,int]] = []
        if lbl_path.exists():
            try:
                lines = [l for l in lbl_path.read_text().splitlines() if l.strip()]
                for line in lines:
                    _, x, y, w, h = yolo_to_box(line, img.shape[1], img.shape[0])
                    pre_boxes.append((x,y,w,h))
            except Exception:
                pass

        lab.boxes = pre_boxes[:]  # type: ignore
        out = lab.annotate(img, win_name=f"label - {img_path.name}")
        if out is None:
            break

        lines = [box_to_yolo(x,y,w,h, img.shape[1], img.shape[0], class_id=0) for (x,y,w,h) in out]
        lbl_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        print(f"[save] {lbl_path.name} ({len(lines)} kutu)")

        idx += 1

    cv2.destroyAllWindows()
    print("? Etiketleme bitti")
