from __future__ import annotations
import argparse
import base64
import json
import sys
import time
from typing import Any, Dict, Optional

import numpy as np

try:
    import cv2  # optional (watch --annotated)
except Exception:
    cv2 = None  # type: ignore

from . import bridge as _bridge
from .tracking import Tracker  # used by watch
# direct fallbacks
from . import labeling as _labeling
from . import synth as _synth


def _print_json(obj: Any) -> None:
    print(json.dumps(obj))


def _eprint(s: str) -> None:
    print(s, file=sys.stderr, flush=True)


# ---------- commands ----------
def cmd_probe(args: argparse.Namespace) -> None:
    fn = getattr(_bridge, "capture_background", None)
    if fn is None:
        raise RuntimeError("bridge.capture_background not found")
    path = fn()
    _print_json({"ok": True, "background": path})


def cmd_capture(args: argparse.Namespace) -> None:
    fn = getattr(_bridge, "capture_product_photo", None)
    if fn is None:
        raise RuntimeError("bridge.capture_product_photo not found")
    pid = int(args.product)
    path = fn(pid)
    _print_json({"ok": True, "image": path})


def cmd_products_list(args: argparse.Namespace) -> None:
    fn = getattr(_bridge, "list_products", None)
    if fn is None:
        raise RuntimeError("bridge.list_products not found")
    _print_json({"products": fn()})


def cmd_products_create(args: argparse.Namespace) -> None:
    fn = getattr(_bridge, "create_product", None)
    if fn is None:
        raise RuntimeError("bridge.create_product not found")
    prod = fn(args.name)
    _print_json({"product": prod})


def cmd_label_folder(args: argparse.Namespace) -> None:
    pid = int(args.product)
    # prefer bridge.label_images; fall back to labeling.label_images_in_folder
    fn = getattr(_bridge, "label_images", None) or getattr(_labeling, "label_images_in_folder", None)
    if fn is None:
        raise RuntimeError("label_images not found")
    folder = args.folder or None
    res = fn(pid, folder) if folder is not None else fn(pid)
    _print_json(res)


def cmd_label_camera(args: argparse.Namespace) -> None:
    pid = int(args.product)
    # prefer bridge.label_camera; fall back to labeling.label_from_camera
    fn = getattr(_bridge, "label_camera", None) or getattr(_labeling, "label_from_camera", None)
    if fn is None:
        raise RuntimeError("label_camera/label_from_camera not found")
    res = fn(pid)
    _print_json(res)


def cmd_synth_init(args: argparse.Namespace) -> None:
    pid = int(args.product)
    # prefer bridge.synth_init_poly; fall back to synth.synth_init
    fn = getattr(_bridge, "synth_init_poly", None) or getattr(_synth, "synth_init", None)
    if fn is None:
        raise RuntimeError("synth_init not found")
    img = args.image or None
    res = fn(pid, img) if img is not None else fn(pid)
    _print_json(res)


def cmd_synth_init_rect(args: argparse.Namespace) -> None:
    pid = int(args.product)
    # prefer bridge.synth_init_rect; fall back to synth.synth_init_rect
    fn = getattr(_bridge, "synth_init_rect", None) or getattr(_synth, "synth_init_rect", None)
    if fn is None:
        raise RuntimeError("synth_init_rect not found")
    img = args.image or None
    res = fn(pid, img) if img is not None else fn(pid)
    _print_json(res)


def cmd_synth_make(args: argparse.Namespace) -> None:
    pid = int(args.product)
    count = int(args.count)
    size = args.size.lower()
    w, h = map(int, size.split("x"))
    bg = args.bg or None
    # prefer bridge.synth_make_data; fall back to synth.synth_make
    fn = getattr(_bridge, "synth_make_data", None) or getattr(_synth, "synth_make", None)
    if fn is None:
        raise RuntimeError("synth_make not found")
    res = fn(pid, count=count, bg_dir=bg, out_w=w, out_h=h) 
    _print_json(res)


def cmd_train(args: argparse.Namespace) -> None:
    pid = int(args.product)
    fn = getattr(_bridge, "train", None)
    if fn is None:
        raise RuntimeError("bridge.train not found")
    dataset = args.dataset or ""
    out = fn(pid, dataset)
    _print_json(out)


def cmd_track_start(args: argparse.Namespace) -> None:
    pid = int(args.product)
    fn = getattr(_bridge, "track_start", None)
    if fn is None:
        raise RuntimeError("bridge.track_start not found")
    fn(pid)
    _print_json({"ok": True, "product": pid})


def cmd_track_stop(args: argparse.Namespace) -> None:
    fn = getattr(_bridge, "track_stop", None)
    if fn is None:
        raise RuntimeError("bridge.track_stop not found")
    fn()
    _print_json({"ok": True})


def cmd_track_status(args: argparse.Namespace) -> None:
    fn = getattr(_bridge, "track_status", None)
    if fn is None:
        raise RuntimeError("bridge.track_status not found")
    st = fn()
    # short status to stderr for live UIs
    _eprint(json.dumps({
        "type": "status",
        "detected": st.get("detected"),
        "angle_deg": st.get("angle_deg"),
        "confidence": st.get("confidence"),
    }))
    # full JSON to stdout
    _print_json(st)


def cmd_track_getframe(args: argparse.Namespace) -> None:
    fn = getattr(_bridge, "get_frame", None)
    if fn is None:
        raise RuntimeError("bridge.get_frame not found")
    jpg = fn()
    if not jpg:
        _print_json({"ok": False, "reason": "no_frame"})
        return
    print(base64.b64encode(jpg).decode("ascii"), flush=True)

def _draw_annotated(jpg: bytes, status: Dict[str, Any]) -> bytes:
    if cv2 is None:
        return jpg
    try:
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jpg
        if status.get("detected"):
            x = int(status.get("bbox_x", 0))
            y = int(status.get("bbox_y", 0))
            w = int(status.get("bbox_w", 0))
            h = int(status.get("bbox_h", 0))
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)

            gx = status.get("grip_x"); gy = status.get("grip_y")
            if gx is not None and gy is not None:
                cv2.circle(img, (int(gx), int(gy)), 4, (0, 255, 0), -1)

            ang = status.get("angle_deg")
            gwx = status.get("grip_wx"); gwy = status.get("grip_wy")
            cws = status.get("corners_w") or []
            lines = []
            if ang is not None:
                lines.append(f"angle: {float(ang):.1f} deg")
            if gwx is not None and gwy is not None:
                lines.append(f"grip (w): {float(gwx):.0f}, {float(gwy):.0f}")
            if len(cws) == 4:
                a,b = cws[0]; c,d = cws[1]; e,f = cws[2]; g,h2 = cws[3]
                lines.append(f"A(w): {a:.0f},{b:.0f}")
                lines.append(f"B(w): {c:.0f},{d:.0f}")
                lines.append(f"C(w): {e:.0f},{f:.0f}")
                lines.append(f"D(w): {g:.0f},{h2:.0f}")

            pad, lh = 6, 18
            W = max([cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] for s in lines] + [0]) + 2*pad
            H = lh*len(lines) + 2*pad
            bx, by = max(0, x-4), max(0, y- H - 8)
            cv2.rectangle(img, (bx, by), (bx+W, by+H), (30,30,30), -1)
            ytxt = by + pad + 14
            for s in lines:
                cv2.putText(img, s, (bx+pad, ytxt), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                ytxt += lh

        ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return enc.tobytes() if ok else jpg
    except Exception:
        return jpg

def cmd_watch(args: argparse.Namespace) -> None:
    """
    Live stream: frames -> stdout (base64), status -> stderr (NDJSON).
    """
    pid = int(args.product)
    fps = int(args.fps) if args.fps else 0
    period = 1.0 / fps if fps > 0 else 0.0
    emit_frames = bool(args.emit_frames)
    emit_status = bool(args.emit_status)
    status_every = float(args.status_every)
    status_to_stderr = (args.status_channel == "stderr")
    annotated = bool(args.annotated)

    tr = Tracker()
    tr.start(pid)
    try:
        next_t = 0.0
        last_status = 0.0
        while True:
            now = time.time()
            s = tr.status()


            if emit_status and (now - last_status) >= status_every:
                line = json.dumps({"type": "status", **s})
                if status_to_stderr: _eprint(line)
                else: print(line, flush=True)
                last_status = now

            if emit_frames:
                jpg = tr.get_jpeg()
                if jpg:
                    if annotated:
                        jpg = _draw_annotated(jpg, s)
                    print(base64.b64encode(jpg).decode("ascii"), flush=True)

            if period > 0.0:
                next_t = (next_t or now) + period
                sleep = max(0.0, next_t - time.time())
                if sleep > 0.0:
                    time.sleep(sleep)
            else:
                time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        tr.stop()

def cmd_sync(args: argparse.Namespace) -> None:
    fn = getattr(_bridge, "sync_camera", None)
    if fn is None:
        raise RuntimeError("bridge.sync_camera not found")
    ok = fn()
    _print_json({"ok": ok})

def cmd_get_product_image(args: argparse.Namespace) -> None:
    fn = getattr(_bridge, "get_product_image", None)
    if fn is None:
        raise RuntimeError("bridge.get_product_image not found")
    path = fn(int(args.product))
    _print_json(str(path)) 

# ---------- parser ----------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("yolovizi_core", add_help=True)
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("probe", help="quick camera/pylon check")
    pr.set_defaults(func=cmd_probe)

    cp = sub.add_parser("capture", help="capture one product photo")
    cp.add_argument("--product", required=True)
    cp.set_defaults(func=cmd_capture)

    pl = sub.add_parser("products", help="product management")
    psub = pl.add_subparsers(dest="action", required=True)
    pll = psub.add_parser("list", help="list products")
    pll.set_defaults(func=cmd_products_list)
    plc = psub.add_parser("create", help="create product")
    plc.add_argument("--name", required=True)
    plc.set_defaults(func=cmd_products_create)

    lb = sub.add_parser("label", help="label images")
    lsub = lb.add_subparsers(dest="action", required=True)
    lbf = lsub.add_parser("folder", help="label from folder")
    lbf.add_argument("--product", required=True)
    lbf.add_argument("--folder", required=False, help="folder path (optional)")
    lbf.set_defaults(func=cmd_label_folder)
    lbc = lsub.add_parser("camera", help="label from camera")
    lbc.add_argument("--product", required=True)
    lbc.set_defaults(func=cmd_label_camera)

    syi = sub.add_parser("synth-init", help="init synthetic template (polygon mode)")
    syi.add_argument("--product", required=True)
    syi.add_argument("--image", required=False, help="seed image (optional)")
    syi.set_defaults(func=cmd_synth_init)

    syir = sub.add_parser("synth-init-rect", help="init rectangular synthetic template")
    syir.add_argument("--product", required=True)
    syir.add_argument("--image", required=False, help="seed image (optional)")
    syir.set_defaults(func=cmd_synth_init_rect)

    sym = sub.add_parser("synth", help="generate synthetic images")
    sym.add_argument("--product", required=True)
    sym.add_argument("--count", required=True, help="number of images")
    sym.add_argument("--size", default="640x640", help="WxH (e.g. 640x640)")
    sym.add_argument("--bg", default="", help="background file/folder (optional)")
    sym.add_argument("--angle-center", type=float, default=0.0) 
    sym.add_argument("--angle-jitter", type=float, default=15.0) 
    sym.set_defaults(func=cmd_synth_make)

    trn = sub.add_parser("train", help="train model")
    trn.add_argument("--product", required=True)
    trn.add_argument("--dataset", default="", help="custom images folder (optional)")
    trn.set_defaults(func=cmd_train)

    ts = sub.add_parser("track", help="live tracking controls")
    tss = ts.add_subparsers(dest="action", required=True)
    tstart = tss.add_parser("start", help="start tracker")
    tstart.add_argument("--product", required=True)
    tstart.set_defaults(func=cmd_track_start)
    tstop = tss.add_parser("stop", help="stop tracker")
    tstop.set_defaults(func=cmd_track_stop)
    tstatus = tss.add_parser("status", help="print status (JSON)")
    tstatus.set_defaults(func=cmd_track_status)
    tframe = tss.add_parser("get-frame", help="print one frame as base64 JPEG")
    tframe.set_defaults(func=cmd_track_getframe)

    w = sub.add_parser("watch", help="live: frames->stdout(base64), status->stderr(JSON)")
    w.add_argument("--product", required=True)
    w.add_argument("--annotated", action="store_true", help="draw bbox/point on frames")
    w.add_argument("--fps", type=int, default=0, help="0 = unlimited")
    w.add_argument("--emit-frames", action="store_true")
    w.add_argument("--emit-status", action="store_true")
    w.add_argument("--status-every", default="0.5")
    w.add_argument("--status-channel", choices=["stdout", "stderr"], default="stderr")
    w.set_defaults(func=cmd_watch)

    sc = sub.add_parser("sync", help="synchronize camera parameters")
    sc.set_defaults(func=cmd_sync)

    iph = sub.add_parser("getImage", help="get product image path")
    iph.add_argument("--product", required=True)
    iph.set_defaults(func=cmd_get_product_image)
    return p

def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)

if __name__ == "__main__":
    main()
