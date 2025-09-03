# -*- coding: utf-8 -*-
# yolovizi_core/training.py
from __future__ import annotations
from pathlib import Path
from typing import Literal
import os, sys, json
import torch

from .config import MODELS_DIR, IMGSZ, EPOCHS, MODEL, MODEL_SEG, MODEL_POSE
from .storage import ProductStore

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None  # type: ignore

def _emit(ev: str, **kw):
    try:
        sys.stderr.write(json.dumps({"event": ev, **kw}, ensure_ascii=False) + "\n")
        sys.stderr.flush()
    except Exception:
        pass

def _infer_kpt_count(lbl_base: Path) -> int:
    """
    Inspect a label line and infer N for 'cls cx cy w h (x y v)*N'.
    Returns N>=1 if pose-like, else 0.
    """
    dirs: list[Path] = []
    if (lbl_base / "train").exists() or (lbl_base / "val").exists():
        if (lbl_base / "train").exists(): dirs.append(lbl_base / "train")
        if (lbl_base / "val").exists():   dirs.append(lbl_base / "val")
    else:
        dirs.append(lbl_base)

    for d in dirs:
        for p in d.glob("*.txt"):
            with p.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    toks = s.split()
                    n = len(toks)
                    if n >= 8 and (n - 5) % 3 == 0:
                        return (n - 5) // 3
                    # continue scanning other files/lines if not pose-like
    return 0


def _dataset_yaml(images_dir: Path, kind: Literal["detect","segment","pose"], kpt_n: int | None = None) -> Path:
    product_root = images_dir.parent
    use_split = (images_dir / "train").exists() and (images_dir / "val").exists()
    train_rel = "images/train" if use_split else "images"
    val_rel   = "images/val"   if use_split else "images"

    yaml = product_root / "dataset.yaml"
    base = (
        f"path: {product_root.as_posix()}\n"
        f"train: {train_rel}\n"
        f"val: {val_rel}\n"
        "names:\n"
        "  0: object\n"
    )
    if kind == "pose":
        n = int(kpt_n or 1)
        base += f"kpt_shape: [{n}, 3]\n"
        # if you ever enable horizontal flip, define swap. For 2 KPs (grip, dir) we usually swap:
        if n == 2:
            base += "flip_idx: [1, 0]\n"
        else:
            # identity mapping (no swap)
            base += "flip_idx: [" + ",".join(str(i) for i in range(n)) + "]\n"

    yaml.write_text(base, encoding="utf-8")
    return yaml



def _detect_label_type(lbl_base: Path) -> Literal["detect","segment","pose"]:
    cands: list[Path] = []
    if (lbl_base / "train").exists() or (lbl_base / "val").exists():
        if (lbl_base / "train").exists(): cands.append(lbl_base / "train")
        if (lbl_base / "val").exists():   cands.append(lbl_base / "val")
    else:
        cands.append(lbl_base)

    found_pose = found_seg = found_det = False

    def classify_line(n: int):
        nonlocal found_pose, found_seg, found_det
        # pose: cls cx cy w h (x y v)+  => n >= 8 ve (n-5) % 3 == 0
        if n >= 8 and (n - 5) % 3 == 0:
            found_pose = True
        # segment: cls (x y)+ en az 3 nokta => n >= 7 ve (n-1) % 2 == 0
        elif n >= 7 and (n - 1) % 2 == 0:
            found_seg = True
        # detect: cls cx cy w h
        elif n == 5:
            found_det = True

    for d in cands:
        for p in d.glob("*.txt"):
            with p.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    n = len(line.split())
                    classify_line(n)
                    break
        if found_pose:
            return "pose"
    if found_seg:
        return "segment"
    if found_det:
        return "detect"
    return "detect"


def _resolve_images_dir(p: Path) -> Path:
    name = p.name.lower()
    if name in ("train", "val") and p.parent.name.lower() == "images":
        return p.parent
    if (p / "images").exists():
        return p / "images"
    return p


class Trainer:
    def __init__(self):
        self.store = ProductStore()

    def train(self, product_id: int, dataset_dir: str) -> str:
        if YOLO is None:
            raise RuntimeError("Ultralytics bulunamadi (`pip install ultralytics`).")

        p_images = Path(dataset_dir) if dataset_dir else self.store.images_dir(product_id)
        p_images = _resolve_images_dir(p_images)
        p_labels = self.store.labels_dir(product_id)

        print(f"[train] running training.py at: {__file__}")
        print(f"[train] images_dir={p_images}")
        print(f"[train] labels_dir={p_labels}")

        if not p_images.exists():
            raise FileNotFoundError(f"Image Folder Not Found: {p_images}")

        for cache_name in ("train.cache", "val.cache"):
            try:
                cpath = p_labels / cache_name
                if cpath.exists():
                    print(f"[train] removing cache: {cpath}")
                    cpath.unlink()
            except Exception as ex:
                print(f"[train] cache remove warn: {ex}")

        def _label_line_count(ldir: Path) -> int:
            return sum(
                len([l for l in p.read_text(encoding='utf-8', errors='ignore').splitlines() if l.strip()])
                for p in ldir.glob("*.txt")
            )

        lbl_dirs: list[Path] = []
        if (p_labels / "train").exists() or (p_labels / "val").exists():
            if (p_labels / "train").exists(): lbl_dirs.append(p_labels / "train")
            if (p_labels / "val").exists():   lbl_dirs.append(p_labels / "val")
        else:
            lbl_dirs.append(p_labels)

        total_lines = sum(_label_line_count(d) for d in lbl_dirs)
        if total_lines == 0:
            raise FileNotFoundError(
                f"Etiket bulunamadi: {p_labels}\n"
                f"Once polygon/synth ile veri uretin veya etiketleyin."
            )

        def _peek_line(d: Path):
            for p in d.glob("*.txt"):
                with p.open("r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        s = line.strip()
                        if s:
                            toks = s.split()
                            return p.name, s, len(toks)
            return None

        peek = _peek_line(lbl_dirs[0])
        if peek:
            fn, sample, n = peek
            print(f"[train] sample_label_file={fn}  tokens={n}  line='{sample}'")

        env_force = os.environ.get("VISIONSUITE_FORCE_TASK", "").strip().lower()
        if env_force in {"pose","segment","detect"}:
            kind: Literal["pose","segment","detect"] = env_force  # type: ignore
            print(f"[train] OVERRIDE: VISIONSUITE_FORCE_TASK='{env_force}'")
        else:
            kind = _detect_label_type(p_labels)

        print(f"[train] label_mode={kind}  images={p_images}  labels={p_labels}")

        def _looks_like_pose(sample_line: str) -> bool:
            n = len(sample_line.split())
            return (n >= 8) and ((n - 5) % 3 == 0)

        if peek and _looks_like_pose(peek[1]) and kind != "pose":
            raise RuntimeError(
                f"Label looks POSE (tokens={peek[2]}), but kind='{kind}'. "
                f"Yanlis pipeline'a dusmemek icin durduruldu."
            )

        kpt_n = 0
        if kind == "pose":
            kpt_n = _infer_kpt_count(p_labels)
            if kpt_n <= 0:
                kpt_n = 1
            print(f"[train] pose kpt_n={kpt_n}")
            _emit(f"[train] pose kpt_n={kpt_n}")

        data_yaml = _dataset_yaml(p_images, kind, kpt_n if kind == "pose" else None)

        if kind == "pose":
            print(f"[train] using MODEL_POSE={MODEL_POSE}")
            _emit(f"[train] using MODEL_POSE={MODEL_POSE}")
            model = YOLO(MODEL_POSE)
        elif kind == "segment":
            print(f"[train] using MODEL_SEG={MODEL_SEG}")
            _emit(f"[train] using MODEL_SEG={MODEL_SEG}")
            model = YOLO(MODEL_SEG)
        else:
            print(f"[train] using MODEL={MODEL}")
            _emit(f"[train] using MODEL={MODEL}")
            model = YOLO(MODEL)

        _emit("device", device=str(getattr(model, "device", "unknown")))

        use_gpu = torch.cuda.is_available()
        if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "-1":
            del os.environ["CUDA_VISIBLE_DEVICES"]

        device_arg = 0 if use_gpu else "cpu"

        def _on_train_epoch_end(trainer):
            ep  = int(getattr(trainer, "epoch", 0)) + 1
            eps = int(getattr(trainer, "epochs", EPOCHS))
            loss = None
            try:
                m = getattr(trainer, "metrics", None) or {}
                loss = float(m.get("loss", None) or m.get("train/box_loss", None) or m.get("train/loss", None) or 0.0)
            except Exception:
                pass
            _emit("epoch_end", epoch=ep, epochs=eps, loss=loss)

        def _on_train_end(trainer):
            _emit("train_end")

        for name in ["on_train_epoch_end", "on_fit_epoch_end"]:
            try:
                model.add_callback(name, _on_train_epoch_end)
            except Exception:
                pass
        try:
            model.add_callback("on_train_end", _on_train_end)
        except Exception:
            pass

        r = model.train(
            data=str(data_yaml),
            epochs=EPOCHS,
            imgsz=IMGSZ,
            name=f"product_{product_id}",
            device=device_arg,
            project=str(MODELS_DIR),
            fliplr=0.0, flipud=0.0, degrees=0.0, shear=0.0,
            mixup=0.0, copy_paste=0.0, mosaic=0.0,
            workers=4,
            cache=True 
        )

        run = Path(r.save_dir) / "weights" / "best.pt"
        print(f"[device] device = {model.device}")

        if not run.exists():
            raise FileNotFoundError("best.pt not found")
        final = MODELS_DIR / f"product_{product_id}.pt"
        final.parent.mkdir(parents=True, exist_ok=True)

        _emit("final_model", path=str(final))

        try:
            if final.exists(): final.unlink()
            run.replace(final)
        except Exception:
            import shutil; shutil.copy2(run, final)
        return str(final)


if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser("training")
    parser.add_argument("--product", required=True, type=int)
    parser.add_argument("--dataset", help="images folder")
    args = parser.parse_args()
    print(json.dumps({"result": Trainer().train(args.product, args.dataset or "")}, ensure_ascii=False, indent=2))
