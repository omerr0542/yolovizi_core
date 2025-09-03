# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional
import threading
import time
import numpy as np
import cv2

try:
    from pypylon import pylon
except Exception as e:
    pylon = None  # type: ignore

from ..config import PYLON_SERIAL, FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS, PIXEL_FORMAT, HORIZ_FLIP

class PylonGrabber:
    # """
    # Basler Pylon kamera sarmalay�c�.
    # - grab_one(): tek kare
    # - start()/stop(): s�rekli �ekim (LatestImageOnly) ? last_frame g�ncellenir
    # - get_last(): son kare (numpy BGR)
    # """

    def __init__(self, serial: str = PYLON_SERIAL):
        if pylon is None:
            raise RuntimeError("pypylon not found. Basler pylon + 'pip install pypylon' kurun.")
        self.serial = serial.strip() or None
        self.cam: Optional["pylon.InstantCamera"] = None
        self.conv = pylon.ImageFormatConverter()
        self.conv.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.conv.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        self._run = False
        self._th: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._last: Optional[np.ndarray] = None

    def open(self):
        if self.cam is not None and self.cam.IsOpen():
            return
        tl = pylon.TlFactory.GetInstance()
        if self.serial:
            devs = tl.EnumerateDevices()
            dev = next((d for d in devs if d.GetSerialNumber() == self.serial), None)
            if dev is None:
                raise RuntimeError(f"Pylon not found (serial={self.serial})")
            self.cam = pylon.InstantCamera(tl.CreateDevice(dev))
        else:
            self.cam = pylon.InstantCamera(tl.CreateFirstDevice())
        self.cam.Open()

        try:
            # varsa bu knoblar �ok i�e yarar:
            if hasattr(self.cam, "ExposureAuto"):
                self.cam.ExposureAuto.SetValue("Off")
            if hasattr(self.cam, "GainAuto"):
                self.cam.GainAuto.SetValue("Off")
            if hasattr(self.cam, "BalanceWhiteAuto"):
                self.cam.BalanceWhiteAuto.SetValue("Off")
            # sabit pozlama �rne�i (mikrosaniye):
            if hasattr(self.cam, "ExposureTime"):
                self.cam.ExposureTime.SetValue(10000.0)  # 10 ms; ����a g�re ayarla
        except Exception:
            pass

        # Temel nodlar (uygunsa)
        try:
            if "Width" in self.cam.GetNodeMap():
                self.cam.Width.Value = min(self.cam.Width.Max, FRAME_WIDTH)
            if "Height" in self.cam.GetNodeMap():
                self.cam.Height.Value = min(self.cam.Height.Max, FRAME_HEIGHT)
        except Exception:
            pass

        # Frame rate (uygunsa)
        try:
            if hasattr(self.cam, "AcquisitionFrameRateEnable"):
                self.cam.AcquisitionFrameRateEnable.Value = True
                self.cam.AcquisitionFrameRate.Value = float(TARGET_FPS)
        except Exception:
            pass

    def close(self):
        if self.cam is not None:
            try:
                if self.cam.IsGrabbing():
                    self.cam.StopGrabbing()
                if self.cam.IsOpen():
                    self.cam.Close()
            finally:
                self.cam = None

    def grab_one(self, timeout_ms: int = 1000) -> Optional[np.ndarray]:
        self.open()
        assert self.cam is not None
        # E�er continuous grabbing a��ksa, GrabOne �a��rmayal�m!
        if self.cam.IsGrabbing():
            # _last'�n dolmas�n� bekle
            waited = 0
            step = 10  # ms
            while waited < timeout_ms:
                with self._lock:
                    if self._last is not None:
                        return self._last.copy()
                time.sleep(step / 1000.0)
                waited += step
            return None

        # Normal tek-kare �ekimi (grabbing kapal�yken g�venli)
        res = self.cam.GrabOne(timeout_ms)
        try:
            if res.GrabSucceeded():
                img = self.conv.Convert(res).GetArray()
                if HORIZ_FLIP:
                    img = cv2.flip(img, 1)
                return img
            return None
        finally:
            res.Release()

    def _loop(self):
        assert self.cam is not None
        self.cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        try:
            while self._run and self.cam.IsGrabbing():
                res = self.cam.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
                try:
                    if res.GrabSucceeded():
                        img = self.conv.Convert(res).GetArray()
                        if HORIZ_FLIP:
                            img = cv2.flip(img, 1)
                        with self._lock:
                            self._last = img
                finally:
                    res.Release()
        except Exception:
            pass
        finally:
            try:
                self.cam.StopGrabbing()
            except Exception:
                pass

    def start(self):
        self.open()
        self._run = True
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        # �lk kare gelene kadar �ok k�sa bekle (kullan�c� taraf�nda None g�rmeyelim)
        for _ in range(100):  # ~1 sn
            with self._lock:
                if self._last is not None:
                    break
            time.sleep(0.01)

    def stop(self):
        self._run = False
        if self._th and self._th.is_alive():
            self._th.join(timeout=1.0)
        self._th = None

    def get_last(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._last is None else self._last.copy()

    # Context manager
    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        self.close()

    def is_running(self) -> bool:
        return self.cam is not None and self._run and self.cam.IsGrabbing()

    def synchronize(self) -> bool:
        minLowerLimit = self.cam.AutoGainLowerLimit.Min
        maxUpperLimit = self.cam.AutoGainUpperLimit.Max
        self.cam.AutoGainLowerLimit.Value = minLowerLimit
        self.cam.AutoGainUpperLimit.Value = maxUpperLimit
        self.cam.AutoTargetBrightness.Value = 0.6
        self.cam.AutoFunctionROISelector.Value = "ROI1"
        self.cam.AutoFunctionROIUseBrightness.Value = True
        self.cam.GainAuto.Value = "Continuous"
        self.cam.ExposureAuto.Value = "Continuous"


if __name__ == "__main__":
    import argparse, cv2, time, numpy as np
    parser = argparse.ArgumentParser("pylon_cam")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("probe", help="")
    p1.add_argument("--show", action="store_true")

    p2 = sub.add_parser("stream", help="live")
    p2.add_argument("--show", action="store_true")
    p2.add_argument("--duration", type=float, help="time (sn)")

    args = parser.parse_args()

    if args.cmd == "probe":
        from .pylon_cam import PylonGrabber
        with PylonGrabber() as g:
            img = g.grab_one()
            if img is None: raise SystemExit("Kamera kare vermedi.")
            h, w = img.shape[:2]
            print(f"Tek kare: {w}x{h}")
            if args.show:
                cv2.imshow("probe", img); cv2.waitKey(0); cv2.destroyAllWindows()

    elif args.cmd == "stream":
        from .pylon_cam import PylonGrabber
        t0 = time.time()
        with PylonGrabber() as g:
            g.start()
            try:
                while True:
                    img = g.get_last() or g.grab_one()
                    if img is None: break
                    if args.show:
                        cv2.imshow("stream (q=quit)", img)
                        if cv2.waitKey(1) & 0xFF == ord('q'): break
                    if args.duration and (time.time()-t0) >= args.duration: break
            finally:
                g.stop()
        print("stream bitti")