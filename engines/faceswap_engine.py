"""
Realistic face swap (source -> target) for both images and video.

  - InsightFace `inswapper_128` does the swap.
  - GFPGAN restores/enhances the swapped face.
  - Image path: detect + swap all faces, composite at full resolution.
  - Video path: swap every frame, then mux the original audio back with ffmpeg.

Runs in the main Colab env (onnxruntime-gpu + GFPGAN).
"""
import os
import subprocess
import tempfile

import cv2
import numpy as np

from core.base_engine import BaseEngine
from core.model_manager import load_model
from core.utils import timestamp_file
from core.device import empty_cache
from core.config import (
    INSWAPPER_PATH, INSIGHTFACE_ROOT, GFPGAN_PATH, FFMPEG_PATH,
)

MAX_DET_SIZE = 1920   # cap detection resolution; composite stays full-res

_face_analyser = None
_swapper       = None
_enhancer      = None


def load_face_analyser():
    global _face_analyser
    if _face_analyser is None:
        import insightface
        print("[FaceSwap] Loading InsightFace analyser (buffalo_l)...")
        _face_analyser = insightface.app.FaceAnalysis(
            name="buffalo_l",
            root=INSIGHTFACE_ROOT,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        _face_analyser.prepare(ctx_id=0, det_size=(640, 640))
    return _face_analyser


def load_swapper():
    global _swapper
    if _swapper is None:
        import insightface
        print("[FaceSwap] Loading inswapper_128...")
        _swapper = insightface.model_zoo.get_model(
            INSWAPPER_PATH,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
    return _swapper


def load_enhancer():
    global _enhancer
    if _enhancer is None:
        from gfpgan import GFPGANer
        print("[FaceSwap] Loading GFPGAN enhancer...")
        _enhancer = GFPGANer(
            model_path=GFPGAN_PATH,
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
        )
    return _enhancer


def _largest_face(faces):
    return sorted(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        reverse=True,
    )[0]


def _detect_scaled(analyser, bgr):
    """Detect faces on a downscaled copy, rescale landmarks back to full res."""
    h, w = bgr.shape[:2]
    scale = min(MAX_DET_SIZE / max(h, w), 1.0)
    small = cv2.resize(bgr, (int(w * scale), int(h * scale))) if scale < 1.0 else bgr
    faces = analyser.get(small)
    if scale < 1.0:
        for f in faces:
            f.bbox /= scale
            if f.kps is not None:
                f.kps /= scale
            for attr in ("landmark_2d_106", "landmark_3d_68"):
                v = getattr(f, attr, None)
                if v is not None:
                    setattr(f, attr, v / scale)
    return faces


class FaceSwapEngine(BaseEngine):

    def load(self):
        load_model("faceswap_analyser", load_face_analyser)
        # swapper + enhancer share the slot; load lazily on first run

    def unload(self):
        global _face_analyser, _swapper, _enhancer
        _face_analyser = None
        _swapper       = None
        _enhancer      = None
        from core.model_manager import unload_model
        unload_model("faceswap_analyser")
        empty_cache()

    # ── shared swap of one frame ────────────────────────────────────────────
    def _swap_frame(self, frame_bgr, src_face, analyser, swapper, enhance):
        tgt_faces = _detect_scaled(analyser, frame_bgr)
        if not tgt_faces:
            return frame_bgr, False
        result = frame_bgr.copy()
        for tgt_face in tgt_faces:
            result = swapper.get(result, tgt_face, src_face, paste_back=True)
        if enhance:
            try:
                _, _, result = load_enhancer().enhance(result, paste_back=True)
            except Exception as e:
                print(f"[FaceSwap] Enhance skipped on frame: {e}")
        return result, True

    # ── image -> image ──────────────────────────────────────────────────────
    def run_image(self, source_path, target_path, enhance=True):
        analyser = load_face_analyser()
        swapper  = load_swapper()

        src_bgr = cv2.imread(source_path)
        tgt_bgr = cv2.imread(target_path)
        if src_bgr is None:
            raise ValueError(f"Cannot read source image: {source_path}")
        if tgt_bgr is None:
            raise ValueError(f"Cannot read target image: {target_path}")

        src_faces = analyser.get(src_bgr)
        if not src_faces:
            raise ValueError("No face detected in source image.")
        src_face = _largest_face(src_faces)

        result, ok = self._swap_frame(tgt_bgr, src_face, analyser, swapper, enhance)
        if not ok:
            raise ValueError("No face detected in target image.")

        out_path = timestamp_file("faceswap", "png")
        cv2.imwrite(out_path, result)
        print(f"[FaceSwap] Saved: {out_path}")
        return out_path

    # ── image -> video ────────────────────────────────────────────────────────
    def run_video(self, source_path, target_video, enhance=True, progress=None):
        analyser = load_face_analyser()
        swapper  = load_swapper()

        src_bgr = cv2.imread(source_path)
        if src_bgr is None:
            raise ValueError(f"Cannot read source image: {source_path}")
        src_faces = analyser.get(src_bgr)
        if not src_faces:
            raise ValueError("No face detected in source image.")
        src_face = _largest_face(src_faces)

        cap = cv2.VideoCapture(target_video)
        if not cap.isOpened():
            raise ValueError(f"Cannot open target video: {target_video}")
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        silent = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        writer = cv2.VideoWriter(
            silent, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )

        i = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                out, _ = self._swap_frame(frame, src_face, analyser, swapper, enhance)
                writer.write(out)
                i += 1
                if progress and total:
                    progress(i / total, desc=f"Swapping frame {i}/{total}")
        finally:
            cap.release()
            writer.release()

        out_path = timestamp_file("faceswap", "mp4")
        if not self._mux_audio(silent, target_video, out_path):
            os.replace(silent, out_path)   # no audio track in source
        else:
            try:
                os.unlink(silent)
            except Exception:
                pass
        print(f"[FaceSwap] Saved: {out_path}")
        return out_path

    def _mux_audio(self, video_no_audio, audio_source, out_path):
        """Copy the audio track from audio_source onto video_no_audio."""
        r = subprocess.run([
            FFMPEG_PATH, "-y",
            "-i", video_no_audio,
            "-i", audio_source,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-map", "0:v:0", "-map", "1:a:0?",
            "-shortest",
            out_path,
        ], capture_output=True, text=True)
        return r.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 10000

    # ── dispatch ────────────────────────────────────────────────────────────
    def run(self, source_path, target_path, enhance=True, is_video=False, progress=None):
        if is_video:
            return self.run_video(source_path, target_path, enhance, progress)
        return self.run_image(source_path, target_path, enhance)
