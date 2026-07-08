"""
Avatar Studio (Tier A) — identity-preserving portrait from several photos, via
InstantID (zero-shot, no per-person training). Runs in the main env (diffusers
+ insightface). The app then voices + animates the portrait with the existing
voice + SadTalker/MuseTalk engines.

Pipeline: average the ArcFace embeddings from all uploaded faces for a robust
identity, use the clearest face's keypoints as the ControlNet condition, then
generate the prompted portrait with that identity locked in.
"""
import os

import cv2
import numpy as np

from core.base_engine import BaseEngine
from core.model_manager import load_model
from core.utils import timestamp_file, transcode_h264
from core.device import DEVICE
from core.config import INSTANTID_DIR, INSTANTID_MODELS, INSIGHTFACE_ROOT
from engines.diffusion_engine import SDXL_REAL_REPO, SDXL_NEGATIVE

_app = None


def _fix_antelopev2():
    """The antelopev2 zip extracts into a nested antelopev2/antelopev2/ folder,
    so InsightFace can't find the models ('detection' assert). Flatten it."""
    import glob, shutil
    base = os.path.join(INSIGHTFACE_ROOT, "models", "antelopev2")
    nested = os.path.join(base, "antelopev2")
    if os.path.isdir(nested) and not glob.glob(os.path.join(base, "*.onnx")):
        for f in glob.glob(os.path.join(nested, "*")):
            shutil.move(f, base)
        print("[Avatar] flattened antelopev2 folder")


def _load_app():
    """antelopev2 face analysis (InstantID's expected detector)."""
    global _app
    if _app is None:
        from insightface.app import FaceAnalysis
        print("[Avatar] Loading antelopev2 ...")
        _fix_antelopev2()
        kwargs = dict(name="antelopev2", root=INSIGHTFACE_ROOT,
                     providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        try:
            _app = FaceAnalysis(**kwargs)
        except AssertionError:
            # First-ever run: insightface downloads+extracts the zip as a side
            # effect of this same constructor call, in nested form, AFTER our
            # pre-check above already ran and found nothing to flatten yet.
            # The files exist now -- flatten and retry once instead of making
            # the user click Generate a second time.
            _fix_antelopev2()
            _app = FaceAnalysis(**kwargs)
        _app.prepare(ctx_id=0, det_size=(640, 640))
    return _app


def _load_pipe():
    import sys
    import torch
    from diffusers import ControlNetModel
    sys.path.insert(0, INSTANTID_DIR)   # InstantID's pipeline module
    from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline

    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    print("[Avatar] Loading InstantID (ControlNet + IP-Adapter on RealVisXL) ...")
    controlnet = ControlNetModel.from_pretrained(
        os.path.join(INSTANTID_MODELS, "ControlNetModel"), torch_dtype=dtype)
    pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
        SDXL_REAL_REPO, controlnet=controlnet, torch_dtype=dtype)
    pipe.load_ip_adapter_instantid(os.path.join(INSTANTID_MODELS, "ip-adapter.bin"))
    if DEVICE == "cuda":
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        pipe.to("cuda") if total_gb >= 38 else pipe.enable_model_cpu_offload()
    return pipe


def _largest(faces):
    return sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                  reverse=True)[0]


class AvatarEngine(BaseEngine):

    def unload(self):
        from core.model_manager import unload_model
        unload_model("avatar_instantid")

    def extract_frames(self, video_path, n=8):
        """Sample n evenly-spaced frames from a video for identity capture."""
        video_path = transcode_h264(video_path)
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        idxs = [int(total * i / (n + 1)) for i in range(1, n + 1)] if total > 0 else []
        paths = []
        for j, idx in enumerate(idxs):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                p = timestamp_file(f"avatframe{j}", "png")
                cv2.imwrite(p, frame)
                paths.append(p)
        cap.release()
        return paths

    def build_portrait(self, image_paths, prompt, negative_prompt=None,
                       steps=30, guidance=5.0, seed=-1,
                       width=896, height=1152, id_scale=0.8):
        from PIL import Image
        import torch
        import sys
        sys.path.insert(0, INSTANTID_DIR)
        from pipeline_stable_diffusion_xl_instantid import draw_kps

        if not image_paths:
            raise ValueError("Upload a few photos of the person (5–12 is ideal).")
        if not prompt or not prompt.strip():
            raise ValueError("Describe the desired portrait/scene.")

        app = _load_app()
        embeds, ref_kps_img, ref_face = [], None, None
        for p in image_paths:
            img = cv2.imread(p)
            if img is None:
                continue
            faces = app.get(img)
            if not faces:
                continue
            f = _largest(faces)
            embeds.append(f.normed_embedding)
            if ref_face is None or (f.bbox[2] - f.bbox[0]) > (ref_face.bbox[2] - ref_face.bbox[0]):
                ref_face = f
                ref_kps_img = img
        if not embeds:
            raise ValueError("No face detected in the uploaded photos.")

        avg_embed = np.mean(embeds, axis=0)
        face_kps = draw_kps(Image.fromarray(cv2.cvtColor(ref_kps_img, cv2.COLOR_BGR2RGB)),
                            ref_face.kps)

        pipe = load_model("avatar_instantid", _load_pipe)
        pipe.set_ip_adapter_scale(float(id_scale))

        generator = None
        if seed is not None and int(seed) >= 0:
            generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

        with torch.inference_mode():
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt or SDXL_NEGATIVE,
                image_embeds=avg_embed,
                image=face_kps,
                controlnet_conditioning_scale=0.8,
                num_inference_steps=int(steps),
                guidance_scale=float(guidance),
                width=(int(width) // 8) * 8,
                height=(int(height) // 8) * 8,
                generator=generator,
            ).images[0]

        out_path = timestamp_file("avatar_portrait", "png")
        image.save(out_path)
        print(f"[Avatar] Portrait: {out_path}  (from {len(embeds)} faces)")
        return out_path
