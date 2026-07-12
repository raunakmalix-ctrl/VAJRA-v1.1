"""
VAJRA v1.1 — AI Media Studio (Colab / Gradio)

Run:  python app.py        (set IMAGE_TALK_SHARE=1 for a public link)
"""
import os
import sys
import threading

PROJECT_ROOT = os.environ.get(
    "IMAGE_TALK_ROOT", os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, PROJECT_ROOT)

# An empty HF_TOKEN makes huggingface_hub send an illegal "Bearer " header.
if not os.environ.get("HF_TOKEN"):
    os.environ.pop("HF_TOKEN", None)

import gradio as gr

# Work around a gradio_client bug where a JSON schema with a boolean
# `additionalProperties` crashes the API-info endpoint that the share tunnel
# pings ("argument of type 'bool' is not iterable"). Harmless to the UI, but
# floods the log — short-circuit non-dict schemas.
try:
    import gradio_client.utils as _gcu
    _orig_j2p = _gcu._json_schema_to_python_type

    def _safe_j2p(schema, defs=None):
        if not isinstance(schema, dict):
            return "Any"
        return _orig_j2p(schema, defs)

    _gcu._json_schema_to_python_type = _safe_j2p
except Exception:
    pass

from core.engine_registry import ENGINES
from core.model_manager import vram_status
from core.utils import audio_duration
from core.config import OUTPUTS_DIR

diffusion   = ENGINES["diffusion"]
faceswap    = ENGINES["faceswap"]
transcript  = ENGINES["transcript"]
ltx         = ENGINES["ltx"]
media       = ENGINES["media"]
motion      = ENGINES["motion"]


# ── GPU helpers ─────────────────────────────────────────────────────────────
# Gradio's queue processes multiple tab requests concurrently (max_size=4), but
# they all share one physical GPU. Without this, e.g. a Text->Image generation
# and an Image Edit click a few seconds apart can both be mid-load at once and
# collectively exceed VRAM (observed: FLUX.1-Kontext-dev OOM while SDXL was
# still resident from a concurrent request). Serialize all GPU-heavy tabs.
GPU_LOCK = threading.Lock()


def free_inprocess():
    """Free VRAM held by the main-env engines before a subprocess job."""
    try:
        diffusion.unload()
    except Exception:
        pass
    try:
        faceswap.unload()
    except Exception:
        pass


def vram_html():
    return (f"<div class='vram-footer'>"
            f"<span class='vram-text'>◆ VAJRA &nbsp;·&nbsp; Offline AI Suite</span>"
            f"<span class='vram-text vj-clock' id='vajra-clock'>SYS TIME --:--:--Z</span>"
            f"<span class='vram-text vram-accent'>{vram_status()}</span></div>")


def ribbon_html():
    import torch
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (no GPU)"
    return (
        "<div class='status-ribbon'>"
        "<span class='sr-item'><span class='sr-dot'></span><b class='sr-live'>ONLINE</b></span>"
        "<span class='sr-item'><i class='ti ti-shield-lock'></i>100% OFFLINE / LOCAL</span>"
        f"<span class='sr-item'><i class='ti ti-cpu'></i>Compute&nbsp;<b>{dev}</b></span>"
        "<span class='sr-item'><i class='ti ti-stack-2'></i><b>6</b>&nbsp;Modules</span>"
        "</div>"
    )


def hero(icon, title, sub):
    return (f"<div class='tab-hero'><div class='th-ico'><i class='ti {icon}'></i></div>"
            f"<div class='th-txt'><span class='th-title'>{title}</span>"
            f"<span class='th-sub'>{sub}</span></div></div>")


AWAIT = "<span class='empty-hint'><i class='ti ti-player-play'></i>Awaiting input…</span>"


def ok(msg):  return f"<span class='status-ok'>✔ {msg}</span>"

def err(msg):
    # Print the full traceback to the Colab cell when called inside an except.
    import sys, traceback
    if sys.exc_info()[0] is not None:
        traceback.print_exc()
    return f"<span class='status-err'>✖ {msg}</span>"
def warn(msg): return f"<span class='status-warn'>⚠ {msg}</span>"


# ── Feature: Transcript edit + relip ─────────────────────────────────────────
def do_extract(video, progress=gr.Progress()):
    if video is None:
        return "", None, warn("Upload a video first")
    GPU_LOCK.acquire()
    try:
        free_inprocess()
        progress(0.3, desc="Transcribing (WhisperX) ...")
        text, state = transcript.extract_transcript(video)
        return text, state, ok(f"{len(state['segments'])} segments · "
                               f"lang={state['language']}")
    except Exception as e:
        return "", None, err(str(e))
    finally:
        GPU_LOCK.release()


def do_relip(state, edited_text, method, steps, guidance,
             progress=gr.Progress()):
    if not state:
        return None, warn("Extract a transcript first")
    if not edited_text or not edited_text.strip():
        return None, warn("Transcript is empty")
    GPU_LOCK.acquire()
    try:
        free_inprocess()
        method_map = {"LatentSync": "latentsync", "MuseTalk": "musetalk",
                      "Wav2Lip": "wav2lip"}
        m = next((v for k, v in method_map.items() if method.startswith(k)),
                 "latentsync")
        out = transcript.apply_edits(
            state, edited_text,
            method=m,
            inference_steps=int(steps), guidance_scale=float(guidance),
            progress=progress,
        )
        return out, ok(os.path.basename(out))
    except Exception as e:
        return None, err(str(e))
    finally:
        GPU_LOCK.release()


# ── Feature: Text → image ────────────────────────────────────────────────────
def run_txt2img(prompt, variant, negative, width, height, steps, guidance, seed):
    if not prompt or not prompt.strip():
        return None, warn("Prompt required")
    GPU_LOCK.acquire()
    try:
        free_inprocess()
        v = {"SDXL Realistic (best, open)": "sdxl_real",
             "SDXL base (open)": "sdxl",
             "FLUX Schnell (token)": "schnell",
             "FLUX Dev (token)": "dev"}.get(variant, "sdxl_real")
        out = diffusion.run(prompt=prompt, variant=v, negative_prompt=negative,
                            width=int(width), height=int(height),
                            steps=int(steps), guidance=float(guidance),
                            seed=int(seed))
        return out, ok(os.path.basename(out))
    except Exception as e:
        return None, err(str(e))
    finally:
        GPU_LOCK.release()


# ── Image Edit (FLUX.1-Kontext-dev) ─────────────────────────────────────────
def run_edit(image, prompt, steps, guidance, seed):
    if image is None:
        return None, warn("Upload an image to edit")
    if not prompt or not prompt.strip():
        return None, warn("Describe the edit")
    GPU_LOCK.acquire()
    try:
        free_inprocess()
        out = diffusion.edit(image_path=image, prompt=prompt,
                             steps=int(steps), guidance=float(guidance), seed=int(seed))
        return out, ok(os.path.basename(out))
    except Exception as e:
        return None, err(str(e))
    finally:
        GPU_LOCK.release()


# ── Feature: Face swap ───────────────────────────────────────────────────────
def run_faceswap(source, mode, target_img, target_vid, enhancer,
                 progress=gr.Progress()):
    if source is None:
        return None, None, warn("Upload a source face")
    is_video = (mode == "Video")
    target = target_vid if is_video else target_img
    if target is None:
        return None, None, warn(f"Upload a target {'video' if is_video else 'image'}")
    GPU_LOCK.acquire()
    try:
        free_inprocess()
        faceswap.load()
        e = {"GFPGAN (default)": "gfpgan",
             "CodeFormer (non-commercial)": "codeformer",
             "None": "none"}.get(enhancer, "gfpgan")
        out = faceswap.run(source_path=source, target_path=target,
                           enhancer=e, is_video=is_video,
                           progress=progress if is_video else None)
        if is_video:
            return None, out, ok(os.path.basename(out))
        return out, None, ok(os.path.basename(out))
    except Exception as e:
        return None, None, err(str(e))
    finally:
        GPU_LOCK.release()


# ── Feature: Text → Video (LTX text-only, or Wan2.2-I2V with a reference photo)
def run_ltx(image, prompt, negative, width, height, num_frames, steps, guidance):
    if not prompt or not prompt.strip():
        return None, warn("Prompt required")
    GPU_LOCK.acquire()
    try:
        free_inprocess()
        if image:
            out = motion.run(image_path=image, prompt=prompt,
                             negative_prompt=negative,
                             width=int(width), height=int(height),
                             num_frames=int(num_frames),
                             guidance=float(guidance))
        else:
            out = ltx.run(prompt=prompt, negative_prompt=negative,
                          width=int(width), height=int(height),
                          num_frames=int(num_frames), steps=int(steps),
                          guidance=float(guidance))
        return out, ok(os.path.basename(out))
    except Exception as e:
        return None, err(str(e))
    finally:
        GPU_LOCK.release()


def toggle_swap_mode(mode):
    is_video = (mode == "Video")
    return gr.update(visible=not is_video), gr.update(visible=is_video), \
           gr.update(visible=not is_video), gr.update(visible=is_video)


def audio_info(path):
    if not path:
        return ""
    d = audio_duration(path)
    if d is None:
        return ""
    m, s = int(d // 60), int(d % 60)
    return f"<div class='audio-info'>▶ {m}m {s:02d}s reference</div>"


# ── Media Studio callbacks (all ffmpeg/CPU) ─────────────────────────────────
def _media(fn, *a):
    try:
        return fn(*a), ok("Done")
    except Exception as e:
        return None, err(str(e))

def _media_dl(fn, *a):
    """Like _media, but also returns a value for a companion DownloadButton."""
    try:
        out = fn(*a)
        return out, ok("Done"), gr.DownloadButton(value=out, visible=True)
    except Exception as e:
        return None, err(str(e)), gr.DownloadButton(visible=False)

def m_trim_video(v, s, e): return _media_dl(media.trim_video, v, s, e)
def m_trim_audio(a, s, e): return _media_dl(media.trim_audio, a, s, e)
def m_grab(v, t):          return _media(media.grab_frame, v, t)
def m_clean(a):            return _media_dl(media.clean_audio, a)
def m_convert(f, t):       return _media(media.convert, f, t)
def m_resize(v, w, h):     return _media(media.resize_video, v, w, h)
def m_gif(v, fps, w):      return _media(media.to_gif, v, fps, w)
def m_caption(v):          return _media(media.burn_captions, v)
def m_removebg(img, bg):   return _media(media.remove_bg, img, bg)

def m_merge(files, kind):
    try:
        paths = [f.name if hasattr(f, "name") else f for f in (files or [])]
        return media.concat(paths, "video" if kind == "Video" else "audio"), ok("Merged")
    except Exception as e:
        return None, err(str(e))


# ── Output history ──────────────────────────────────────────────────────────
def list_outputs():
    import glob
    files = [f for f in glob.glob(os.path.join(OUTPUTS_DIR, "*")) if os.path.isfile(f)]
    return sorted(files, key=os.path.getmtime, reverse=True)[:60]

def zip_outputs():
    import glob, zipfile, tempfile, time as _t
    files = [f for f in glob.glob(os.path.join(OUTPUTS_DIR, "*")) if os.path.isfile(f)]
    if not files:
        return None
    zpath = os.path.join(tempfile.gettempdir(),
                         _t.strftime("image_talk_outputs_%Y%m%d_%H%M%S.zip"))
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, os.path.basename(f))
    return zpath


# ════════════════════════════════════════════════════════════════════════════
#  UI
# ════════════════════════════════════════════════════════════════════════════
from app_theme import CSS, THEME_JS, MASTHEAD   # noqa: E402

with gr.Blocks(css=CSS, title="VAJRA", analytics_enabled=False) as demo:
    gr.HTML(f"<script>{THEME_JS}</script>")
    gr.HTML(MASTHEAD)
    gr.HTML(ribbon_html())

    with gr.Tabs() as tabs:

        # ── 01 Edit & Relip ─────────────────────────────────────────────────
        with gr.Tab("01 · Edit & Relip", id=0):
            ed_state = gr.State(None)
            gr.HTML(hero("ti-pencil", "Edit & Relip",
                "Pull a video's transcript, change words, re-sync the lips."))
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    gr.HTML("<div class='section-label'>Source Video</div>")
                    ed_video = gr.Video(label="Upload a talking-head video",
                                        elem_classes=["output-media"])
                    ed_extract = gr.Button("◐  Extract Transcript", variant="secondary")
                    gr.HTML("<div class='section-label'>Transcript "
                            "(edit words · keep one segment per line)</div>")
                    ed_text = gr.Textbox(label="", lines=8,
                        placeholder="Transcript appears here after extraction…")
                    with gr.Row():
                        ed_method = gr.Radio(
                            ["LatentSync (best)", "MuseTalk (sharp)", "Wav2Lip (fast)"],
                            value="LatentSync (best)", label="Lip-sync")
                        ed_steps  = gr.Slider(10, 50, value=20, step=1,
                                              label="Diffusion steps")
                        ed_guid   = gr.Slider(1.0, 3.0, value=1.5, step=0.1,
                                              label="Guidance")
                    ed_relip = gr.Button("▶  Apply Edits & Re-sync", variant="primary")
                with gr.Column(scale=1):
                    gr.HTML("<div class='section-label'>Output</div>")
                    ed_out = gr.Video(label="", elem_classes=["output-media"])
                    ed_status = gr.HTML(AWAIT)
            ed_extract.click(do_extract, [ed_video], [ed_text, ed_state, ed_status])
            ed_relip.click(do_relip,
                           [ed_state, ed_text, ed_method, ed_steps, ed_guid],
                           [ed_out, ed_status])

        # ── 02 Text → Image ─────────────────────────────────────────────────
        with gr.Tab("02 · Text → Image", id=1):
            gr.HTML(hero("ti-photo", "Text → Image",
                "Generate photoreal images from a prompt (SDXL / FLUX)."))
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    gr.HTML("<div class='section-label'>Prompt</div>")
                    ti_prompt = gr.Textbox(label="Prompt", lines=4,
                        placeholder="A cinematic portrait, golden hour, ultra-detailed…")
                    ti_variant = gr.Radio(
                        ["SDXL Realistic (best, open)", "SDXL base (open)",
                         "FLUX Schnell (token)", "FLUX Dev (token)"],
                        value="SDXL Realistic (best, open)",
                        label="Model  (FLUX needs HF_TOKEN + license)")
                    ti_neg = gr.Textbox(label="Negative prompt (SDXL only)", lines=2,
                        placeholder="leave blank for the built-in quality default")
                    with gr.Row():
                        ti_w = gr.Slider(512, 1536, value=1024, step=64, label="Width")
                        ti_h = gr.Slider(512, 1536, value=1024, step=64, label="Height")
                    with gr.Row():
                        ti_steps = gr.Slider(4, 50, value=30, step=1, label="Steps")
                        ti_guid  = gr.Slider(0.0, 9.0, value=6.0, step=0.5, label="Guidance")
                    ti_seed = gr.Number(label="Seed (−1 = random)", value=-1, precision=0)
                    ti_btn  = gr.Button("▶  Generate", variant="primary")
                with gr.Column(scale=1):
                    gr.HTML("<div class='section-label'>Output</div>")
                    ti_out = gr.Image(label="", elem_classes=["output-media"])
                    ti_status = gr.HTML(AWAIT)
            ti_btn.click(run_txt2img,
                         [ti_prompt, ti_variant, ti_neg, ti_w, ti_h,
                          ti_steps, ti_guid, ti_seed],
                         [ti_out, ti_status])

        # ── 03 Face Swap ─────────────────────────────────────────────────────
        with gr.Tab("03 · Face Swap", id=2):
            gr.HTML(hero("ti-mask", "Face Swap",
                "Swap a source face onto a target image or every video frame."))
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    gr.HTML("<div class='section-label'>Source &amp; Target</div>")
                    fs_src  = gr.Image(label="Source face (copy FROM)",
                                       type="filepath", elem_classes=["output-media"])
                    fs_mode = gr.Radio(["Image", "Video"], value="Image",
                                       label="Target type")
                    fs_timg = gr.Image(label="Target image (paste INTO)",
                                       type="filepath", visible=True,
                                       elem_classes=["output-media"])
                    fs_tvid = gr.Video(label="Target video (paste INTO every frame)",
                                       visible=False, elem_classes=["output-media"])
                    fs_enh  = gr.Radio(
                        ["GFPGAN (default)", "CodeFormer (non-commercial)", "None"],
                        value="GFPGAN (default)",
                        label="Face enhancer  (CodeFormer = sharper, image-only, "
                              "S-Lab non-commercial license)")
                    fs_btn  = gr.Button("▶  Swap Face", variant="primary")
                with gr.Column(scale=1):
                    gr.HTML("<div class='section-label'>Output</div>")
                    fs_oimg = gr.Image(label="", visible=True, elem_classes=["output-media"])
                    fs_ovid = gr.Video(label="", visible=False, elem_classes=["output-media"])
                    fs_status = gr.HTML(AWAIT)
            fs_mode.change(toggle_swap_mode, [fs_mode],
                           [fs_timg, fs_tvid, fs_oimg, fs_ovid])
            fs_btn.click(run_faceswap,
                         [fs_src, fs_mode, fs_timg, fs_tvid, fs_enh],
                         [fs_oimg, fs_ovid, fs_status])

        # ── 04 Text → Video (LTX-0.9.7-distilled · Wan2.2-I2V w/ a photo) ────
        with gr.Tab("04 · Text → Video", id=3):
            gr.HTML(hero("ti-movie", "Text → Video",
                "Prompt → video (LTX-Video). Add a reference photo to animate "
                "it instead — identity-preserving motion video (Wan2.2-I2V)."))
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    gr.HTML("<div class='section-label'>LTX-Video 0.9.7-distilled "
                            "(open · needs the optional venv_ltx) — or Wan2.2-I2V "
                            "(needs the optional venv_wan) if a photo is given. "
                            "Width/height/frames are shared between both modes and "
                            "tuned for LTX by default.</div>")
                    lx_img = gr.Image(label="Reference photo (optional) — animate "
                        "this instead of generating from scratch", type="filepath",
                        elem_classes=["output-media"])
                    lx_prompt = gr.Textbox(label="Prompt", lines=3,
                        placeholder="A cinematic drone shot over snowy mountains at "
                                    "sunrise… or, with a photo: make them dance in joy")
                    lx_neg = gr.Textbox(label="Negative prompt (text-only mode)", lines=1,
                        value="shaky, glitchy, low quality, watermark")
                    with gr.Row():
                        lx_w = gr.Slider(384, 1280, value=704, step=32, label="Width")
                        lx_h = gr.Slider(384, 1280, value=512, step=32, label="Height")
                    with gr.Row():
                        lx_frames = gr.Slider(49, 193, value=121, step=8,
                                              label="Frames (~fps·sec)")
                        lx_steps  = gr.Slider(4, 30, value=7, step=1,
                                              label="Steps (text-only mode)")
                        lx_guid   = gr.Slider(1.0, 5.0, value=1.0, step=0.5, label="Guidance")
                    lx_btn = gr.Button("▶  Generate Video", variant="primary")
                with gr.Column(scale=1):
                    gr.HTML("<div class='section-label'>Output</div>")
                    lx_out = gr.Video(label="", elem_classes=["output-media"])
                    lx_status = gr.HTML(AWAIT)
            lx_btn.click(run_ltx,
                         [lx_img, lx_prompt, lx_neg, lx_w, lx_h,
                          lx_frames, lx_steps, lx_guid],
                         [lx_out, lx_status])

        # ── 05 Image Edit (FLUX.1-Kontext-dev) ───────────────────────────────
        with gr.Tab("05 · Image Edit", id=4):
            gr.HTML(hero("ti-wand", "Image Edit",
                "Edit an image by instruction — change background, style, add objects."))
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    gr.HTML("<div class='section-label'>Instruction editing "
                            "(FLUX-Kontext · needs HF_TOKEN + license + venv_ltx / cell 7b)</div>")
                    ie_img = gr.Image(label="Image to edit", type="filepath",
                                      elem_classes=["output-media"])
                    ie_prompt = gr.Textbox(label="Edit instruction", lines=3,
                        placeholder="e.g. change the background to a sunset beach; "
                                    "make it black-and-white; add sunglasses")
                    with gr.Row():
                        ie_steps = gr.Slider(10, 40, value=28, step=1, label="Steps")
                        ie_guid  = gr.Slider(1.0, 5.0, value=2.5, step=0.5, label="Guidance")
                    ie_seed = gr.Number(label="Seed (−1 = random)", value=-1, precision=0)
                    ie_btn  = gr.Button("▶  Apply Edit", variant="primary")
                with gr.Column(scale=1):
                    gr.HTML("<div class='section-label'>Output</div>")
                    ie_out = gr.Image(label="", elem_classes=["output-media"])
                    ie_status = gr.HTML(AWAIT)
            ie_btn.click(run_edit, [ie_img, ie_prompt, ie_steps, ie_guid, ie_seed],
                         [ie_out, ie_status])

        # ── 06 Media Studio (ffmpeg/CPU — no GPU cost) ───────────────────────
        with gr.Tab("06 · Media Studio", id=5):
            gr.HTML(hero("ti-adjustments", "Media Studio",
                "Trim, grab frames, clean audio, convert, merge — all CPU, no GPU."))
            gr.HTML("<div class='section-label'>Prep your media — free (no A100). "
                    "Use the “→ Send to” buttons to push results into the AI tabs.</div>")

            with gr.Accordion("✂  Trim video", open=True):
                with gr.Row():
                    with gr.Column():
                        ms_tv_in = gr.Video(label="Video", elem_classes=["output-media"])
                        with gr.Row():
                            ms_tv_s = gr.Number(label="Start (sec)", value=0)
                            ms_tv_e = gr.Number(label="End (sec)", value=0)
                        ms_tv_btn = gr.Button("Trim", variant="primary")
                    with gr.Column():
                        ms_tv_out = gr.Video(label="Trimmed", elem_classes=["output-media"])
                        ms_tv_st = gr.HTML("")
                        with gr.Row():
                            ms_tv_dl = gr.DownloadButton("⬇ Download", size="sm", visible=False)
                            ms_tv_send_relip = gr.Button("→ Edit & Relip", size="sm")
                ms_tv_btn.click(m_trim_video, [ms_tv_in, ms_tv_s, ms_tv_e],
                                [ms_tv_out, ms_tv_st, ms_tv_dl])

            with gr.Accordion("✂  Trim audio", open=False):
                with gr.Row():
                    with gr.Column():
                        ms_ta_in = gr.Audio(label="Audio", type="filepath")
                        with gr.Row():
                            ms_ta_s = gr.Number(label="Start (sec)", value=0)
                            ms_ta_e = gr.Number(label="End (sec)", value=0)
                        ms_ta_btn = gr.Button("Trim", variant="primary")
                    with gr.Column():
                        ms_ta_out = gr.Audio(label="Trimmed", type="filepath")
                        ms_ta_st = gr.HTML("")
                        ms_ta_dl = gr.DownloadButton("⬇ Download", size="sm", visible=False)
                ms_ta_btn.click(m_trim_audio, [ms_ta_in, ms_ta_s, ms_ta_e],
                                [ms_ta_out, ms_ta_st, ms_ta_dl])

            with gr.Accordion("🎞  Grab frame", open=False):
                with gr.Row():
                    with gr.Column():
                        ms_gf_in = gr.Video(label="Video", elem_classes=["output-media"])
                        ms_gf_t = gr.Number(label="Time (sec)", value=0)
                        ms_gf_btn = gr.Button("Capture frame", variant="primary")
                    with gr.Column():
                        ms_gf_out = gr.Image(label="Frame", type="filepath",
                                             elem_classes=["output-media"])
                        ms_gf_st = gr.HTML("")
                        ms_gf_send_face = gr.Button("→ Face Swap source", size="sm")
                ms_gf_btn.click(m_grab, [ms_gf_in, ms_gf_t], [ms_gf_out, ms_gf_st])

            with gr.Accordion("🎧  Clean audio (denoise + normalize)", open=False):
                with gr.Row():
                    with gr.Column():
                        ms_ca_in = gr.Audio(label="Audio or video", type="filepath")
                        ms_ca_btn = gr.Button("Denoise + normalize", variant="primary")
                    with gr.Column():
                        ms_ca_out = gr.Audio(label="Cleaned", type="filepath")
                        ms_ca_st = gr.HTML("")
                        ms_ca_dl = gr.DownloadButton("⬇ Download", size="sm", visible=False)
                ms_ca_btn.click(m_clean, [ms_ca_in], [ms_ca_out, ms_ca_st, ms_ca_dl])

            with gr.Accordion("🪄  Remove background (portrait)", open=False):
                with gr.Row():
                    with gr.Column():
                        ms_bg_in = gr.Image(label="Portrait", type="filepath",
                                            elem_classes=["output-media"])
                        ms_bg_color = gr.Radio(["white", "green", "gray", "black"],
                                               value="white", label="New background")
                        ms_bg_btn = gr.Button("Remove background", variant="primary")
                    with gr.Column():
                        ms_bg_out = gr.Image(label="Result", type="filepath",
                                             elem_classes=["output-media"])
                        ms_bg_st = gr.HTML("")
                        ms_bg_send_face = gr.Button("→ Face Swap source", size="sm")
                ms_bg_btn.click(m_removebg, [ms_bg_in, ms_bg_color], [ms_bg_out, ms_bg_st])

            with gr.Accordion("💬  Burn captions onto video", open=False):
                with gr.Row():
                    with gr.Column():
                        ms_cap_in = gr.Video(label="Video", elem_classes=["output-media"])
                        ms_cap_btn = gr.Button("Transcribe + burn subtitles",
                                               variant="primary")
                    with gr.Column():
                        ms_cap_out = gr.Video(label="Captioned", elem_classes=["output-media"])
                        ms_cap_st = gr.HTML("")
                ms_cap_btn.click(m_caption, [ms_cap_in], [ms_cap_out, ms_cap_st])

            with gr.Accordion("🔁  Convert · compress · resize · GIF", open=False):
                with gr.Row():
                    with gr.Column():
                        ms_cv_in = gr.File(label="File")
                        ms_cv_target = gr.Dropdown(
                            ["MP4 (H.264)", "Compress MP4", "MP3", "WAV"],
                            value="Compress MP4", label="Convert to")
                        ms_cv_btn = gr.Button("Convert", variant="primary")
                        gr.HTML("<div class='section-label'>Resize / GIF (video)</div>")
                        ms_rs_in = gr.Video(label="Video", elem_classes=["output-media"])
                        with gr.Row():
                            ms_rs_w = gr.Number(label="Width", value=720)
                            ms_rs_h = gr.Number(label="Height", value=1280)
                        ms_rs_btn = gr.Button("Resize", size="sm")
                        with gr.Row():
                            ms_gif_fps = gr.Slider(5, 24, value=12, step=1, label="GIF fps")
                            ms_gif_w = gr.Slider(240, 720, value=480, step=20, label="GIF width")
                        ms_gif_btn = gr.Button("Export GIF", size="sm")
                    with gr.Column():
                        ms_cv_out = gr.File(label="Output")
                        ms_cv_st = gr.HTML("")
                        ms_rs_out = gr.Video(label="Resized", elem_classes=["output-media"])
                        ms_gif_out = gr.File(label="GIF")
                ms_cv_btn.click(m_convert, [ms_cv_in, ms_cv_target], [ms_cv_out, ms_cv_st])
                ms_rs_btn.click(m_resize, [ms_rs_in, ms_rs_w, ms_rs_h], [ms_rs_out, ms_cv_st])
                ms_gif_btn.click(m_gif, [ms_rs_in, ms_gif_fps, ms_gif_w], [ms_gif_out, ms_cv_st])

            with gr.Accordion("➕  Merge clips", open=False):
                with gr.Row():
                    with gr.Column():
                        ms_mg_in = gr.File(label="Files (2+)", file_count="multiple")
                        ms_mg_kind = gr.Radio(["Video", "Audio"], value="Video", label="Type")
                        ms_mg_btn = gr.Button("Merge", variant="primary")
                    with gr.Column():
                        ms_mg_out = gr.File(label="Merged")
                        ms_mg_st = gr.HTML("")
                ms_mg_btn.click(m_merge, [ms_mg_in, ms_mg_kind], [ms_mg_out, ms_mg_st])

            with gr.Accordion("🗂  Output history", open=False):
                with gr.Row():
                    ms_hist_refresh = gr.Button("Refresh", size="sm")
                    ms_hist_zip_btn = gr.Button("Download all (zip)", size="sm")
                ms_hist = gr.Files(label="This session's outputs")
                ms_hist_zip = gr.File(label="Zip")
                ms_hist_refresh.click(list_outputs, outputs=ms_hist)
                ms_hist_zip_btn.click(zip_outputs, outputs=ms_hist_zip)

            # ── Send-to wiring (set target value + switch tab) ───────────────
            ms_tv_send_relip.click(lambda p: (p, gr.Tabs(selected=0)),
                                   [ms_tv_out], [ed_video, tabs])
            ms_gf_send_face.click(lambda p: (p, gr.Tabs(selected=2)),
                                  [ms_gf_out], [fs_src, tabs])
            ms_bg_send_face.click(lambda p: (p, gr.Tabs(selected=2)),
                                  [ms_bg_out], [fs_src, tabs])

    vram = gr.HTML(vram_html())
    for b in [ed_relip, ti_btn, fs_btn, lx_btn, ie_btn]:
        b.click(vram_html, outputs=vram)


if __name__ == "__main__":
    share = os.environ.get("IMAGE_TALK_SHARE", "1") == "1"
    _fav = os.path.join(PROJECT_ROOT, "assets", "favicon.svg")
    demo.queue(max_size=4)
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("IMAGE_TALK_PORT", "7860")),
        share=share,
        show_error=True,
        favicon_path=_fav if os.path.exists(_fav) else None,
    )
