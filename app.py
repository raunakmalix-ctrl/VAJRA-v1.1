"""
Image-Talk v1.1 — AI Media Studio (Colab / Gradio)

Run:  python app.py        (set IMAGE_TALK_SHARE=1 for a public link)
"""
import os
import sys

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

diffusion   = ENGINES["diffusion"]
faceswap    = ENGINES["faceswap"]
voice       = ENGINES["voice"]
talkingface = ENGINES["talkingface"]
transcript  = ENGINES["transcript"]

LANGS = {"English": "en", "Hindi": "hi"}


# ── GPU helpers ─────────────────────────────────────────────────────────────
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
            f"<span class='vram-text'>⬡ IMAGE·TALK &nbsp;·&nbsp; v1.1</span>"
            f"<span class='vram-text vram-accent'>{vram_status()}</span></div>")


def ok(msg):  return f"<span class='status-ok'>✔ {msg}</span>"

def err(msg):
    # Print the full traceback to the Colab cell when called inside an except.
    import sys, traceback
    if sys.exc_info()[0] is not None:
        traceback.print_exc()
    return f"<span class='status-err'>✖ {msg}</span>"
def warn(msg): return f"<span class='status-warn'>⚠ {msg}</span>"


# ── Feature 1: Talking video ────────────────────────────────────────────────
def run_talking_video(image, ref_audio, text, language, size, enhance, still,
                      progress=gr.Progress()):
    if image is None:
        return None, warn("Upload a portrait image")
    if ref_audio is None:
        return None, warn("Upload reference audio for the voice")
    if not text or not text.strip():
        return None, warn("Enter the text to speak")
    try:
        free_inprocess()
        progress(0.15, desc="Cloning voice & synthesizing speech ...")
        wav = voice.run(text=text, reference_audio_path=ref_audio,
                        language=LANGS.get(language, "en"))
        progress(0.55, desc="Animating portrait (SadTalker) ...")
        out = talkingface.run(portrait_path=image, audio_path=wav,
                              size=int(size), enhance_face=enhance,
                              still_mode=still)
        return out, ok(os.path.basename(out))
    except Exception as e:
        return None, err(str(e))


# ── Feature 2: Transcript edit + relip ──────────────────────────────────────
def do_extract(video, progress=gr.Progress()):
    if video is None:
        return "", None, warn("Upload a video first")
    try:
        free_inprocess()
        progress(0.3, desc="Transcribing (WhisperX) ...")
        text, state = transcript.extract_transcript(video)
        return text, state, ok(f"{len(state['segments'])} segments · "
                               f"lang={state['language']}")
    except Exception as e:
        return "", None, err(str(e))


def do_relip(state, edited_text, method, steps, guidance,
             progress=gr.Progress()):
    if not state:
        return None, warn("Extract a transcript first")
    if not edited_text or not edited_text.strip():
        return None, warn("Transcript is empty")
    try:
        free_inprocess()
        out = transcript.apply_edits(
            state, edited_text,
            method=("latentsync" if method.startswith("LatentSync") else "wav2lip"),
            inference_steps=int(steps), guidance_scale=float(guidance),
            progress=progress,
        )
        return out, ok(os.path.basename(out))
    except Exception as e:
        return None, err(str(e))


# ── Feature 3: Text → image ─────────────────────────────────────────────────
def run_txt2img(prompt, variant, width, height, steps, guidance, seed):
    if not prompt or not prompt.strip():
        return None, warn("Prompt required")
    try:
        free_inprocess()
        v = {"SDXL (open)": "sdxl",
             "FLUX Schnell (token)": "schnell",
             "FLUX Dev (token)": "dev"}.get(variant, "sdxl")
        out = diffusion.run(prompt=prompt, variant=v,
                            width=int(width), height=int(height),
                            steps=int(steps), guidance=float(guidance),
                            seed=int(seed))
        return out, ok(os.path.basename(out))
    except Exception as e:
        return None, err(str(e))


# ── Feature 4: Face swap ────────────────────────────────────────────────────
def run_faceswap(source, mode, target_img, target_vid, enhance,
                 progress=gr.Progress()):
    if source is None:
        return None, None, warn("Upload a source face")
    is_video = (mode == "Video")
    target = target_vid if is_video else target_img
    if target is None:
        return None, None, warn(f"Upload a target {'video' if is_video else 'image'}")
    try:
        free_inprocess()
        faceswap.load()
        out = faceswap.run(source_path=source, target_path=target,
                           enhance=enhance, is_video=is_video,
                           progress=progress if is_video else None)
        if is_video:
            return None, out, ok(os.path.basename(out))
        return out, None, ok(os.path.basename(out))
    except Exception as e:
        return None, None, err(str(e))


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


# ════════════════════════════════════════════════════════════════════════════
#  UI
# ════════════════════════════════════════════════════════════════════════════
from app_theme import CSS, THEME_JS, MASTHEAD   # noqa: E402

with gr.Blocks(css=CSS, title="Image-Talk", analytics_enabled=False) as demo:
    gr.HTML(f"<script>{THEME_JS}</script>")
    gr.HTML(MASTHEAD)

    with gr.Tabs():

        # ── 01 Talking Video ────────────────────────────────────────────────
        with gr.Tab("01 · Talking Video"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    gr.HTML("<div class='section-label'>Portrait &amp; Voice</div>")
                    tv_img   = gr.Image(label="Portrait (clear frontal face)",
                                        type="filepath", elem_classes=["output-media"])
                    tv_audio = gr.Audio(label="Reference voice (5–30s to clone)",
                                        type="filepath")
                    tv_ainfo = gr.HTML("")
                    tv_text  = gr.Textbox(label="Text to speak", lines=4,
                        placeholder="Type what the person should say (English or Hindi)…")
                    tv_lang  = gr.Radio(["English", "Hindi"], value="English",
                                        label="Language")
                    with gr.Row():
                        tv_size = gr.Radio(["256", "512"], value="512",
                                           label="Resolution")
                        tv_enh  = gr.Checkbox(label="GFPGAN enhance", value=True)
                        tv_still = gr.Checkbox(label="Still mode", value=False)
                    tv_btn = gr.Button("▶  Generate Talking Video", variant="primary")
                with gr.Column(scale=1):
                    gr.HTML("<div class='section-label'>Output</div>")
                    tv_out = gr.Video(label="", elem_classes=["output-media"])
                    tv_status = gr.HTML(ok("Ready"))
            tv_audio.change(audio_info, [tv_audio], [tv_ainfo])
            tv_btn.click(run_talking_video,
                         [tv_img, tv_audio, tv_text, tv_lang, tv_size, tv_enh, tv_still],
                         [tv_out, tv_status])

        # ── 02 Edit & Relip ─────────────────────────────────────────────────
        with gr.Tab("02 · Edit & Relip"):
            ed_state = gr.State(None)
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
                        ed_method = gr.Radio(["LatentSync (best)", "Wav2Lip (fast)"],
                                             value="LatentSync (best)", label="Lip-sync")
                        ed_steps  = gr.Slider(10, 50, value=20, step=1,
                                              label="Diffusion steps")
                        ed_guid   = gr.Slider(1.0, 3.0, value=1.5, step=0.1,
                                              label="Guidance")
                    ed_relip = gr.Button("▶  Apply Edits & Re-sync", variant="primary")
                with gr.Column(scale=1):
                    gr.HTML("<div class='section-label'>Output</div>")
                    ed_out = gr.Video(label="", elem_classes=["output-media"])
                    ed_status = gr.HTML(ok("Ready"))
            ed_extract.click(do_extract, [ed_video], [ed_text, ed_state, ed_status])
            ed_relip.click(do_relip,
                           [ed_state, ed_text, ed_method, ed_steps, ed_guid],
                           [ed_out, ed_status])

        # ── 03 Text → Image ─────────────────────────────────────────────────
        with gr.Tab("03 · Text → Image"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    gr.HTML("<div class='section-label'>Prompt</div>")
                    ti_prompt = gr.Textbox(label="Prompt", lines=4,
                        placeholder="A cinematic portrait, golden hour, ultra-detailed…")
                    ti_variant = gr.Radio(
                        ["SDXL (open)", "FLUX Schnell (token)", "FLUX Dev (token)"],
                        value="SDXL (open)", label="Model  (FLUX needs HF_TOKEN + license)")
                    with gr.Row():
                        ti_w = gr.Slider(512, 1536, value=1024, step=64, label="Width")
                        ti_h = gr.Slider(512, 1536, value=1024, step=64, label="Height")
                    with gr.Row():
                        ti_steps = gr.Slider(4, 50, value=28, step=1, label="Steps")
                        ti_guid  = gr.Slider(0.0, 7.0, value=3.5, step=0.5, label="Guidance")
                    ti_seed = gr.Number(label="Seed (−1 = random)", value=-1, precision=0)
                    ti_btn  = gr.Button("▶  Generate", variant="primary")
                with gr.Column(scale=1):
                    gr.HTML("<div class='section-label'>Output</div>")
                    ti_out = gr.Image(label="", elem_classes=["output-media"])
                    ti_status = gr.HTML(ok("Ready"))
            ti_btn.click(run_txt2img,
                         [ti_prompt, ti_variant, ti_w, ti_h, ti_steps, ti_guid, ti_seed],
                         [ti_out, ti_status])

        # ── 04 Face Swap ────────────────────────────────────────────────────
        with gr.Tab("04 · Face Swap"):
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
                    fs_enh  = gr.Checkbox(label="GFPGAN enhance", value=True)
                    fs_btn  = gr.Button("▶  Swap Face", variant="primary")
                with gr.Column(scale=1):
                    gr.HTML("<div class='section-label'>Output</div>")
                    fs_oimg = gr.Image(label="", visible=True, elem_classes=["output-media"])
                    fs_ovid = gr.Video(label="", visible=False, elem_classes=["output-media"])
                    fs_status = gr.HTML(ok("Ready"))
            fs_mode.change(toggle_swap_mode, [fs_mode],
                           [fs_timg, fs_tvid, fs_oimg, fs_ovid])
            fs_btn.click(run_faceswap,
                         [fs_src, fs_mode, fs_timg, fs_tvid, fs_enh],
                         [fs_oimg, fs_ovid, fs_status])

    vram = gr.HTML(vram_html())
    for b in [tv_btn, ed_relip, ti_btn, fs_btn]:
        b.click(vram_html, outputs=vram)


if __name__ == "__main__":
    share = os.environ.get("IMAGE_TALK_SHARE", "1") == "1"
    demo.queue(max_size=4)
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("IMAGE_TALK_PORT", "7860")),
        share=share,
        show_error=True,
    )
