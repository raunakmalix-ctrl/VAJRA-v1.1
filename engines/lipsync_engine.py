"""
Re-sync the lips of a real video to a new audio track.

Primary:  LatentSync (current SOTA diffusion lip-sync) via venv_latentsync.
Fallback: Wav2Lip GAN (same venv) if LatentSync fails.

Both tools ship their own CLIs; we invoke them with the isolated venv's python
and return the muxed output path.

Two modes:

  run()           Re-sync the whole video. Correct when most of the audio
                  changed, or when no edit windows are known.
  run_windowed()  Re-sync ONLY the time windows that changed, and composite the
                  result into the original frames through a feathered mouth mask.
                  Use this after a transcript edit: replacing four words should
                  not put ten minutes of footage through a generative model.

Why windowed is not merely faster. Model time drops in proportion to how little
changed -- typically well under 1% of a long clip -- but the more important
property is that untouched frames are never regenerated at all, so no colour
shift, softening or pose jitter can reach them. See vision/composite.py.
"""
import os
import json
import shutil
import subprocess
import tempfile

from core.base_engine import BaseEngine
from core.utils import timestamp_file, to_wav
from core.subprocess_runner import clean_env
from core.config import (
    VENV_LATENTSYNC_PY, LATENTSYNC_DIR, LATENTSYNC_CKPT, LATENTSYNC_CONFIG,
    WAV2LIP_DIR, WAV2LIP_CKPT, FFMPEG_PATH, FFPROBE_PATH,
)

# Lip-sync models need surrounding frames for a stable mouth, and the audio
# splice's crossfade extends slightly past the edit, so the visual window must
# cover a little more than the audio edit did.
DEFAULT_PAD_SEC = 0.35
# Two windows closer than this cost two model invocations and two pairs of
# temporal seams; one slightly longer window is cheaper and cleaner.
DEFAULT_MERGE_GAP_SEC = 0.75
# Past this fraction of the clip, windowing stops paying for itself and the
# whole-video path is simpler and produces one consistent result.
WINDOW_COVERAGE_LIMIT = 0.55


def probe_video(path):
    """(fps, n_frames, width, height, duration_sec) via ffprobe."""
    out = subprocess.run(
        [FFPROBE_PATH, "-v", "error", "-select_streams", "v:0",
         "-show_entries",
         "stream=r_frame_rate,nb_read_packets,width,height:format=duration",
         "-count_packets", "-of", "json", path],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {out.stderr[-500:]}")
    data = json.loads(out.stdout or "{}")
    st = (data.get("streams") or [{}])[0]
    rate = st.get("r_frame_rate") or "25/1"
    try:
        num, den = rate.split("/")
        fps = float(num) / float(den or 1)
    except Exception:
        fps = 25.0
    n_frames = int(st.get("nb_read_packets") or 0)
    w = int(st.get("width") or 0)
    h = int(st.get("height") or 0)
    dur = float((data.get("format") or {}).get("duration") or 0.0)
    if n_frames <= 0 and dur > 0:
        n_frames = int(round(dur * fps))
    return fps, n_frames, w, h, dur


class LipSyncEngine(BaseEngine):

    def run(self, video_path, audio_path, method="latentsync",
            inference_steps=20, guidance_scale=1.5):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        if not os.path.exists(VENV_LATENTSYNC_PY):
            raise RuntimeError(
                "Lip-sync environment not built. In the notebook, set "
                "LIPSYNC = True in Step 6 and run that cell "
                "(or: bash setup/make_venvs.sh lipsync)."
            )

        wav_path, is_tmp = to_wav(audio_path)
        tmp = [wav_path] if is_tmp else []
        try:
            if method == "latentsync":
                try:
                    rendered = self._run_latentsync(
                        video_path, wav_path, inference_steps, guidance_scale)
                    return self._remux_master(rendered, audio_path)
                except Exception as e:
                    print(f"[LipSync] LatentSync failed ({e}); "
                          f"falling back to Wav2Lip.")
            return self._remux_master(
                self._run_wav2lip(video_path, wav_path), audio_path)
        finally:
            for f in tmp:
                try:
                    os.unlink(f)
                except Exception:
                    pass

    # ── windowed lip-sync ───────────────────────────────────────────────────
    def run_windowed(self, video_path, audio_path, windows,
                     method="latentsync", inference_steps=20,
                     guidance_scale=1.5, pad_sec=DEFAULT_PAD_SEC,
                     merge_gap_sec=DEFAULT_MERGE_GAP_SEC,
                     mask_mouth=True, progress=None, report=None):
        """Re-sync only `windows` (list of (t_start, t_end) in seconds).

        Falls back to whole-video sync -- and says so -- when windowing would not
        pay for itself: no windows given, or the windows already cover most of
        the clip.

        report: optional dict, filled in with what was ACTUALLY done -- the
        merged windows, the coverage, and whether the whole video was synced
        after all. The caller cannot re-derive this without duplicating the
        fallback rules, and a duplicate could disagree with reality.

        Returns the output video path.
        """
        if report is None:
            report = {}
        report.update({"windowed": False, "windows": [], "coverage": 0.0,
                       "reason": ""})
        import numpy as np
        import vision

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        fps, n_frames, w, h, dur = probe_video(video_path)
        if fps <= 0 or n_frames <= 0 or w <= 0 or h <= 0:
            print("[LipSync] Could not probe the video reliably; "
                  "falling back to whole-video sync.")
            report["reason"] = "the video could not be probed reliably"
            return self.run(video_path, audio_path, method,
                            inference_steps, guidance_scale)

        wins = vision.merge_windows(windows, pad_sec=pad_sec,
                                    min_gap_sec=merge_gap_sec,
                                    duration_sec=dur)
        if not wins:
            print("[LipSync] No edit windows supplied; syncing the whole video.")
            report["reason"] = "no edit windows were supplied"
            return self.run(video_path, audio_path, method,
                            inference_steps, guidance_scale)

        cov = vision.coverage(wins, dur)
        if cov >= WINDOW_COVERAGE_LIMIT:
            print(f"[LipSync] Windows cover {cov*100:.0f}% of the clip; "
                  f"syncing the whole video instead (simpler and more "
                  f"consistent than stitching that many windows).")
            report["coverage"] = float(cov)
            report["reason"] = (f"edits cover {cov*100:.0f}% of the clip, so the "
                                f"whole video was synced instead")
            return self.run(video_path, audio_path, method,
                            inference_steps, guidance_scale)

        print(f"[LipSync] Windowed sync: {len(wins)} window(s), "
              f"{sum(b - a for a, b in wins):.2f}s of {dur:.2f}s "
              f"({cov*100:.2f}% of the video).")

        # Defined before first use: the windowing summary below references it,
        # and the compositor fills it in later during the frame loop.
        texture_report = {}
        report["texture"] = texture_report
        report.update({"windowed": True, "windows": [(float(a), float(b))
                                                     for a, b in wins],
                       "coverage": float(cov), "duration_sec": float(dur)})

        wav_path, is_tmp = to_wav(audio_path)
        workdir = tempfile.mkdtemp(prefix="vajra_relip_win_")
        cleanup = [wav_path] if is_tmp else []

        try:
            # 1. Sync each window independently, in its own sub-clip.
            synced_by_window = []
            for wi, (t0, t1) in enumerate(wins):
                if progress is not None:
                    progress(wi / max(len(wins), 1),
                             desc=f"Lip-sync window {wi+1}/{len(wins)}")
                seg_v = os.path.join(workdir, f"win{wi}_v.mp4")
                seg_a = os.path.join(workdir, f"win{wi}_a.wav")
                _cut_video(video_path, t0, t1, seg_v)
                _cut_audio(wav_path, t0, t1, seg_a)

                try:
                    out_seg = self.run(seg_v, seg_a, method,
                                       inference_steps, guidance_scale)
                except Exception as e:
                    # One bad window must not lose the whole edit; the original
                    # frames for that window are simply kept.
                    print(f"[LipSync] window {wi+1} failed ({e}); "
                          f"keeping original frames there.")
                    synced_by_window.append(None)
                    continue
                synced_by_window.append(out_seg)

            if all(s is None for s in synced_by_window):
                raise RuntimeError(
                    "Every lip-sync window failed. See the console for the "
                    "per-window errors above."
                )

            # 2. Composite. Original frames stream through untouched except
            #    inside a window, and inside a window only the mask area changes.
            out_path = timestamp_file("relip", "mp4")
            frame_windows = []
            for (t0, t1), seg in zip(wins, synced_by_window):
                i0, i1 = vision.frame_range(t0, t1, fps, n_frames)
                frame_windows.append((i0, i1, seg))

            # audio_path, NOT wav_path: wav_path is the 16 kHz copy made for
            # the model. Muxing that would throw away the master's full rate.
            _composite_video(video_path, frame_windows, out_path, audio_path,
                             fps, w, h, mask_mouth=mask_mouth,
                             texture_report=texture_report)
            print(f"[LipSync] Output: {out_path}")
            return out_path
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
            for f in cleanup:
                try:
                    os.unlink(f)
                except Exception:
                    pass

    @staticmethod
    def _remux_master(video_path, audio_path):
        """Put the FULL-RATE master audio onto a finished render.

        Both engines are fed a 16 kHz copy, because that is what their mel
        front-ends want -- and both bake that copy into their output. Left
        alone, the finished video would carry 16 kHz audio no matter what the
        rest of the pipeline preserved, so every edit would sound like a phone
        call. Re-muxing costs one stream copy and no re-encode of the video.

        Best-effort: if it fails, the render is returned unchanged rather than
        lost.
        """
        if not audio_path or not os.path.exists(audio_path):
            return video_path
        out = video_path.replace(".mp4", "_hq.mp4")
        r = subprocess.run(
            [FFMPEG_PATH, "-y", "-i", video_path, "-i", audio_path,
             "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
             "-map", "0:v:0", "-map", "1:a:0", "-shortest", out],
            capture_output=True, text=True,
        )
        if r.returncode == 0 and os.path.exists(out):
            os.replace(out, video_path)
        else:
            print("[LipSync] Master re-mux failed; keeping the engine's audio.")
        return video_path

    # ── LatentSync ──────────────────────────────────────────────────────────
    def _run_latentsync(self, video_path, wav_path, steps, guidance):
        out_path = timestamp_file("relip", "mp4")
        cmd = [
            VENV_LATENTSYNC_PY,
            os.path.join(LATENTSYNC_DIR, "scripts", "inference.py"),
            "--unet_config_path", LATENTSYNC_CONFIG,
            "--inference_ckpt_path", LATENTSYNC_CKPT,
            "--video_path", video_path,
            "--audio_path", wav_path,
            "--video_out_path", out_path,
            "--inference_steps", str(steps),
            "--guidance_scale", str(guidance),
        ]
        print("[LipSync] Running LatentSync ...")
        # LatentSync's script imports its own `latentsync` package from the repo
        # root, so put that on PYTHONPATH.
        proc = subprocess.run(
            cmd, cwd=LATENTSYNC_DIR, capture_output=True, text=True,
            env=clean_env({"PYTHONPATH": LATENTSYNC_DIR}),
        )
        if proc.returncode != 0 or not os.path.exists(out_path):
            raise RuntimeError(
                f"LatentSync exit {proc.returncode}\n"
                f"{proc.stdout[-800:]}\n{proc.stderr[-1500:]}"
            )
        print(f"[LipSync] Output: {out_path}")
        return out_path

    # ── Wav2Lip fallback ──────────────────────────────────────────────────────
    def _run_wav2lip(self, video_path, wav_path):
        out_path = timestamp_file("relip", "mp4")
        cmd = [
            VENV_LATENTSYNC_PY,
            os.path.join(WAV2LIP_DIR, "inference.py"),
            "--checkpoint_path", WAV2LIP_CKPT,
            "--face", video_path,
            "--audio", wav_path,
            "--outfile", out_path,
            "--nosmooth",
            "--pads", "0", "20", "0", "0",
        ]
        print("[LipSync] Running Wav2Lip (fallback) ...")
        proc = subprocess.run(cmd, cwd=WAV2LIP_DIR,
                              capture_output=True, text=True, env=clean_env())
        if proc.returncode != 0 or not os.path.exists(out_path):
            raise RuntimeError(
                f"Wav2Lip exit {proc.returncode}\n"
                f"{proc.stdout[-800:]}\n{proc.stderr[-1500:]}"
            )

        # Ensure the output carries an audio track.
        probe = subprocess.run(
            [FFPROBE_PATH, "-v", "quiet", "-select_streams", "a",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", out_path],
            capture_output=True, text=True,
        )
        if "audio" not in probe.stdout:
            muxed = out_path.replace(".mp4", "_a.mp4")
            r = subprocess.run(
                [FFMPEG_PATH, "-y", "-i", out_path, "-i", wav_path,
                 "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                 "-shortest", muxed],
                capture_output=True, text=True,
            )
            if r.returncode == 0 and os.path.exists(muxed):
                os.replace(muxed, out_path)
        print(f"[LipSync] Output: {out_path}")
        return out_path


# ── ffmpeg helpers for the windowed path ─────────────────────────────────────
def _cut_video(src, t0, t1, dst):
    """Extract [t0, t1) as a re-encoded clip.

    Re-encoded rather than stream-copied on purpose: a stream copy can only cut
    at keyframes, so it would silently shift the window by up to a whole GOP and
    the synced mouth would land on the wrong frames.
    """
    r = subprocess.run(
        [FFMPEG_PATH, "-y", "-ss", f"{t0:.4f}", "-to", f"{t1:.4f}",
         "-i", src, "-an", "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "16", "-pix_fmt", "yuv420p", dst],
        capture_output=True, text=True,
    )
    if r.returncode != 0 or not os.path.exists(dst):
        raise RuntimeError(f"ffmpeg window cut failed: {r.stderr[-600:]}")
    return dst


def _cut_audio(src, t0, t1, dst):
    r = subprocess.run(
        [FFMPEG_PATH, "-y", "-ss", f"{t0:.4f}", "-to", f"{t1:.4f}",
         "-i", src, "-ac", "1", dst],
        capture_output=True, text=True,
    )
    if r.returncode != 0 or not os.path.exists(dst):
        raise RuntimeError(f"ffmpeg audio cut failed: {r.stderr[-600:]}")
    return dst


def _read_frames(path, w, h):
    """Yield raw RGB frames from `path` scaled to (w, h)."""
    import numpy as np
    proc = subprocess.Popen(
        [FFMPEG_PATH, "-v", "error", "-i", path,
         "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-vf", f"scale={w}:{h}", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    size = w * h * 3
    try:
        while True:
            buf = proc.stdout.read(size)
            if not buf or len(buf) < size:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 3))
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        proc.wait()


def _composite_video(src_video, frame_windows, out_path, wav_path, fps, w, h,
                     mask_mouth=True, texture_report=None):
    """Stream `src_video`'s frames to a new file, swapping in synced mouths.

    texture_report: optional dict, filled in with what the per-frame grain and
    sharpness matching actually did. Collected here because it happens inside
    the frame loop, and reported outward because a correction the operator
    cannot see is indistinguishable from one that never ran.

    frame_windows: [(i0, i1, synced_clip_path_or_None), ...] sorted by i0.

    Frames outside every window are passed through unmodified. Inside a window,
    the synced frame is composited through a feathered mask derived from what the
    model actually changed, with a temporal ramp at the window edges.
    """
    import numpy as np
    import vision

    # Pre-load each window's synced frames. Windows are a tiny fraction of the
    # clip, so this is bounded -- and it is what lets the mask be derived from
    # the whole window's difference rather than one frame at a time.
    prepared = []
    for (i0, i1, seg) in frame_windows:
        if seg is None:
            continue
        frames = list(_read_frames(seg, w, h))
        if not frames:
            print(f"[LipSync] window at frame {i0} produced no frames; skipping.")
            continue
        prepared.append({"i0": i0, "i1": i1, "frames": frames,
                         "mask": None, "base": []})

    enc = subprocess.Popen(
        [FFMPEG_PATH, "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
         "-r", f"{fps:.6f}", "-i", "-",
         "-i", wav_path,
         "-c:v", "libx264", "-preset", "medium", "-crf", "17",
         "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k",
         "-map", "0:v:0", "-map", "1:a:0", "-shortest",
         out_path],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )

    def _window_for(idx):
        for p in prepared:
            if p["i0"] <= idx < p["i1"]:
                return p
        return None

    # First pass over each window's base frames is needed before the mask can be
    # derived, so buffer a window's originals as we reach it.
    try:
        for idx, frame in enumerate(_read_frames(src_video, w, h)):
            p = _window_for(idx)
            if p is None:
                enc.stdin.write(frame.tobytes())
                continue

            k = idx - p["i0"]
            if k < len(p["frames"]):
                p["base"].append(frame)
                # Derive the mask once, from the whole window, at its end.
                if p["mask"] is None and (idx + 1 >= p["i1"]
                                          or k + 1 >= len(p["frames"])):
                    box = (vision.changed_region_box(p["base"], p["frames"])
                           if mask_mouth else None)
                    if box is None:
                        p["mask"] = "full"
                        why = ("masking declined (change is global or "
                               "negligible)" if mask_mouth else "masking off")
                        print(f"[LipSync] window at frame {p['i0']}: {why}; "
                              f"using synced frames within the window.")
                    else:
                        p["mask"] = vision.feathered_ellipse_mask(h, w, box)
                        print(f"[LipSync] window at frame {p['i0']}: mouth mask "
                              f"{int(box[2]-box[0])}x{int(box[3]-box[1])} px.")
                    n = min(len(p["base"]), len(p["frames"]))
                    if p["mask"] == "full":
                        outs = [vision.blend(
                            p["base"][j], p["frames"][j],
                            np.ones((h, w), np.float32),
                            strength=vision.temporal_weight(j, n))
                            for j in range(n)]
                    else:
                        outs = vision.composite_window(
                            p["base"][:n], p["frames"][:n], p["mask"],
                            report=(texture_report
                                    if texture_report is not None else None))
                    for o in outs:
                        enc.stdin.write(np.ascontiguousarray(o).tobytes())
                    # Any base frames beyond the synced clip pass through.
                    for j in range(n, len(p["base"])):
                        enc.stdin.write(p["base"][j].tobytes())
                    p["base"] = []
                    p["frames"] = []
            else:
                # Window longer than the synced clip: keep the original frame.
                enc.stdin.write(frame.tobytes())

        # Flush any window that never hit its end condition.
        for p in prepared:
            for f in p["base"]:
                enc.stdin.write(f.tobytes())
    finally:
        try:
            enc.stdin.close()
        except Exception:
            pass
        _, err = enc.communicate()

    if enc.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(
            f"ffmpeg composite encode failed (exit {enc.returncode}): "
            f"{(err or b'').decode('utf-8', 'replace')[-800:]}"
        )
    return out_path
