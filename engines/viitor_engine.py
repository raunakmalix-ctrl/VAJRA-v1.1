"""
ViiTorVoice-NAR — tier A local speech infill, and voice cloning.

What makes this different from every other engine here. XTTS is a sentence
synthesiser: it cannot hear the recording it is editing, so it regenerates a
whole phrase which then has to be disguised to match its surroundings (the whole
reason dsp/ exists) and which may not even fit the slot it has to occupy. This
model completes MASKED audio tokens conditioned on the real audio either side,
so it regenerates only the changed words, in place, already matching the voice,
room and level around them because it heard them.

The integration cost is that it ships as five gRPC services behind an HTTP
gateway rather than as a library. So this engine supervises a process group and
speaks HTTP to localhost, instead of the run-a-worker-in-a-venv pattern the
other engines use. Consequences worth knowing:

  * Startup is slow (five models) and is therefore lazy and cached -- the group
    is started on first use and left running, not restarted per call.
  * Readiness is polled on /health rather than assumed after spawn, because the
    gateway binds its port before the models behind it have finished loading.
  * The upstream aligner defaults to Chinese. This deployment is English-only by
    choice, so the language is set explicitly rather than left to default; a
    silent zh alignment on English audio would degrade every edit.
"""
import os
import subprocess
import time

from core.base_engine import BaseEngine
from core.utils import timestamp_file
from core.config import (VIITOR_DIR, VENV_VIITOR_PY, VIITOR_MODELS,
                         VIITOR_BASE_URL, VIITOR_HTTP_PORT, VIITOR_HOST,
                         VIITOR_LANGUAGE, VIITOR_START_TIMEOUT_SEC,
                         VIITOR_REQUEST_TIMEOUT_SEC, VENV_ROOT)

# Only English is enabled. The model also supports Chinese upstream, but every
# language claimed here has to be one an operator can actually rely on, and the
# router reports coverage from this set.
LANGUAGES = ("en",)

_LAUNCHER = "run_grpc_v2.sh"


def _repo_ok():
    return os.path.isfile(os.path.join(VIITOR_DIR, _LAUNCHER))


def available():
    """True when the environment, the repo and the weights are all present.

    All three are required: the venv alone runs nothing, and the launcher exits
    immediately without weights. Reporting availability on a partial install
    would make the router promise a tier it cannot deliver.
    """
    return (os.path.exists(VENV_VIITOR_PY) and _repo_ok()
            and os.path.isdir(VIITOR_MODELS)
            and bool(os.listdir(VIITOR_MODELS)))


def _service_env():
    """Environment that points the upstream launcher at OUR paths.

    Every value in its deploy.env is written as ${VAR:-default}, so exporting
    these overrides them without patching upstream files -- which would conflict
    on the next `git pull` of that repo.
    """
    env = dict(os.environ)
    env.update({
        "VIITORVOICE_V2_VENV_DIR": os.path.dirname(
            os.path.dirname(VENV_VIITOR_PY)),
        "VIITORVOICE_V2_STATE_DIR": os.path.join(VENV_ROOT, ".viitor-state"),
        "VIITORVOICE_LOCAL_MODELS": VIITOR_MODELS,
        "VIITORVOICE_V2_HTTP_PORT": str(VIITOR_HTTP_PORT),
        "VIITORVOICE_V2_TARGET_HOST": VIITOR_HOST,
        # The default is 'zh'; aligning English audio with a Chinese aligner
        # would silently mislocate every edit.
        "VIITORVOICE_ALIGNER_LANGUAGE": VIITOR_LANGUAGE,
        "VIITORVOICE_V2_REQUEST_TIMEOUT_SEC": str(VIITOR_REQUEST_TIMEOUT_SEC),
    })
    return env


def _health(timeout=2.0):
    import urllib.request
    import urllib.error
    try:
        with urllib.request.urlopen(f"{VIITOR_BASE_URL}/health",
                                    timeout=timeout) as r:
            return 200 <= r.status < 300
    except Exception:
        return False


class ViitorEngine(BaseEngine):

    def __init__(self):
        self._started = False

    # ── service lifecycle ────────────────────────────────────────────────
    def is_running(self):
        return _health()

    def start(self, timeout=None, progress=None):
        """Start the service group and wait until it actually answers.

        Idempotent: an already-healthy group is left alone, so this is safe to
        call before every request.
        """
        if _health():
            self._started = True
            return True
        if not available():
            raise RuntimeError(_unavailable_message())

        timeout = int(timeout or VIITOR_START_TIMEOUT_SEC)
        print("[ViiTor] starting the service group (five models — the first "
              "start takes a few minutes)")
        proc = subprocess.run(
            ["bash", _LAUNCHER, "start", "all"],
            cwd=VIITOR_DIR, env=_service_env(),
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-12:]
            raise RuntimeError(
                "ViiTorVoice services failed to start:\n  "
                + "\n  ".join(tail))

        # The gateway binds its port before the models behind it are loaded, so
        # a successful connect is not readiness -- poll /health.
        deadline = time.time() + timeout
        waited = 0
        while time.time() < deadline:
            if _health():
                self._started = True
                print(f"[ViiTor] ready after {waited}s")
                return True
            time.sleep(3)
            waited += 3
            if progress is not None and waited % 15 == 0:
                progress(min(0.9, waited / float(timeout)),
                         desc=f"Loading ViiTorVoice models ({waited}s)")
        raise RuntimeError(
            f"ViiTorVoice did not become healthy within {timeout}s. Check the "
            f"service logs:  bash {_LAUNCHER} logs orchestrator  (in "
            f"{VIITOR_DIR})")

    def stop(self):
        if not _repo_ok():
            return False
        subprocess.run(["bash", _LAUNCHER, "stop", "all"],
                       cwd=VIITOR_DIR, env=_service_env(),
                       capture_output=True, text=True)
        self._started = False
        return True

    # ── inference ────────────────────────────────────────────────────────
    def local_edit(self, source_audio_path, original_text, edited_text,
                   language=None, align_granularity="word",
                   expand_mask_ratio=0.1, progress=None):
        """Regenerate ONLY the changed words, conditioned on the real audio.

        Returns the path to the edited audio. Unlike the tier C path, the result
        needs no spectral/room disguise: the model heard the surrounding
        recording, so what it produces already sits in it.
        """
        if not (edited_text or "").strip():
            raise ValueError("Edited text is empty.")
        self.start(progress=progress)
        fields = {
            "original_text": original_text or "",
            "edited_text": edited_text,
            "language": language or VIITOR_LANGUAGE,
            "align_granularity": align_granularity,
            "expand_mask_ratio": str(expand_mask_ratio),
            "output_format": "wav",
        }
        return self._post("/v1/text-local-edit", fields,
                          {"source_audio": source_audio_path},
                          "viitor_edit")

    def clone(self, ref_audio_path, text, language=None,
              allow_missing_ref_text=True, emotion_guidance_scale=None,
              nvv_guidance_scale=None, progress=None):
        """Speak `text` in the voice of `ref_audio_path`."""
        if not (text or "").strip():
            raise ValueError("Text is empty.")
        if not ref_audio_path or not os.path.exists(ref_audio_path):
            raise ValueError("A reference audio clip is required.")
        self.start(progress=progress)
        fields = {
            "text": text,
            "language": language or VIITOR_LANGUAGE,
            "allow_missing_ref_text": "true" if allow_missing_ref_text else "false",
        }
        if emotion_guidance_scale is not None:
            fields["emotion_guidance_scale"] = str(emotion_guidance_scale)
        if nvv_guidance_scale is not None:
            fields["nvv_guidance_scale"] = str(nvv_guidance_scale)
        return self._post("/v1/voice-clone", fields,
                          {"ref_audio": ref_audio_path}, "viitor_clone")

    # ── HTTP ─────────────────────────────────────────────────────────────
    def _post(self, path, fields, files, out_stem):
        body, content_type = _multipart(fields, files)
        import urllib.request
        import urllib.error
        req = urllib.request.Request(
            VIITOR_BASE_URL + path, data=body,
            headers={"Content-Type": content_type})
        try:
            with urllib.request.urlopen(
                    req, timeout=VIITOR_REQUEST_TIMEOUT_SEC) as r:
                payload = r.read()
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", "replace")[:400]
            except Exception:
                pass
            raise RuntimeError(
                f"ViiTorVoice {path} returned HTTP {e.code}. {detail}") from None
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Could not reach ViiTorVoice at {VIITOR_BASE_URL} ({e.reason}). "
                f"The service group may have died — check "
                f"`bash {_LAUNCHER} status all` in {VIITOR_DIR}.") from None

        if not payload:
            raise RuntimeError(f"ViiTorVoice {path} returned an empty response.")
        out = timestamp_file(out_stem, "wav")
        with open(out, "wb") as fh:
            fh.write(payload)
        return out

    def run(self, *a, **kw):
        """BaseEngine entry point — local editing is the primary use."""
        return self.local_edit(*a, **kw)


def _multipart(fields, files):
    """Build a multipart/form-data body without adding a dependency.

    `requests` would be shorter, but this engine runs in the principal runtime
    and adding a hard dependency there for one upload is not worth it.
    """
    import mimetypes
    import uuid

    boundary = "----vajra" + uuid.uuid4().hex
    out = bytearray()
    for key, value in fields.items():
        out += (f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n'
                f"{value}\r\n").encode("utf-8")
    for key, path in files.items():
        if not path or not os.path.exists(path):
            raise ValueError(f"{key}: file not found: {path}")
        name = os.path.basename(path)
        ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
        out += (f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{key}"; '
                f'filename="{name}"\r\n'
                f"Content-Type: {ctype}\r\n\r\n").encode("utf-8")
        with open(path, "rb") as fh:
            out += fh.read()
        out += b"\r\n"
    out += f"--{boundary}--\r\n".encode("utf-8")
    return bytes(out), f"multipart/form-data; boundary={boundary}"


def _unavailable_message():
    """Say which of the three prerequisites is missing, not just 'unavailable'."""
    missing = []
    if not os.path.exists(VENV_VIITOR_PY):
        missing.append("the environment (Step 6: set VIITOR = True)")
    if not _repo_ok():
        missing.append("the source repo (built by Step 6 as well)")
    if not os.path.isdir(VIITOR_MODELS) or not os.listdir(VIITOR_MODELS):
        missing.append("the weights (Step 7: add 'viitor' to DOWNLOAD)")
    return ("ViiTorVoice-NAR is not installed — missing "
            + "; ".join(missing) + ".")
