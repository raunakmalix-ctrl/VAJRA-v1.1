from engines.diffusion_engine import DiffusionEngine
from engines.faceswap_engine import FaceSwapEngine
from engines.voice_engine import VoiceEngine
from engines.lipsync_engine import LipSyncEngine
from engines.transcript_engine import TranscriptEngine
from engines.ltx2_engine import LTX2Engine
from engines.media_engine import MediaEngine
from engines.motion_engine import MotionVideoEngine
from engines.qwen_edit_engine import QwenEditEngine

ENGINES = {
    "diffusion":   DiffusionEngine(),
    "faceswap":    FaceSwapEngine(),
    "voice":       VoiceEngine(),
    "lipsync":     LipSyncEngine(),
    "transcript":  TranscriptEngine(),
    "ltx2":        LTX2Engine(),
    "media":       MediaEngine(),
    "motion":      MotionVideoEngine(),
    "qwen_edit":   QwenEditEngine(),
}
