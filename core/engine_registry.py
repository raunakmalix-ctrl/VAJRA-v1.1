from engines.diffusion_engine import DiffusionEngine
from engines.faceswap_engine import FaceSwapEngine
from engines.voice_engine import VoiceEngine
from engines.lipsync_engine import LipSyncEngine
from engines.transcript_engine import TranscriptEngine
from engines.ltx_engine import LTXEngine
from engines.musetalk_engine import MuseTalkEngine
from engines.media_engine import MediaEngine
from engines.motion_engine import MotionVideoEngine

ENGINES = {
    "diffusion":   DiffusionEngine(),
    "faceswap":    FaceSwapEngine(),
    "voice":       VoiceEngine(),
    "lipsync":     LipSyncEngine(),
    "transcript":  TranscriptEngine(),
    "ltx":         LTXEngine(),
    "musetalk":    MuseTalkEngine(),
    "media":       MediaEngine(),
    "motion":      MotionVideoEngine(),
}
