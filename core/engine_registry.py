from engines.diffusion_engine import DiffusionEngine
from engines.faceswap_engine import FaceSwapEngine
from engines.voice_engine import VoiceEngine
from engines.talkingface_engine import TalkingFaceEngine
from engines.lipsync_engine import LipSyncEngine
from engines.transcript_engine import TranscriptEngine

ENGINES = {
    "diffusion":   DiffusionEngine(),
    "faceswap":    FaceSwapEngine(),
    "voice":       VoiceEngine(),
    "talkingface": TalkingFaceEngine(),
    "lipsync":     LipSyncEngine(),
    "transcript":  TranscriptEngine(),
}
