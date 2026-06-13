import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from faster_whisper import WhisperModel

WAV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input.wav")

for size in ["base", "small", "medium"]:
    m = WhisperModel(size, device="cuda", compute_type="float16")
    segs, info = m.transcribe(WAV)
    txt = " ".join(s.text.strip() for s in segs).strip()
    print(f"[whisper-{size}] lang={info.language} p={info.language_probability:.2f}: {txt}")
