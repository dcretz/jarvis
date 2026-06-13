import os
import warnings

# Suprima warnings Hugging Face
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
from transformers import pipeline, BitsAndBytesConfig, AutoProcessor

MODEL_ID = "google/gemma-4-E4B-it"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_PATH = os.path.join(BASE_DIR, "input.wav")
OUTPUT_PATH = os.path.join(BASE_DIR, "transcription.txt")
# Folder unde salvam modelul DEJA cuantizat (~5 GB) ca pornirile sa fie rapide.
QUANT_DIR = os.path.join(BASE_DIR, "gemma-4-e4b-4bit")

if os.path.isdir(QUANT_DIR):
    # Pornire RAPIDA: incarcam direct modelul 4-bit salvat (fara re-cuantizare).
    print(f"[Incarc modelul 4-bit salvat din {QUANT_DIR} ...]")
    pipe = pipeline(
        task="any-to-any",
        model=QUANT_DIR,
        device_map="auto",
    )
else:
    # Prima rulare: descarca bf16, cuantizeaza 4-bit, apoi SALVEAZA pentru data viitoare.
    print("[Prima rulare: cuantizez 4-bit si salvez pentru pornirile urmatoare ...]")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    pipe = pipeline(
        task="any-to-any",
        model=MODEL_ID,
        device_map="auto",
        model_kwargs={"quantization_config": quant_config},
    )
    # Salvam modelul cuantizat -> urmatoarele porniri sar peste cuantizare.
    pipe.model.save_pretrained(QUANT_DIR)
    # Processor-ul il salvam curat din modelul original (cel din pipe contine
    # quant_config ne-serializabil -> bug in transformers dev).
    AutoProcessor.from_pretrained(MODEL_ID).save_pretrained(QUANT_DIR)
    print(f"[Model 4-bit salvat in {QUANT_DIR}. Urmatoarele porniri vor fi rapide.]")

prompt = (
    "Transcribe the following speech segment into English text. "
    "Only output the transcription, with no explanation. "
    "If the speech is not in English, translate it to English."
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": AUDIO_PATH},
            {"type": "text", "text": prompt},
        ],
    }
]

out = pipe(text=messages, max_new_tokens=256)

# Extract just the assistant's transcription
transcription = out[0]["generated_text"][-1]["content"].strip()

# Afiseaza clar in terminal
print("\n" + "=" * 60)
print("TRANSCRIERE input.wav:")
print("=" * 60)
print(transcription)
print("=" * 60)

# Salveaza si in fisier ca sa nu se piarda printre mesaje
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(transcription + "\n")
print(f"\n[Salvat in: {OUTPUT_PATH}]")