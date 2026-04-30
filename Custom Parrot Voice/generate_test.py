from pathlib import Path
import csv
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

METADATA_PATH = Path("metadata_audio_1300.csv")
VOICE_PROMPT = Path("voice_prompt.wav")
OUTPUT_DIR = Path("wavs_test")
DEVICE = "cpu"
MAX_FILES = 10  # only generate the first 10 for testing

def read_metadata(path: Path):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="|", quotechar='"')
        for row in reader:
            if len(row) >= 2:
                stem = row[0].strip()
                text = row[1].strip()
                if stem and text:
                    rows.append((stem, text))
    return rows

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing metadata file: {METADATA_PATH}")

    if not VOICE_PROMPT.exists():
        raise FileNotFoundError(f"Missing voice prompt file: {VOICE_PROMPT}")

    rows = read_metadata(METADATA_PATH)[:MAX_FILES]
    print(f"Loaded {len(rows)} rows")

    model = ChatterboxTTS.from_pretrained(device=DEVICE)

    for i, (stem, text) in enumerate(rows, start=1):
        out_path = OUTPUT_DIR / f"{stem}.wav"
        print(f"[{i}/{len(rows)}] Generating {out_path.name} -> {text}")

        wav = model.generate(
            text,
            audio_prompt_path=str(VOICE_PROMPT),
        )

        ta.save(str(out_path), wav, model.sr)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()