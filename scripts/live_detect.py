print("✅ live_detect.py started...")

import torch
import torchaudio
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import numpy as np
import warnings
import sounddevice as sd
from pathlib import Path

warnings.filterwarnings("ignore")


class LiveLanguageDetector:
    def __init__(self, model_path=None):
        try:
            if model_path is None:
                model_path = Path(__file__).parent.parent / "models"
            model_path = Path(model_path).resolve()

            if not model_path.exists():
                raise FileNotFoundError(f"❌ Model path not found: {model_path}")

            print(f"📂 Loading model from local path: {model_path}")

            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
                str(model_path),
                local_files_only=True
            )
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
                str(model_path),
                local_files_only=True
            )
            self.model.eval()

            print("✅ Language detection model loaded successfully!")
            print(f"🌍 Supported languages: {list(self.model.config.id2label.values())}")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

    def detect_from_file(self, file_path: str) -> str:
        """Detect language from a given audio file (wav/mp3/flac)."""
        if not Path(file_path).exists():
            raise FileNotFoundError(f"❌ Audio file not found: {file_path}")

        speech, sr = torchaudio.load(file_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            speech = resampler(speech)
            sr = 16000

        input_values = self.processor(
            speech.squeeze().numpy(),
            sampling_rate=sr,
            return_tensors="pt",
            padding=True
        ).input_values

        with torch.no_grad():
            logits = self.model(input_values).logits
            predicted_id = torch.argmax(logits, dim=-1).item()

        return self.model.config.id2label[predicted_id]

    def detect_from_microphone(self, duration: int = 5, samplerate: int = 16000):
        """Detect language from live microphone audio."""
        print(f"\n🎤 Recording {duration}s of audio...")
        audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
        sd.wait()
        audio = audio.squeeze()

        if samplerate != 16000:
            resampler = torchaudio.transforms.Resample(samplerate, 16000)
            audio = resampler(torch.tensor(audio)).numpy()
            samplerate = 16000

        input_values = self.processor(
            audio,
            sampling_rate=samplerate,
            return_tensors="pt",
            padding=True
        ).input_values

        with torch.no_grad():
            logits = self.model(input_values).logits
            predicted_id = torch.argmax(logits, dim=-1).item()
            confidence = torch.nn.functional.softmax(logits, dim=-1)[0][predicted_id].item()

        predicted_lang = self.model.config.id2label[predicted_id]
        return predicted_lang, confidence


# --------------------------
# Live streaming usage
# --------------------------
if __name__ == "__main__":
    detector = LiveLanguageDetector()
    print("🎯 Live language detection (press Ctrl+C to stop)")
    try:
        while True:
            lang, conf = detector.detect_from_microphone(duration=5)
            print(f"Detected Language: {lang} ({conf:.2%} confidence)")
    except KeyboardInterrupt:
        print("\n🛑 Live detection stopped by user.")
