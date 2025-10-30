# asr_models/whisper_asr.py
"""
Whisper ASR Model Implementation
OpenAI Whisper integration for speech recognition
"""

from pathlib import Path


class WhisperASR:
    """Whisper ASR model wrapper"""

    def __init__(self, model_size='base'):
        self.model_size = model_size
        self.model = None
        self.whisper_available = False

        try:
            import whisper
            self.whisper_available = True
            self.whisper = whisper
        except ImportError:
            print("⚠️ Whisper not available. Install: pip install openai-whisper")

    def _load_model(self):
        """Lazy load the model"""
        if self.whisper_available and self.model is None:
            print(f"Loading Whisper {self.model_size} model...")
            self.model = self.whisper.load_model(self.model_size)
            print("✅ Model loaded")

    def transcribe(self, audio_path, language='es'):
        """Transcribe audio file"""
        audio_path = Path(audio_path)

        if not audio_path.exists():
            return ""

        if self.whisper_available:
            self._load_model()
            if self.model:
                result = self.model.transcribe(
                    str(audio_path),
                    language=language,
                    temperature=0.0,
                    beam_size=5
                )
                return result['text'].strip()

        # Mock transcription if Whisper not available
        return f"mock transcription for {audio_path.name}"