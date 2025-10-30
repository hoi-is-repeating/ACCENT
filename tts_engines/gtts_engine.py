# tts_engines/gtts_engine.py
"""
gTTS Engine Implementation
Simple TTS using Google Text-to-Speech
"""

from pathlib import Path


class GTTSEngine:
    """Google TTS engine wrapper"""

    def __init__(self, language='es'):
        self.language = language
        self.gtts_available = False

        try:
            from gtts import gTTS
            self.gtts_available = True
            self.gTTS = gTTS
        except ImportError:
            print("⚠️ gTTS not available. Install: pip install gtts")

    def synthesize(self, text, output_path):
        """Generate TTS audio"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.gtts_available:
            tts = self.gTTS(text=text, lang=self.language, slow=False)
            tts.save(str(output_path))
            return True
        else:
            # Create empty file as placeholder
            output_path.touch()
            return False