#!/usr/bin/env python3
"""
ACCENT Complete Pipeline for Spanish ASR Experiments
Using LOCAL TTS (pyttsx3) - No rate limits!
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
import subprocess

warnings.filterwarnings('ignore')

# Fix the import paths
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "data_loaders"))


# Check for ffmpeg
def check_ffmpeg():
    """Check if ffmpeg is available"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        return True
    except FileNotFoundError:
        return False


FFMPEG_AVAILABLE = check_ffmpeg()

# TTS Engine - Try pyttsx3 first (local), fallback to gTTS
TTS_AVAILABLE = False
TTS_TYPE = None

try:
    import pyttsx3

    TTS_AVAILABLE = True
    TTS_TYPE = "pyttsx3"
    print("✅ Using pyttsx3 (local TTS - no rate limits!)")
except ImportError:
    print("⚠️ pyttsx3 not found, trying gTTS...")
    try:
        from gtts import gTTS

        TTS_AVAILABLE = True
        TTS_TYPE = "gtts"
        print("✅ Using gTTS (may have rate limits)")
    except ImportError:
        print("❌ No TTS available. Install: pip install pyttsx3")

# Audio processing
try:
    import soundfile as sf
    import librosa

    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("❌ Audio libs not available. Install: pip install soundfile librosa")

# Whisper ASR
try:
    import whisper

    WHISPER_AVAILABLE = True and FFMPEG_AVAILABLE
except ImportError:
    WHISPER_AVAILABLE = False
    print("❌ Whisper not available. Install: pip install openai-whisper")


class SPECTRALValidatorSimple:
    """Simplified SPECTRAL validator"""

    def validate_audio_pair(self, audio1_path: Path, audio2_path: Path) -> Dict:
        if not AUDIO_AVAILABLE:
            return {'combined_score': 0.35, 'classification': 'moderate'}

        try:
            y1, sr1 = librosa.load(audio1_path, sr=16000)
            y2, sr2 = librosa.load(audio2_path, sr=16000)

            duration_diff = abs(len(y1) - len(y2)) / max(len(y1), len(y2))

            energy1 = np.sqrt(np.mean(y1 ** 2))
            energy2 = np.sqrt(np.mean(y2 ** 2))
            energy_diff = abs(energy1 - energy2) / max(energy1, energy2) if max(energy1, energy2) > 0 else 0

            combined = (duration_diff + energy_diff) / 2

            return {
                'combined_score': combined,
                'classification': 'strong' if combined > 0.15 else 'moderate',
                'duration_diff': duration_diff,
                'energy_diff': energy_diff
            }
        except:
            return {'combined_score': 0.0, 'classification': 'low'}


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate"""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))

    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(d[i - 1][j - 1] + 1, d[i][j - 1] + 1, d[i - 1][j] + 1)

    return min(d[len(ref_words)][len(hyp_words)] / len(ref_words), 1.0)


class SpanishACCENTExperiment:
    """Complete pipeline using LOCAL TTS"""

    def __init__(self, output_dir: str = None):
        self.language = "Spanish"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if output_dir is None:
            output_dir = f"spanish_results_{self.timestamp}"

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.audio_dir = self.output_dir / "audio"
        self.audio_dir.mkdir(exist_ok=True)
        (self.audio_dir / "loanwords").mkdir(exist_ok=True)
        (self.audio_dir / "native").mkdir(exist_ok=True)

        self.whisper_model = None
        self.validator = SPECTRALValidatorSimple()

        # Initialize pyttsx3 engine if available
        self.tts_engine = None
        if TTS_TYPE == "pyttsx3":
            self.tts_engine = pyttsx3.init()
            # Set Spanish voice if available
            voices = self.tts_engine.getProperty('voices')
            spanish_voice = None
            for voice in voices:
                if 'spanish' in voice.name.lower() or 'es' in voice.id.lower():
                    spanish_voice = voice.id
                    break
            if spanish_voice:
                self.tts_engine.setProperty('voice', spanish_voice)
                print(f"✅ Using Spanish voice: {spanish_voice}")
            # Set rate and volume
            self.tts_engine.setProperty('rate', 175)  # Speed
            self.tts_engine.setProperty('volume', 1.0)  # Volume

        self._check_requirements()

    def _check_requirements(self):
        print("\n📋 Component Status:")
        print(f"  TTS: {'✅ ' + TTS_TYPE if TTS_AVAILABLE else '❌'}")
        print(f"  Audio Processing: {'✅' if AUDIO_AVAILABLE else '❌'}")
        print(f"  Whisper ASR: {'✅' if WHISPER_AVAILABLE else '❌'}")
        print(f"  FFmpeg: {'✅' if FFMPEG_AVAILABLE else '❌'}")

    def load_conloan_sentences(self, n_pairs: Optional[int] = None) -> List[Dict]:
        """Load Spanish sentences from ConLoan JSON"""
        json_file = Path('data/Spanish.json')
        if not json_file.exists():
            print(f"❌ ConLoan data file not found: {json_file}")
            return []

        print(f"📂 Loading ConLoan sentences from {json_file}")

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        sentence_pairs = []
        data_to_process = data[:n_pairs] if n_pairs else data

        for entry in data_to_process:
            if 'source_annotated_loanwords' in entry and 'source_annotated_loanwords_replaced' in entry:
                loanword_sentence = entry['source_annotated_loanwords']
                native_sentence = entry['source_annotated_loanwords_replaced']
                corresponding = entry.get('corresponding_words', {})

                if loanword_sentence != native_sentence:
                    # Remove annotation tags for TTS
                    clean_loanword = loanword_sentence
                    clean_native = native_sentence
                    for tag in ['<L1>', '</L1>', '<L2>', '</L2>', '<N1>', '</N1>', '<N2>', '</N2>']:
                        clean_loanword = clean_loanword.replace(tag, '')
                        clean_native = clean_native.replace(tag, '')

                    sentence_pairs.append({
                        'sentence_loanword': clean_loanword,
                        'sentence_native': clean_native,
                        'corresponding_words': corresponding,
                        'pair_id': f"es_{len(sentence_pairs):03d}"
                    })

        print(f"✅ Loaded {len(sentence_pairs)} sentence pairs")
        return sentence_pairs

    def generate_tts_audio_local(self, sentence: str, output_path: Path) -> bool:
        """Generate TTS using LOCAL pyttsx3 - NO RATE LIMITS!"""
        if output_path.exists() and output_path.stat().st_size > 0:
            return True

        if not TTS_AVAILABLE:
            return False

        try:
            if TTS_TYPE == "pyttsx3":
                # Use local TTS - no rate limits!
                self.tts_engine.save_to_file(sentence, str(output_path))
                self.tts_engine.runAndWait()
                return True
            elif TTS_TYPE == "gtts":
                # Fallback to gTTS (has rate limits)
                import time
                time.sleep(0.5)  # Rate limiting
                tts = gTTS(text=sentence, lang='es', slow=False)
                tts.save(str(output_path))
                return True
        except Exception as e:
            print(f"❌ TTS failed: {e}")
            return False

    def transcribe_with_whisper(self, audio_path: Path) -> str:
        """Transcribe audio using Whisper"""
        if not WHISPER_AVAILABLE or not audio_path.exists():
            return ""

        if self.whisper_model is None:
            print("Loading Whisper model...")
            try:
                self.whisper_model = whisper.load_model("base")
                print("✅ Whisper model loaded")
            except Exception as e:
                print(f"❌ Failed to load Whisper: {e}")
                return ""

        try:
            result = self.whisper_model.transcribe(
                str(audio_path),
                language='es',
                temperature=0.0,
                beam_size=5,
                fp16=False
            )
            return result['text'].strip()
        except Exception as e:
            return ""

    def run_complete_pipeline(self, n_pairs: Optional[int] = None) -> Dict:
        """Run the complete Spanish ASR experiment"""
        print("""
╔══════════════════════════════════════════════════════════════╗
║         ACCENT Spanish ASR Loanword Preference Test         ║
║            Using LOCAL TTS - No Rate Limits! 🚀             ║
╚══════════════════════════════════════════════════════════════╝
        """)

        results = {
            'timestamp': self.timestamp,
            'language': 'Spanish',
            'pairs_tested': 0,
            'validation_results': [],
            'asr_results': [],
            'summary': {}
        }

        # Step 1: Load ConLoan sentences
        print("\n📂 Step 1: Loading ConLoan sentence pairs...")
        sentence_pairs = self.load_conloan_sentences(n_pairs)

        if not sentence_pairs:
            print("❌ No sentence pairs loaded.")
            return results

        print(f"✅ Loaded {len(sentence_pairs)} sentence pairs")

        # Step 2: Generate TTS audio - NO RATE LIMITS with pyttsx3!
        print(f"\n🔊 Step 2: Generating TTS audio for {len(sentence_pairs)} pairs...")
        if TTS_TYPE == "pyttsx3":
            print("   Using LOCAL TTS - No rate limits! This will be fast! 🚀")

        audio_pairs = []
        failed_count = 0

        for i, sent_pair in enumerate(sentence_pairs):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(sentence_pairs)} pairs...")

            lw_path = self.audio_dir / "loanwords" / f"{sent_pair['pair_id']}_lw.wav"
            nat_path = self.audio_dir / "native" / f"{sent_pair['pair_id']}_nat.wav"

            lw_success = self.generate_tts_audio_local(sent_pair['sentence_loanword'], lw_path)
            nat_success = self.generate_tts_audio_local(sent_pair['sentence_native'], nat_path)

            if lw_success and nat_success:
                audio_pairs.append((lw_path, nat_path, sent_pair))
            else:
                failed_count += 1

        print(f"\n✅ Generated {len(audio_pairs)} audio pairs")
        if failed_count > 0:
            print(f"⚠️ Failed: {failed_count} pairs")

        # Step 3: Validation
        print("\n🔬 Step 3: Running SPECTRAL validation...")
        validation_scores = []
        for lw_path, nat_path, sent_pair in audio_pairs[:5]:
            if lw_path.exists() and nat_path.exists():
                score = self.validator.validate_audio_pair(lw_path, nat_path)
                validation_scores.append(score)

        # Step 4: ASR transcription
        print(f"\n🎯 Step 4: Transcribing {len(audio_pairs)} audio pairs...")

        loanword_wers = []
        native_wers = []
        pair_results = []

        for i, (lw_path, nat_path, sent_pair) in enumerate(audio_pairs):
            if i % 5 == 0:
                print(f"  Progress: {i}/{len(audio_pairs)} pairs...")

            lw_transcript = self.transcribe_with_whisper(lw_path)
            nat_transcript = self.transcribe_with_whisper(nat_path)

            lw_wer = calculate_wer(sent_pair['sentence_loanword'], lw_transcript)
            nat_wer = calculate_wer(sent_pair['sentence_native'], nat_transcript)

            loanword_wers.append(lw_wer)
            native_wers.append(nat_wer)

            first_pair = list(sent_pair['corresponding_words'].values())[0] if sent_pair['corresponding_words'] else [
                '', '']

            pair_results.append({
                'pair_id': sent_pair['pair_id'],
                'loanword_example': first_pair[0] if isinstance(first_pair, list) else str(first_pair),
                'native_example': first_pair[1] if isinstance(first_pair, list) and len(first_pair) > 1 else '',
                'loanword_wer': lw_wer,
                'native_wer': nat_wer,
                'wer_difference': lw_wer - nat_wer,
            })

        # Step 5: Analysis
        print("\n📊 Step 5: Analyzing results...")

        mean_lw_wer = np.mean(loanword_wers) if loanword_wers else 0
        mean_nat_wer = np.mean(native_wers) if native_wers else 0
        wer_difference = mean_lw_wer - mean_nat_wer

        pooled_std = np.sqrt((np.std(loanword_wers) ** 2 + np.std(native_wers) ** 2) / 2) if loanword_wers else 0
        effect_size = abs(wer_difference) / pooled_std if pooled_std > 0 else 0

        favor_loanword = sum(1 for r in pair_results if r['wer_difference'] < 0)
        favor_native = sum(1 for r in pair_results if r['wer_difference'] > 0)

        results['summary'] = {
            'n_pairs': len(pair_results),
            'mean_loanword_wer': mean_lw_wer,
            'mean_native_wer': mean_nat_wer,
            'wer_difference': wer_difference,
            'preference': 'loanword' if wer_difference < 0 else 'native',
            'effect_size': effect_size,
            'pairs_favoring_loanword': favor_loanword,
            'pairs_favoring_native': favor_native,
        }

        results['asr_results'] = pair_results
        results['pairs_tested'] = len(pair_results)

        # Print results
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(f"\n📊 Overall Statistics:")
        print(f"  Pairs tested: {len(pair_results)}")
        print(f"  Mean Loanword WER: {mean_lw_wer:.3f}")
        print(f"  Mean Native WER: {mean_nat_wer:.3f}")
        print(f"  WER Difference: {wer_difference:.3f}")
        print(f"  Effect Size: {effect_size:.2f}")
        print(f"  Preference: {results['summary']['preference'].upper()}")

        print(f"\n📊 Direction:")
        print(f"  Favoring loanwords: {favor_loanword}/{len(pair_results)}")
        print(f"  Favoring native: {favor_native}/{len(pair_results)}")

        # Save results
        json_path = self.output_dir / f"results_{self.timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Results saved to: {json_path}")

        if results['asr_results']:
            df = pd.DataFrame(results['asr_results'])
            csv_path = self.output_dir / f"pairs_{self.timestamp}.csv"
            df.to_csv(csv_path, index=False)
            print(f"💾 Details saved to: {csv_path}")

        return results


def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║     ACCENT: Spanish ASR Loanword Preference Testing         ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # First, install pyttsx3 if not available
    if TTS_TYPE != "pyttsx3":
        print("\n⚠️ pyttsx3 not installed. Install it for unlimited TTS:")
        print("   pip install pyttsx3\n")

    n_pairs_input = input("\nNumber of pairs (Enter for ALL): ").strip()

    if n_pairs_input == "":
        n_pairs = None
        print("Loading ALL pairs...")
    elif n_pairs_input.isdigit():
        n_pairs = int(n_pairs_input)
    else:
        n_pairs = 30

    experiment = SpanishACCENTExperiment()
    results = experiment.run_complete_pipeline(n_pairs=n_pairs)

    if results.get('summary'):
        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETE!")
        print("=" * 60)


if __name__ == "__main__":
    main()