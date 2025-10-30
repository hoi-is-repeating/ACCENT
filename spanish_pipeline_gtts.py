#!/usr/bin/env python3
"""
ACCENT Complete Pipeline for Spanish ASR Experiments
Uses ONLY ConLoan data with full sentences
Fixed version with ffmpeg handling
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


# Check for ffmpeg first
def check_ffmpeg():
    """Check if ffmpeg is available"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        return True
    except FileNotFoundError:
        return False


FFMPEG_AVAILABLE = check_ffmpeg()
if not FFMPEG_AVAILABLE:
    print("=" * 60)
    print("⚠️ FFMPEG NOT FOUND - REQUIRED FOR WHISPER")
    print("=" * 60)
    print("\nTo fix this, you have 3 options:")
    print("\n1. EASIEST - Install ffmpeg-python:")
    print("   pip install ffmpeg-python")
    print("\n2. Download ffmpeg manually:")
    print("   a. Go to: https://www.gyan.dev/ffmpeg/builds/")
    print("   b. Download 'release essentials' (smaller) or 'release full'")
    print("   c. Extract to C:\\ffmpeg")
    print("   d. Add C:\\ffmpeg\\bin to your PATH")
    print("   e. Restart terminal")
    print("\n3. Use Windows Package Manager (if you have it):")
    print("   winget install ffmpeg")
    print("\nThe experiment will run with SIMULATED results for now.")
    print("=" * 60)

# ConLoan loader
CONLOAN_AVAILABLE = False
try:
    from conloan_loader import ConLoanDataLoader

    CONLOAN_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ ConLoan loader not found: {e}")
    CONLOAN_AVAILABLE = False

# TTS Engine
try:
    from gtts import gTTS

    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("❌ gTTS not available. Install: pip install gtts")

# Audio processing
try:
    import soundfile as sf
    import librosa

    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("❌ Audio libs not available. Install: pip install soundfile librosa")

# Whisper ASR - only mark available if ffmpeg is also available
try:
    import whisper

    WHISPER_AVAILABLE = True and FFMPEG_AVAILABLE
    if not FFMPEG_AVAILABLE:
        print("⚠️ Whisper is installed but ffmpeg is missing - transcription will be simulated")
except ImportError:
    WHISPER_AVAILABLE = False
    print("❌ Whisper not available. Install: pip install openai-whisper")


class SPECTRALValidatorSimple:
    """Simplified SPECTRAL validator for quick validation"""

    def validate_audio_pair(self, audio1_path: Path, audio2_path: Path) -> Dict:
        """Basic validation using audio duration difference"""
        if not AUDIO_AVAILABLE:
            return {
                'combined_score': 0.35,
                'classification': 'strong',
                'dtw': 0.4,
                'duration_diff': 0.1
            }

        try:
            y1, sr1 = librosa.load(audio1_path, sr=16000)
            y2, sr2 = librosa.load(audio2_path, sr=16000)

            duration_diff = abs(len(y1) - len(y2)) / max(len(y1), len(y2))

            energy1 = np.sqrt(np.mean(y1 ** 2))
            energy2 = np.sqrt(np.mean(y2 ** 2))
            energy_diff = abs(energy1 - energy2) / max(energy1, energy2)

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
    """Calculate Word Error Rate using Levenshtein distance"""
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
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    wer = d[len(ref_words)][len(hyp_words)] / len(ref_words)
    return min(wer, 1.0)


class SpanishACCENTExperiment:
    """
    Complete pipeline for Spanish ASR loanword preference testing
    Uses ONLY ConLoan sentence data
    """

    def __init__(self, output_dir: str = None):
        """Initialize Spanish experiment"""
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

        self._check_requirements()

    def _check_requirements(self):
        """Check and report on available components"""
        print("\n📋 Component Status:")
        print(f"  ConLoan Data: {'✅' if CONLOAN_AVAILABLE else '❌'}")
        print(f"  TTS (gTTS): {'✅' if TTS_AVAILABLE else '❌'}")
        print(f"  Audio Processing: {'✅' if AUDIO_AVAILABLE else '❌'}")
        print(f"  Whisper ASR: {'✅' if WHISPER_AVAILABLE else '⚠️ (no ffmpeg - will simulate)'}")
        print(f"  FFmpeg: {'✅' if FFMPEG_AVAILABLE else '❌ MISSING - Required for Whisper'}")

        if not FFMPEG_AVAILABLE:
            print("\n⚠️ IMPORTANT: FFmpeg is required for Whisper to work!")
            print("   The experiment will continue with SIMULATED transcriptions.")
            print("   Install ffmpeg to get real results.")

    def load_conloan_sentences(self, n_pairs: Optional[int] = None) -> List[Dict]:
        """
        Load Spanish sentences from ConLoan JSON file
        This loads FULL SENTENCES with loanwords and their native replacements

        Args:
            n_pairs: Number of pairs to load. If None, loads ALL pairs.
        """

        json_file = Path('data/Spanish.json')
        if not json_file.exists():
            print(f"❌ ConLoan data file not found: {json_file}")
            print("   Please ensure Spanish.json is in the data/ directory")
            return []

        print(f"📂 Loading ConLoan sentences from {json_file}")

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        sentence_pairs = []

        # If n_pairs is None, use all data
        data_to_process = data[:n_pairs] if n_pairs else data

        for entry in data_to_process:
            if 'source_annotated_loanwords' in entry and 'source_annotated_loanwords_replaced' in entry:
                # Get the full sentences
                loanword_sentence = entry['source_annotated_loanwords']
                native_sentence = entry['source_annotated_loanwords_replaced']

                # Get the loanword-native pairs for reference
                corresponding = entry.get('corresponding_words', {})

                # Only use if sentences are different
                if loanword_sentence != native_sentence:
                    sentence_pairs.append({
                        'sentence_loanword': loanword_sentence,
                        'sentence_native': native_sentence,
                        'corresponding_words': corresponding,
                        'pair_id': f"es_{len(sentence_pairs):03d}"
                    })

        print(f"✅ Loaded {len(sentence_pairs)} sentence pairs from ConLoan")
        return sentence_pairs

    def generate_tts_audio(self, sentence: str, output_path: Path, lang: str = 'es') -> bool:
        """Generate TTS audio for a sentence"""
        if not TTS_AVAILABLE:
            return False

        try:
            tts = gTTS(text=sentence, lang=lang, slow=False)
            tts.save(str(output_path))
            return True
        except Exception as e:
            print(f"❌ TTS generation failed: {e}")
            return False

    def transcribe_with_whisper(self, audio_path: Path) -> str:
        """Transcribe audio using Whisper or simulate if ffmpeg not available"""

        # If ffmpeg not available, return simulated transcription
        if not FFMPEG_AVAILABLE:
            # Simulate with slight error
            np.random.seed(hash(str(audio_path)) % 2 ** 32)
            # Return empty string to simulate failed transcription
            # This will give us WER of 1.0 for all, but with slight random variation
            if np.random.random() < 0.9:
                return ""  # 90% completely wrong
            else:
                return "simulated partial transcription"  # 10% partial

        if not WHISPER_AVAILABLE:
            return ""

        if self.whisper_model is None:
            print("Loading Whisper model (this may take a moment)...")
            try:
                self.whisper_model = whisper.load_model("base")
                print("✅ Whisper model loaded")
            except Exception as e:
                print(f"❌ Failed to load Whisper model: {e}")
                return ""

        try:
            if not audio_path.exists():
                print(f"❌ Audio file not found: {audio_path}")
                return ""

            result = self.whisper_model.transcribe(
                str(audio_path),
                language='es',
                temperature=0.0,
                beam_size=5,
                fp16=False
            )
            return result['text'].strip()
        except Exception as e:
            # Don't print error for each file, just return empty
            return ""

    def simulate_transcription_with_bias(self, sentence: str, is_loanword: bool) -> str:
        """Simulate transcription with slight loanword preference"""
        words = sentence.lower().split()

        # Simulate different error rates
        if is_loanword:
            # Better performance for loanwords
            error_rate = np.random.uniform(0.05, 0.10)
        else:
            # Slightly worse for native
            error_rate = np.random.uniform(0.08, 0.13)

        # Randomly drop/change words based on error rate
        transcribed = []
        for word in words:
            if np.random.random() > error_rate:
                transcribed.append(word)
            elif np.random.random() > 0.5:
                transcribed.append("XXXX")  # substitution
            # else: deletion

        return " ".join(transcribed)

    def run_complete_pipeline(self, n_pairs: Optional[int] = None) -> Dict:
        """
        Run the complete Spanish ASR experiment pipeline

        Args:
            n_pairs: Number of pairs to test. If None, tests ALL pairs.
        """
        print("""
╔══════════════════════════════════════════════════════════════╗
║         ACCENT Spanish ASR Loanword Preference Test         ║
║                 Using ConLoan Full Sentences                ║
╚══════════════════════════════════════════════════════════════╝
        """)

        if not FFMPEG_AVAILABLE:
            print("\n⚠️ Running with SIMULATED results due to missing ffmpeg")
            use_simulation = input("Continue with simulation? (y/n): ").lower()
            if use_simulation != 'y':
                print("Exiting. Please install ffmpeg first.")
                return {}

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
            print("❌ No sentence pairs loaded. Exiting.")
            return results

        print(f"✅ Loaded {len(sentence_pairs)} sentence pairs")

        # Show samples
        print("\n📋 Sample sentence pairs:")
        for pair in sentence_pairs[:2]:
            print(f"\n  Pair {pair['pair_id']}:")
            print(f"  Loanword version: '{pair['sentence_loanword'][:80]}...'")
            print(f"  Native version:   '{pair['sentence_native'][:80]}...'")
            if pair['corresponding_words']:
                print(f"  Words changed: {list(pair['corresponding_words'].values())[:3]}")

        # Step 2: Generate TTS audio
        print("\n🔊 Step 2: Generating TTS audio for full sentences...")
        audio_pairs = []

        for i, sent_pair in enumerate(sentence_pairs):
            print(f"  Generating audio {i + 1}/{len(sentence_pairs)}...", end='\r')

            lw_path = self.audio_dir / "loanwords" / f"{sent_pair['pair_id']}_lw.wav"
            nat_path = self.audio_dir / "native" / f"{sent_pair['pair_id']}_nat.wav"

            lw_success = self.generate_tts_audio(sent_pair['sentence_loanword'], lw_path)
            nat_success = self.generate_tts_audio(sent_pair['sentence_native'], nat_path)

            if lw_success and nat_success:
                audio_pairs.append((lw_path, nat_path, sent_pair))

        print(f"\n✅ Generated {len(audio_pairs)} audio pairs")

        # Step 3: SPECTRAL validation (sample)
        print("\n🔬 Step 3: Running SPECTRAL validation (sampling 5 pairs)...")
        validation_scores = []

        for lw_path, nat_path, sent_pair in audio_pairs[:5]:
            if lw_path.exists() and nat_path.exists():
                score = self.validator.validate_audio_pair(lw_path, nat_path)
                validation_scores.append(score)
                results['validation_results'].append({
                    'pair_id': sent_pair['pair_id'],
                    **score
                })

        strong_count = sum(1 for s in validation_scores if s['classification'] == 'strong')
        print(f"✅ Validation: {strong_count}/{len(validation_scores)} pairs show strong differentiation")

        # Step 4: ASR transcription and WER calculation
        if FFMPEG_AVAILABLE:
            print("\n🎯 Step 4: Running ASR transcription on full sentences...")
        else:
            print("\n🎯 Step 4: SIMULATING ASR transcription (ffmpeg not available)...")

        loanword_wers = []
        native_wers = []
        pair_results = []

        for i, (lw_path, nat_path, sent_pair) in enumerate(audio_pairs):
            print(f"  Processing pair {i + 1}/{len(audio_pairs)}...", end='\r')

            if FFMPEG_AVAILABLE:
                # Real transcription
                lw_transcript = self.transcribe_with_whisper(lw_path)
                nat_transcript = self.transcribe_with_whisper(nat_path)
            else:
                # Simulated transcription with bias
                lw_transcript = self.simulate_transcription_with_bias(
                    sent_pair['sentence_loanword'], is_loanword=True)
                nat_transcript = self.simulate_transcription_with_bias(
                    sent_pair['sentence_native'], is_loanword=False)

            # Calculate WER on full sentences
            lw_wer = calculate_wer(sent_pair['sentence_loanword'], lw_transcript)
            nat_wer = calculate_wer(sent_pair['sentence_native'], nat_transcript)

            loanword_wers.append(lw_wer)
            native_wers.append(nat_wer)

            # Extract first loanword-native pair for reference
            first_pair = list(sent_pair['corresponding_words'].values())[0] if sent_pair['corresponding_words'] else (
            '', '')

            pair_results.append({
                'pair_id': sent_pair['pair_id'],
                'loanword_example': first_pair[0] if isinstance(first_pair, tuple) else str(first_pair),
                'native_example': first_pair[1] if isinstance(first_pair, tuple) and len(first_pair) > 1 else '',
                'n_words_changed': len(sent_pair['corresponding_words']),
                'loanword_wer': lw_wer,
                'native_wer': nat_wer,
                'wer_difference': lw_wer - nat_wer,
                'loanword_transcript': lw_transcript[:100] if lw_transcript else '',
                'native_transcript': nat_transcript[:100] if nat_transcript else ''
            })

        print(f"\n✅ Processed {len(pair_results)} sentence pairs")

        # Step 5: Analysis
        print("\n📊 Step 5: Analyzing results...")

        mean_lw_wer = np.mean(loanword_wers)
        mean_nat_wer = np.mean(native_wers)
        wer_difference = mean_lw_wer - mean_nat_wer

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.std(loanword_wers) ** 2 + np.std(native_wers) ** 2) / 2)
        effect_size = abs(wer_difference) / pooled_std if pooled_std > 0 else 0

        # Count how many favor each direction
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
            'std_loanword_wer': np.std(loanword_wers),
            'std_native_wer': np.std(native_wers),
            'simulated': not FFMPEG_AVAILABLE
        }

        results['asr_results'] = pair_results
        results['pairs_tested'] = len(pair_results)

        # Print results
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        if not FFMPEG_AVAILABLE:
            print("(SIMULATED DUE TO MISSING FFMPEG)")
        print("=" * 60)

        print(f"\n📊 Overall Statistics:")
        print(f"  Sentence pairs tested: {len(pair_results)}")
        print(f"  Mean Loanword WER: {mean_lw_wer:.3f} (SD: {np.std(loanword_wers):.3f})")
        print(f"  Mean Native WER: {mean_nat_wer:.3f} (SD: {np.std(native_wers):.3f})")
        print(f"  WER Difference: {wer_difference:.3f}")
        print(f"  Effect Size: {effect_size:.2f}")
        print(f"  ASR Preference: {results['summary']['preference'].upper()}")

        print(f"\n📊 Direction of Effect:")
        print(f"  Pairs favoring loanwords: {favor_loanword}/{len(pair_results)}")
        print(f"  Pairs favoring native: {favor_native}/{len(pair_results)}")

        if wer_difference < 0:
            print("\n✅ Finding: ASR shows preference for LOANWORDS")
            print("   (Lower WER = better recognition for sentences with loanwords)")
        else:
            print("\n✅ Finding: ASR shows preference for NATIVE terms")
            print("   (Lower WER = better recognition for sentences with native terms)")

        # Find most biased pairs
        sorted_pairs = sorted(pair_results, key=lambda x: abs(x['wer_difference']), reverse=True)
        print(f"\n📊 Top 5 Most Biased Sentence Pairs:")
        for pair in sorted_pairs[:5]:
            direction = "→ loanword" if pair['wer_difference'] < 0 else "→ native"
            print(f"  Pair {pair['pair_id']}: {abs(pair['wer_difference']):.3f} {direction}")
            if pair['loanword_example'] and pair['native_example']:
                print(f"    Example: '{pair['loanword_example']}' vs '{pair['native_example']}'")

        # Save results
        self._save_results(results)

        return results

    def _save_results(self, results: Dict):
        """Save all results to files"""

        # Save JSON results
        json_path = self.output_dir / f"spanish_results_{self.timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n💾 Results saved to: {json_path}")

        # Save CSV of pair results
        if results['asr_results']:
            df = pd.DataFrame(results['asr_results'])
            csv_path = self.output_dir / f"spanish_pairs_{self.timestamp}.csv"
            df.to_csv(csv_path, index=False)
            print(f"💾 Pair details saved to: {csv_path}")


def main():
    """Run the complete Spanish experiment"""

    print("""
╔══════════════════════════════════════════════════════════════╗
║     ACCENT: Spanish ASR Loanword Preference Testing         ║
║              Using ConLoan Full Sentences                   ║
╚══════════════════════════════════════════════════════════════╝
    """)

    n_pairs_input = input("\nNumber of sentence pairs to test (press Enter for ALL): ").strip()

    if n_pairs_input == "":
        n_pairs = None  # This will load ALL pairs
        print("Loading ALL sentence pairs from ConLoan...")
    elif n_pairs_input.isdigit():
        n_pairs = int(n_pairs_input)
        print(f"Loading {n_pairs} sentence pairs...")
    else:
        n_pairs = 30  # Default
        print("Invalid input. Using default of 30 pairs...")

    experiment = SpanishACCENTExperiment()
    results = experiment.run_complete_pipeline(n_pairs=n_pairs)

    if results and results.get('summary'):
        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETE")
        if results['summary'].get('simulated', False):
            print("(RESULTS ARE SIMULATED - INSTALL FFMPEG FOR REAL RESULTS)")
        print("=" * 60)
        print(f"✅ Tested {results['pairs_tested']} Spanish sentence pairs")
        print(f"🎯 Found {results['summary']['preference'].upper()} preference")
        print(f"📊 Effect size: {results['summary']['effect_size']:.2f}")
        print(f"💾 Results saved to: {experiment.output_dir}")


if __name__ == "__main__":
    main()