#!/usr/bin/env python3
"""
ACCENT ASR Analysis Only - Uses Existing Audio Files
Skips TTS generation, uses audio from previous run
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

warnings.filterwarnings('ignore')

# Audio processing
try:
    import librosa

    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("❌ Audio libs not available. Install: pip install librosa soundfile")

# Whisper ASR
try:
    import whisper

    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("❌ Whisper not available. Install: pip install openai-whisper")


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


class SpanishASRAnalysis:
    """Analyze existing audio files with Whisper"""

    def __init__(self, existing_audio_dir: str):
        """
        Initialize with existing audio directory

        Args:
            existing_audio_dir: Path to directory with audio/loanwords and audio/native folders
        """
        self.audio_dir = Path(existing_audio_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"asr_analysis_{self.timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.whisper_model = None

        # Check audio directory exists
        if not self.audio_dir.exists():
            print(f"❌ Audio directory not found: {self.audio_dir}")
            sys.exit(1)

        lw_dir = self.audio_dir / "loanwords"
        nat_dir = self.audio_dir / "native"

        if not lw_dir.exists() or not nat_dir.exists():
            print(f"❌ Expected subdirectories 'loanwords' and 'native' in {self.audio_dir}")
            sys.exit(1)

        print(f"✅ Found audio directory: {self.audio_dir}")
        print(f"   Loanword files: {len(list(lw_dir.glob('*.wav')))}")
        print(f"   Native files: {len(list(nat_dir.glob('*.wav')))}")

    def load_conloan_sentences(self) -> Dict[str, Dict]:
        """Load ConLoan sentences and map to audio files"""
        json_file = Path('data/Spanish.json')

        if not json_file.exists():
            print(f"❌ ConLoan data not found: {json_file}")
            return {}

        print(f"📂 Loading ConLoan sentences from {json_file}")

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Create mapping from pair_id to sentences
        sentence_map = {}
        pair_id = 0

        for entry in data:
            if 'source_annotated_loanwords' in entry and 'source_annotated_loanwords_replaced' in entry:
                loanword_sentence = entry['source_annotated_loanwords']
                native_sentence = entry['source_annotated_loanwords_replaced']

                # Clean sentences (remove tags)
                for tag in ['<L1>', '</L1>', '<L2>', '</L2>', '<N1>', '</N1>', '<N2>', '</N2>']:
                    loanword_sentence = loanword_sentence.replace(tag, '')
                    native_sentence = native_sentence.replace(tag, '')

                if loanword_sentence != native_sentence:
                    sentence_map[f"es_{pair_id:03d}"] = {
                        'sentence_loanword': loanword_sentence,
                        'sentence_native': native_sentence,
                        'corresponding_words': entry.get('corresponding_words', {})
                    }
                    pair_id += 1

        print(f"✅ Loaded {len(sentence_map)} sentence pairs")
        return sentence_map

    def find_audio_pairs(self) -> List[Tuple[Path, Path, str]]:
        """Find matching audio file pairs"""
        lw_dir = self.audio_dir / "loanwords"
        nat_dir = self.audio_dir / "native"

        pairs = []

        # Find all loanword files
        for lw_file in sorted(lw_dir.glob("*.wav")):
            # Extract pair ID (e.g., "es_000" from "es_000_lw.wav")
            pair_id = lw_file.stem.replace("_lw", "")

            # Look for matching native file
            nat_file = nat_dir / f"{pair_id}_nat.wav"

            if nat_file.exists():
                pairs.append((lw_file, nat_file, pair_id))

        print(f"✅ Found {len(pairs)} complete audio pairs")
        return pairs

    def transcribe_with_whisper(self, audio_path: Path) -> str:
        """Transcribe audio using Whisper"""
        if not WHISPER_AVAILABLE:
            print("❌ Whisper not available")
            return ""

        if self.whisper_model is None:
            print("\nLoading Whisper model (base)...")
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
            print(f"❌ Transcription error: {e}")
            return ""

    def run_analysis(self) -> Dict:
        """Run ASR analysis on existing audio files"""

        print("""
╔══════════════════════════════════════════════════════════════╗
║         ACCENT ASR Analysis - Using Existing Audio          ║
╚══════════════════════════════════════════════════════════════╝
        """)

        # Load sentence data
        print("\n📂 Step 1: Loading ConLoan sentences...")
        sentence_map = self.load_conloan_sentences()

        # Find audio pairs
        print("\n🔍 Step 2: Finding audio pairs...")
        audio_pairs = self.find_audio_pairs()

        if not audio_pairs:
            print("❌ No audio pairs found")
            return {}

        # Run transcription and analysis
        print(f"\n🎯 Step 3: Transcribing {len(audio_pairs)} audio pairs with Whisper...")

        results = {
            'timestamp': self.timestamp,
            'language': 'Spanish',
            'audio_directory': str(self.audio_dir),
            'pairs_tested': 0,
            'asr_results': [],
            'summary': {}
        }

        loanword_wers = []
        native_wers = []
        pair_results = []
        skipped = 0

        for i, (lw_path, nat_path, pair_id) in enumerate(audio_pairs):
            print(f"  Processing: {i + 1}/{len(audio_pairs)} ({pair_id})...", end='\r')

            # Get original sentences
            if pair_id not in sentence_map:
                print(f"\n⚠️ No sentence data for {pair_id}, skipping...")
                skipped += 1
                continue

            sent_data = sentence_map[pair_id]

            # Transcribe
            lw_transcript = self.transcribe_with_whisper(lw_path)
            nat_transcript = self.transcribe_with_whisper(nat_path)

            # Calculate WER
            lw_wer = calculate_wer(sent_data['sentence_loanword'], lw_transcript)
            nat_wer = calculate_wer(sent_data['sentence_native'], nat_transcript)

            loanword_wers.append(lw_wer)
            native_wers.append(nat_wer)

            # Get example words
            words = sent_data.get('corresponding_words', {})
            first_pair = list(words.values())[0] if words else ['', '']

            pair_results.append({
                'pair_id': pair_id,
                'loanword_example': first_pair[0] if isinstance(first_pair, list) else str(first_pair),
                'native_example': first_pair[1] if isinstance(first_pair, list) and len(first_pair) > 1 else '',
                'loanword_wer': lw_wer,
                'native_wer': nat_wer,
                'wer_difference': lw_wer - nat_wer,
                'loanword_transcript': lw_transcript[:200],
                'native_transcript': nat_transcript[:200]
            })

        print(f"\n✅ Processed {len(pair_results)} pairs")
        if skipped > 0:
            print(f"⚠️ Skipped {skipped} pairs without sentence data")

        # Analysis
        print("\n📊 Step 4: Analyzing results...")

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
            'std_loanword_wer': np.std(loanword_wers),
            'std_native_wer': np.std(native_wers),
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
        print(f"  Mean Loanword WER: {mean_lw_wer:.4f} (SD: {np.std(loanword_wers):.4f})")
        print(f"  Mean Native WER: {mean_nat_wer:.4f} (SD: {np.std(native_wers):.4f})")
        print(f"  WER Difference: {wer_difference:.4f}")
        print(f"  Effect Size (Cohen's d): {effect_size:.3f}")
        print(f"  ASR Preference: {results['summary']['preference'].upper()}")

        print(f"\n📊 Direction of Effect:")
        print(
            f"  Pairs favoring loanwords: {favor_loanword}/{len(pair_results)} ({favor_loanword * 100 / len(pair_results):.1f}%)")
        print(
            f"  Pairs favoring native: {favor_native}/{len(pair_results)} ({favor_native * 100 / len(pair_results):.1f}%)")

        if abs(wer_difference) < 0.01:
            print("\n📊 Finding: No meaningful preference detected")
        elif wer_difference < 0:
            print("\n✅ Finding: ASR shows preference for LOANWORDS")
            print(f"   Loanword sentences are recognized {abs(wer_difference) * 100:.1f}% better")
        else:
            print("\n✅ Finding: ASR shows preference for NATIVE terms")
            print(f"   Native sentences are recognized {abs(wer_difference) * 100:.1f}% better")

        # Top biased pairs
        sorted_pairs = sorted(pair_results, key=lambda x: abs(x['wer_difference']), reverse=True)
        print(f"\n📊 Top 5 Most Biased Pairs:")
        for pair in sorted_pairs[:5]:
            direction = "→ loanword" if pair['wer_difference'] < 0 else "→ native"
            print(f"  {pair['pair_id']}: Δ={abs(pair['wer_difference']):.3f} {direction}")
            if pair['loanword_example']:
                print(f"    Words: '{pair['loanword_example']}' vs '{pair['native_example']}'")

        # Save results
        json_path = self.output_dir / f"asr_results_{self.timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Results saved to: {json_path}")

        # Save CSV
        if pair_results:
            df = pd.DataFrame(pair_results)
            csv_path = self.output_dir / f"asr_pairs_{self.timestamp}.csv"
            df.to_csv(csv_path, index=False)
            print(f"💾 Detailed results: {csv_path}")

        return results


def main():
    """Main entry point"""

    print("""
╔══════════════════════════════════════════════════════════════╗
║        ACCENT ASR Analysis - Existing Audio Only            ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # Get audio directory
    audio_dir = input("\nPath to audio directory (e.g., spanish_results_20251030_012640/audio): ").strip()

    if not audio_dir:
        # Try to find most recent results directory
        results_dirs = sorted(Path('.').glob('spanish_results_*/audio'))
        if results_dirs:
            audio_dir = str(results_dirs[-1])
            print(f"Using most recent: {audio_dir}")
        else:
            print("❌ No audio directory specified")
            return

    # Run analysis
    analyzer = SpanishASRAnalysis(audio_dir)
    results = analyzer.run_analysis()

    if results.get('summary'):
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE!")
        print("=" * 60)
        print(f"Results saved in: {analyzer.output_dir}")


if __name__ == "__main__":
    main()