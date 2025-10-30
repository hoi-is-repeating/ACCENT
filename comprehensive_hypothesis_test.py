#!/usr/bin/env python3
"""
ACCENT Comprehensive Hypothesis Testing Pipeline
Tests 5 different hypotheses about ASR loanword preference using existing audio
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
from scipy.io import wavfile
from scipy import signal
import librosa
import soundfile as sf

warnings.filterwarnings('ignore')

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


class ComprehensiveASRTester:
    """Test multiple hypotheses about ASR loanword preference"""

    def __init__(self, audio_dir: str):
        """Initialize with existing audio directory"""
        self.audio_dir = Path(audio_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"hypothesis_tests_{self.timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create temp directory for modified audio
        self.temp_audio_dir = self.output_dir / "temp_audio"
        self.temp_audio_dir.mkdir(exist_ok=True)

        self.whisper_model = None

        # Check directories
        if not self.audio_dir.exists():
            raise FileNotFoundError(f"Audio directory not found: {self.audio_dir}")

        self.lw_dir = self.audio_dir / "loanwords"
        self.nat_dir = self.audio_dir / "native"

        print(f"✅ Audio directory: {self.audio_dir}")
        print(f"   Loanword files: {len(list(self.lw_dir.glob('*.wav')))}")
        print(f"   Native files: {len(list(self.nat_dir.glob('*.wav')))}")

    def load_whisper_model(self):
        """Load Whisper model once"""
        if self.whisper_model is None and WHISPER_AVAILABLE:
            print("Loading Whisper model...")
            self.whisper_model = whisper.load_model("base")
            print("✅ Whisper loaded")

    def transcribe(self, audio_path: Path, return_probs: bool = False) -> Dict:
        """Transcribe with optional probability information"""
        if not WHISPER_AVAILABLE or not audio_path.exists():
            return {"text": "", "confidence": 0.0}

        self.load_whisper_model()

        try:
            # Basic transcription
            result = self.whisper_model.transcribe(
                str(audio_path),
                language='es',
                temperature=0.0,
                beam_size=5,
                fp16=False,
                verbose=False
            )

            # Try to get confidence if possible (some Whisper versions)
            output = {"text": result['text'].strip()}

            # If we can get token probabilities (depends on Whisper version)
            if 'segments' in result and result['segments']:
                # Average confidence from segments
                confidences = []
                for segment in result['segments']:
                    if 'avg_logprob' in segment:
                        # Convert log prob to probability
                        conf = np.exp(segment['avg_logprob'])
                        confidences.append(conf)

                if confidences:
                    output['confidence'] = np.mean(confidences)
                else:
                    output['confidence'] = 0.0

            return output

        except Exception as e:
            print(f"Transcription error: {e}")
            return {"text": "", "confidence": 0.0}

    def load_sentence_data(self) -> Dict:
        """Load ConLoan sentences"""
        json_file = Path('data/Spanish.json')
        if not json_file.exists():
            return {}

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        sentence_map = {}
        pair_id = 0

        for entry in data:
            if 'source_annotated_loanwords' in entry and 'source_annotated_loanwords_replaced' in entry:
                lw_sent = entry['source_annotated_loanwords']
                nat_sent = entry['source_annotated_loanwords_replaced']

                # Clean tags
                for tag in ['<L1>', '</L1>', '<L2>', '</L2>', '<N1>', '</N1>', '<N2>', '</N2>']:
                    lw_sent = lw_sent.replace(tag, '')
                    nat_sent = nat_sent.replace(tag, '')

                if lw_sent != nat_sent:
                    sentence_map[f"es_{pair_id:03d}"] = {
                        'sentence_loanword': lw_sent,
                        'sentence_native': nat_sent,
                        'corresponding_words': entry.get('corresponding_words', {})
                    }
                    pair_id += 1

        return sentence_map

    def find_audio_pairs(self) -> List[Tuple[Path, Path, str]]:
        """Find matching audio file pairs"""
        pairs = []
        for lw_file in sorted(self.lw_dir.glob("*.wav")):
            pair_id = lw_file.stem.replace("_lw", "")
            nat_file = self.nat_dir / f"{pair_id}_nat.wav"
            if nat_file.exists():
                pairs.append((lw_file, nat_file, pair_id))
        return pairs

    # ========== HYPOTHESIS 1: ROBUSTNESS ==========

    def add_noise(self, audio_path: Path, snr_db: float) -> Path:
        """Add white noise at specified SNR"""
        y, sr = librosa.load(audio_path, sr=None)

        # Calculate signal power
        signal_power = np.mean(y ** 2)

        # Calculate noise power for desired SNR
        noise_power = signal_power / (10 ** (snr_db / 10))

        # Generate white noise
        noise = np.random.normal(0, np.sqrt(noise_power), len(y))

        # Add noise
        noisy = y + noise

        # Save
        output_path = self.temp_audio_dir / f"noisy_{snr_db}dB_{audio_path.name}"
        sf.write(output_path, noisy, sr)
        return output_path

    def test_robustness_hypothesis(self, audio_pairs: List, sentence_map: Dict) -> Dict:
        """Test if loanwords are more robust to noise"""
        print("\n" + "=" * 60)
        print("HYPOTHESIS 1: ROBUSTNESS TO NOISE")
        print("=" * 60)

        results = []
        snr_levels = [30, 20, 10, 5, 0]  # Clean to very noisy

        for snr in snr_levels:
            print(f"\n📊 Testing at SNR = {snr} dB...")
            lw_wers = []
            nat_wers = []

            for lw_path, nat_path, pair_id in audio_pairs[:20]:  # Test subset
                if pair_id not in sentence_map:
                    continue

                # Add noise
                noisy_lw = self.add_noise(lw_path, snr)
                noisy_nat = self.add_noise(nat_path, snr)

                # Transcribe
                trans_lw = self.transcribe(noisy_lw)['text']
                trans_nat = self.transcribe(noisy_nat)['text']

                # Calculate WER
                ref_lw = sentence_map[pair_id]['sentence_loanword']
                ref_nat = sentence_map[pair_id]['sentence_native']

                wer_lw = calculate_wer(ref_lw, trans_lw)
                wer_nat = calculate_wer(ref_nat, trans_nat)

                lw_wers.append(wer_lw)
                nat_wers.append(wer_nat)

            results.append({
                'snr': snr,
                'mean_loanword_wer': np.mean(lw_wers),
                'mean_native_wer': np.mean(nat_wers),
                'difference': np.mean(lw_wers) - np.mean(nat_wers),
                'n_pairs': len(lw_wers)
            })

            print(f"  Loanword WER: {np.mean(lw_wers):.3f}")
            print(f"  Native WER: {np.mean(nat_wers):.3f}")
            print(f"  Difference: {np.mean(lw_wers) - np.mean(nat_wers):.3f}")

        return {'robustness_results': results}

    # ========== HYPOTHESIS 2: DISAMBIGUATION ==========

    def blend_audio(self, audio1_path: Path, audio2_path: Path, mix_ratio: float = 0.5) -> Path:
        """Blend two audio signals"""
        y1, sr1 = librosa.load(audio1_path, sr=None)
        y2, sr2 = librosa.load(audio2_path, sr=sr1)

        # Align lengths
        min_len = min(len(y1), len(y2))
        y1 = y1[:min_len]
        y2 = y2[:min_len]

        # Blend
        blended = mix_ratio * y1 + (1 - mix_ratio) * y2

        # Save
        output_path = self.temp_audio_dir / f"blend_{mix_ratio}_{audio1_path.stem}_{audio2_path.stem}.wav"
        sf.write(output_path, blended, sr1)
        return output_path

    def test_disambiguation_hypothesis(self, audio_pairs: List, sentence_map: Dict) -> Dict:
        """Test what Whisper chooses when audio is ambiguous"""
        print("\n" + "=" * 60)
        print("HYPOTHESIS 2: DISAMBIGUATION OF AMBIGUOUS AUDIO")
        print("=" * 60)

        results = []
        mix_ratios = [0.3, 0.5, 0.7]  # Different blend ratios

        for ratio in mix_ratios:
            print(f"\n📊 Testing with mix ratio = {ratio}...")
            loanword_chosen = 0
            native_chosen = 0
            neither_chosen = 0

            for lw_path, nat_path, pair_id in audio_pairs[:20]:
                if pair_id not in sentence_map:
                    continue

                # Get the actual words
                words = sentence_map[pair_id].get('corresponding_words', {})
                if not words:
                    continue

                first_pair = list(words.values())[0] if words else None
                if not first_pair or not isinstance(first_pair, list):
                    continue

                lw_word = first_pair[0].lower()
                nat_word = first_pair[1].lower() if len(first_pair) > 1 else ""

                # Create ambiguous blend
                if ratio == 0.5:
                    # True 50/50 blend
                    blended = self.blend_audio(lw_path, nat_path, 0.5)
                else:
                    # Slightly favor one side
                    blended = self.blend_audio(lw_path, nat_path, ratio)

                # Transcribe
                transcription = self.transcribe(blended)['text'].lower()

                # Check what was recognized
                if lw_word in transcription and nat_word not in transcription:
                    loanword_chosen += 1
                elif nat_word in transcription and lw_word not in transcription:
                    native_chosen += 1
                else:
                    neither_chosen += 1

            total = loanword_chosen + native_chosen + neither_chosen
            results.append({
                'mix_ratio': ratio,
                'loanword_chosen': loanword_chosen,
                'native_chosen': native_chosen,
                'neither': neither_chosen,
                'loanword_percentage': (loanword_chosen / total * 100) if total > 0 else 0
            })

            print(f"  Loanword chosen: {loanword_chosen}")
            print(f"  Native chosen: {native_chosen}")
            print(f"  Neither/Both: {neither_chosen}")

        return {'disambiguation_results': results}

    # ========== HYPOTHESIS 3: CORRECTION BIAS ==========

    def distort_audio(self, audio_path: Path, distortion_type: str = "speed") -> Path:
        """Distort audio to create mispronunciations"""
        y, sr = librosa.load(audio_path, sr=None)

        if distortion_type == "speed":
            # Slight speed change (simulates different pronunciation speed)
            y_distorted = librosa.effects.time_stretch(y, rate=1.1)
        elif distortion_type == "pitch":
            # Slight pitch shift
            y_distorted = librosa.effects.pitch_shift(y, sr=sr, n_steps=1)
        else:
            # Add slight reverb/echo
            y_distorted = np.convolve(y, np.array([1, 0.5, 0.25]), mode='same')

        output_path = self.temp_audio_dir / f"distort_{distortion_type}_{audio_path.name}"
        sf.write(output_path, y_distorted, sr)
        return output_path

    def test_correction_hypothesis(self, audio_pairs: List, sentence_map: Dict) -> Dict:
        """Test if Whisper corrects to loanwords more than native"""
        print("\n" + "=" * 60)
        print("HYPOTHESIS 3: CORRECTION BIAS")
        print("=" * 60)

        results = []
        distortion_types = ["speed", "pitch"]

        for dist_type in distortion_types:
            print(f"\n📊 Testing with {dist_type} distortion...")

            lw_to_nat_corrections = 0
            nat_to_lw_corrections = 0
            lw_preserved = 0
            nat_preserved = 0

            for lw_path, nat_path, pair_id in audio_pairs[:20]:
                if pair_id not in sentence_map:
                    continue

                words = sentence_map[pair_id].get('corresponding_words', {})
                if not words:
                    continue

                first_pair = list(words.values())[0] if words else None
                if not first_pair or not isinstance(first_pair, list):
                    continue

                lw_word = first_pair[0].lower()
                nat_word = first_pair[1].lower() if len(first_pair) > 1 else ""

                # Distort both
                distorted_lw = self.distort_audio(lw_path, dist_type)
                distorted_nat = self.distort_audio(nat_path, dist_type)

                # Transcribe
                trans_lw = self.transcribe(distorted_lw)['text'].lower()
                trans_nat = self.transcribe(distorted_nat)['text'].lower()

                # Check corrections
                # From distorted loanword audio
                if lw_word in trans_lw:
                    lw_preserved += 1
                elif nat_word in trans_lw:
                    lw_to_nat_corrections += 1

                # From distorted native audio
                if nat_word in trans_nat:
                    nat_preserved += 1
                elif lw_word in trans_nat:
                    nat_to_lw_corrections += 1

            results.append({
                'distortion_type': dist_type,
                'lw_to_native_corrections': lw_to_nat_corrections,
                'native_to_lw_corrections': nat_to_lw_corrections,
                'lw_preserved': lw_preserved,
                'native_preserved': nat_preserved,
                'correction_bias': nat_to_lw_corrections - lw_to_nat_corrections
            })

            print(f"  Native→Loanword corrections: {nat_to_lw_corrections}")
            print(f"  Loanword→Native corrections: {lw_to_nat_corrections}")
            print(f"  Bias: {nat_to_lw_corrections - lw_to_nat_corrections}")

        return {'correction_results': results}

    # ========== HYPOTHESIS 4: CONFIDENCE ==========

    def test_confidence_hypothesis(self, audio_pairs: List, sentence_map: Dict) -> Dict:
        """Test if Whisper is more confident with loanwords"""
        print("\n" + "=" * 60)
        print("HYPOTHESIS 4: TRANSCRIPTION CONFIDENCE")
        print("=" * 60)

        lw_confidences = []
        nat_confidences = []

        for lw_path, nat_path, pair_id in audio_pairs[:30]:
            # Transcribe with confidence
            lw_result = self.transcribe(lw_path, return_probs=True)
            nat_result = self.transcribe(nat_path, return_probs=True)

            if 'confidence' in lw_result and lw_result['confidence'] > 0:
                lw_confidences.append(lw_result['confidence'])
            if 'confidence' in nat_result and nat_result['confidence'] > 0:
                nat_confidences.append(nat_result['confidence'])

        if lw_confidences and nat_confidences:
            results = {
                'mean_loanword_confidence': np.mean(lw_confidences),
                'mean_native_confidence': np.mean(nat_confidences),
                'confidence_difference': np.mean(lw_confidences) - np.mean(nat_confidences),
                'n_pairs': len(lw_confidences)
            }
            print(f"  Loanword confidence: {results['mean_loanword_confidence']:.4f}")
            print(f"  Native confidence: {results['mean_native_confidence']:.4f}")
            print(f"  Difference: {results['confidence_difference']:.4f}")
        else:
            results = {
                'note': 'Confidence scores not available in this Whisper version'
            }
            print("  ⚠️ Confidence scores not available")

        return {'confidence_results': results}

    # ========== HYPOTHESIS 5: PRIMING ==========

    def test_priming_hypothesis(self, audio_pairs: List, sentence_map: Dict) -> Dict:
        """Test if context primes loanword recognition"""
        print("\n" + "=" * 60)
        print("HYPOTHESIS 5: CONTEXTUAL PRIMING")
        print("=" * 60)

        # Create context audio (would need TTS for this)
        # For now, we'll test with existing audio in sequence

        results = []

        # Test: Does hearing loanwords prime more loanword recognition?
        lw_after_lw = 0
        nat_after_lw = 0
        lw_after_nat = 0
        nat_after_nat = 0

        # Create ambiguous test audio
        for i in range(min(10, len(audio_pairs))):
            lw_path, nat_path, pair_id = audio_pairs[i]

            # Create ambiguous blend for testing
            ambiguous = self.blend_audio(lw_path, nat_path, 0.5)

            # Test after loanword prime (use previous audio as prime)
            if i > 0:
                prime_lw, _, _ = audio_pairs[i - 1]
                # In real implementation, concatenate prime + ambiguous
                # For now, just test the ambiguous after mental "prime"
                trans_after_lw = self.transcribe(ambiguous)['text'].lower()

                # Check if loanword appears
                if pair_id in sentence_map:
                    words = sentence_map[pair_id].get('corresponding_words', {})
                    if words:
                        first_pair = list(words.values())[0]
                        if isinstance(first_pair, list) and len(first_pair) > 1:
                            if first_pair[0].lower() in trans_after_lw:
                                lw_after_lw += 1
                            elif first_pair[1].lower() in trans_after_lw:
                                nat_after_lw += 1

        results = {
            'note': 'Simplified priming test - full implementation needs concatenated audio',
            'ambiguous_transcriptions_tested': 10,
            'tendency': 'Requires more sophisticated setup'
        }

        print(f"  Note: Priming test simplified - needs audio concatenation")

        return {'priming_results': results}

    # ========== MAIN PIPELINE ==========

    def run_all_tests(self) -> Dict:
        """Run all hypothesis tests"""
        print("""
╔══════════════════════════════════════════════════════════════╗
║     COMPREHENSIVE ASR LOANWORD PREFERENCE TESTING           ║
║                    5 Hypotheses                             ║
╚══════════════════════════════════════════════════════════════╝
        """)

        # Load data
        print("\n📂 Loading data...")
        sentence_map = self.load_sentence_data()
        audio_pairs = self.find_audio_pairs()

        if not audio_pairs:
            print("❌ No audio pairs found")
            return {}

        print(f"✅ Found {len(audio_pairs)} audio pairs")
        print(f"✅ Found {len(sentence_map)} sentence mappings")

        all_results = {
            'timestamp': self.timestamp,
            'n_pairs_available': len(audio_pairs),
            'hypotheses': {}
        }

        # Test each hypothesis
        print("\n" + "=" * 60)
        print("STARTING HYPOTHESIS TESTS")
        print("=" * 60)

        # 1. Robustness
        try:
            h1 = self.test_robustness_hypothesis(audio_pairs, sentence_map)
            all_results['hypotheses']['H1_robustness'] = h1
        except Exception as e:
            print(f"❌ H1 failed: {e}")
            all_results['hypotheses']['H1_robustness'] = {'error': str(e)}

        # 2. Disambiguation
        try:
            h2 = self.test_disambiguation_hypothesis(audio_pairs, sentence_map)
            all_results['hypotheses']['H2_disambiguation'] = h2
        except Exception as e:
            print(f"❌ H2 failed: {e}")
            all_results['hypotheses']['H2_disambiguation'] = {'error': str(e)}

        # 3. Correction
        try:
            h3 = self.test_correction_hypothesis(audio_pairs, sentence_map)
            all_results['hypotheses']['H3_correction'] = h3
        except Exception as e:
            print(f"❌ H3 failed: {e}")
            all_results['hypotheses']['H3_correction'] = {'error': str(e)}

        # 4. Confidence
        try:
            h4 = self.test_confidence_hypothesis(audio_pairs, sentence_map)
            all_results['hypotheses']['H4_confidence'] = h4
        except Exception as e:
            print(f"❌ H4 failed: {e}")
            all_results['hypotheses']['H4_confidence'] = {'error': str(e)}

        # 5. Priming
        try:
            h5 = self.test_priming_hypothesis(audio_pairs, sentence_map)
            all_results['hypotheses']['H5_priming'] = h5
        except Exception as e:
            print(f"❌ H5 failed: {e}")
            all_results['hypotheses']['H5_priming'] = {'error': str(e)}

        # Save results
        output_file = self.output_dir / f"all_hypotheses_{self.timestamp}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETE")
        print("=" * 60)
        print(f"\n💾 Results saved to: {output_file}")

        # Print summary
        self.print_summary(all_results)

        return all_results

    def print_summary(self, results: Dict):
        """Print a formatted summary for easy copying"""
        print("\n" + "=" * 60)
        print("SUMMARY FOR ANALYSIS")
        print("=" * 60)

        print("\n```json")
        print(json.dumps(results, indent=2))
        print("```")

        print("\n" + "=" * 60)
        print("Copy the JSON above for analysis")
        print("=" * 60)


def main():
    """Main entry point"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║      Comprehensive ASR Hypothesis Testing Pipeline          ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # Get audio directory
    audio_dir = input("\nPath to audio directory (or Enter for most recent): ").strip()

    if not audio_dir:
        results_dirs = sorted(Path('.').glob('spanish_results_*/audio'))
        if results_dirs:
            audio_dir = str(results_dirs[-1])
            print(f"Using: {audio_dir}")
        else:
            print("❌ No audio directory found")
            return

    # Run all tests
    tester = ComprehensiveASRTester(audio_dir)
    results = tester.run_all_tests()


if __name__ == "__main__":
    main()