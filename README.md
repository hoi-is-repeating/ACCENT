# ACCENT + SPECTRAL Integration

## Project Architecture

```
ACCENT (Main Framework)
├── Research Goal: Detect ASR bias between loanwords and native equivalents
├── Data Source: ConLoan dataset (actual data files)
└── Validation: SPECTRAL module (your existing implementation)
    ├── spectral_proper.py - Core validation metrics
    ├── spectral_validation.py - Enhanced validation pipeline
    └── conloan_loader.py - ConLoan data handling
```

## Directory Structure

```
ACCENT_project/
├── accent_integrated.py      # Main ACCENT framework
├── run_accent.py             # Launcher script with proper paths
├── spectral_module/          # Your SPECTRAL implementation (separate module)
│   ├── spectral_proper.py   # DTW, ASPF, PER metrics
│   ├── spectral_validation.py # Enhanced validation
│   └── conloan_loader.py    # ConLoan data loader
├── data/                     # ConLoan data files go here
│   ├── French.json
│   ├── French_replaced_loanwords.tsv
│   └── [other language files]
├── results/                  # Experiment outputs
│   └── ACCENT_French/        # Per-language results
└── README.md                 # This file
```

## How It Works

### 1. ACCENT Framework (Main)
- **Purpose**: Orchestrates the entire experiment
- **Components**:
  - Data loading (via ConLoan loader)
  - Sentence generation
  - SPECTRAL validation call
  - ASR bias testing
  - Report generation

### 2. SPECTRAL Module (Validation)
- **Purpose**: Validates TTS pronunciation differentiation
- **Your Implementation**:
  - DTW distance calculation
  - ASPF (Angular Similarity of Phoneme Frequencies)
  - PER (Phoneme Error Rate)
  - Combined scoring with literature-based thresholds

### 3. Integration Points

```python
# ACCENT imports SPECTRAL as a module
from spectral_proper import SPECTRALValidator
from spectral_validation import EnhancedSPECTRALValidator
from conloan_loader import ConLoanDataLoader

# ACCENT uses SPECTRAL for validation
validator = EnhancedSPECTRALValidator()
results = validator.validate_batch(data)

# If validation passes, proceed with ASR experiments
if results['percentage_strong'] > 50:
    run_asr_experiments()
```

## Quick Start

### 1. Setup ConLoan Data
Place ConLoan data files in the `data/` directory:
```bash
# Expected files for each language:
data/French.json
data/French_replaced_loanwords.tsv
data/French_all_replacements.tsv
```

### 2. Install Dependencies

```bash
# Core requirements
pip install numpy pandas

# For SPECTRAL validation (your code)
pip install gtts librosa scipy phonemizer

# For ASR experiments (optional)
pip install openai-whisper
```

### 3. Run Experiment

#### Interactive Mode
```bash
cd ACCENT_project
python run_accent.py
# Follow prompts to select language and parameters
```

#### Command Line Mode
```bash
python accent_integrated.py --language French --n-pairs 50
```

#### With Custom Paths
```bash
python accent_integrated.py \
    --language Spanish \
    --n-pairs 100 \
    --output-dir results_spanish \
    --spectral-dir /path/to/spectral
```

## Validation Process (SPECTRAL)

Your SPECTRAL module validates TTS differentiation using:

1. **DTW Distance** (30% weight)
   - Dynamic Time Warping on MFCC features
   - Standard in speech processing

2. **ASPF** (30% weight) 
   - Angular Similarity of Phoneme Frequencies
   - From SIGUL 2022

3. **PER** (20% weight)
   - Phoneme Error Rate 
   - Tahon et al., INTERSPEECH 2016

4. **Duration Ratio** (20% weight)
   - Temporal comparison

### Validation Thresholds (from SynthASR 2021)
- **Strong**: Combined score > 0.3
- **Moderate**: 0.15 - 0.3  
- **Low**: < 0.15

### Decision Criteria
- **>50% strong differentiation**: ✅ Proceed with ASR experiments
- **25-50% strong**: ⚠️ Proceed with caution
- **<25% strong**: ❌ Insufficient differentiation

## ASR Bias Testing (ACCENT)

If SPECTRAL validation passes, ACCENT:

1. **Generates TTS audio** for sentence pairs
2. **Transcribes with ASR** (Whisper-large-v3)
3. **Calculates WER** for loanword vs native sentences
4. **Computes bias metrics**:
   - WER difference
   - Effect size (Cohen's d)
   - Statistical significance

## Output Files

After running an experiment, you'll find:

```
results/ACCENT_French/
├── ACCENT_report_French_20250101_120000.md   # Human-readable report
├── ACCENT_results_French_20250101_120000.json # Raw results data
└── validation_details.json                    # SPECTRAL validation details
```

## Example Report Output

```markdown
# ACCENT Experiment Report
**Language**: French
**Framework**: ACCENT + SPECTRAL

## SPECTRAL Validation
- Strong differentiation: 52.3%
- Status: ✅ PASSED

## ASR Bias Results
- Mean Loanword WER: 0.087
- Mean Native WER: 0.102
- Preference: LOANWORD
- Effect Size: 0.34 (small)
```

## Interpreting Results

### Preference Direction
- **Loanword preference**: WER difference < 0
  - ASR recognizes loanwords better than native equivalents
  - Suggests lexical bias toward international terms

- **Native preference**: WER difference > 0
  - ASR recognizes native terms better
  - Suggests language-specific training effects

### Effect Size Interpretation
- **Large** (>0.8): Strong systematic bias
- **Medium** (0.5-0.8): Moderate bias
- **Small** (0.2-0.5): Small but detectable bias
- **Negligible** (<0.2): No meaningful bias

## Troubleshooting

### "ConLoan loader not found"
- Ensure `conloan_loader.py` is in `spectral_module/`
- Check Python path includes the module directory

### "SPECTRAL modules not found"
- Verify files are in `spectral_module/`
- Use `--spectral-dir` to specify custom location

### "Validation failed"
- Check that ConLoan data has sufficient non-identical pairs
- Verify TTS is generating distinct pronunciations
- Consider using Enhanced validator for better metrics

## Citation

If you use this framework, please cite:

```bibtex
@inproceedings{accent2025,
  title={ACCENT: Revealing Systematic Loanword Preference in Multilingual ASR},
  author={Your Name},
  booktitle={Proceedings of ACL},
  year={2025}
}

@inproceedings{spectral2025,
  title={SPECTRAL: Validating TTS for Pronunciation Studies},
  author={Your Name},
  booktitle={Technical Report},
  year={2025}
}

@inproceedings{conloan2025,
  title={ConLoan: Measuring Lexical Preference in Multilingual Language Models},
  author={ConLoan Authors},
  booktitle={Proceedings of ACL},
  year={2025}
}
```

## Key Innovation

**Controlled Pronunciation as a Feature**: By using TTS with SPECTRAL validation, we ensure consistent pronunciation that isolates lexical bias from acoustic variation. This methodological innovation allows us to definitively test whether ASR preference is due to lexical priors or acoustic confusion.

## Contact & Collaboration

This research bridges:
- **Computational Linguistics**: ASR bias and fairness
- **Phonetics/Phonology**: Loanword adaptation
- **Speech Technology**: Multilingual ASR evaluation

---

*ACCENT integrates with your existing SPECTRAL implementation to provide a comprehensive framework for ASR loanword bias research.*
