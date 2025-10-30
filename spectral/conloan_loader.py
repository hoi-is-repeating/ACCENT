#!/usr/bin/env python3
"""
ConLoan Data Loader - Updated for actual ConLoan format
Handles JSON and TSV files from the ConLoan dataset
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import os


class ConLoanDataLoader:
    """
    Load and process actual ConLoan data files

    ConLoan format:
    - {Language}.json: Full sentences with annotated loanwords/replacements
    - {Language}_replaced_loanwords.tsv: Loanword-native pairs where replacement differs
    - {Language}_all_replacements.tsv: All loanword-native pairs including identical ones
    """

    def __init__(self, data_dir: str = None):
        """
        Initialize ConLoan data loader

        Args:
            data_dir: Directory containing ConLoan files
        """
        # Default to current directory if not specified
        if data_dir is None:
            data_dir = os.getcwd()

        self.data_dir = Path(data_dir)
        self.languages = {}
        self.pairs = {}

    def load_language(self, language: str) -> Dict:
        """
        Load data for a specific language

        Args:
            language: Language name (e.g., 'French', 'Spanish')

        Returns:
            Dictionary with loaded data
        """
        # File paths
        json_file = self.data_dir / f"{language}.json"
        replaced_tsv = self.data_dir / f"{language}_replaced_loanwords.tsv"
        all_tsv = self.data_dir / f"{language}_all_replacements.tsv"

        data = {
            'language': language,
            'json_data': None,
            'replaced_pairs': [],
            'all_pairs': []
        }

        # Load JSON if exists
        if json_file.exists():
            with open(json_file, 'r', encoding='utf-8') as f:
                data['json_data'] = json.load(f)
                print(f"✅ Loaded {len(data['json_data'])} sentences for {language}")

        # Load replaced loanwords TSV
        if replaced_tsv.exists():
            pairs = []
            with open(replaced_tsv, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        loanword = parts[0].strip()
                        native = parts[1].strip()
                        if loanword and native and loanword != native:  # Only non-identical pairs
                            pairs.append({
                                'loanword': loanword,
                                'native': native,
                                'language_code': self._get_language_code(language)
                            })
            data['replaced_pairs'] = pairs
            print(f"✅ Loaded {len(pairs)} replaced loanword pairs for {language}")

        # Load all replacements TSV
        if all_tsv.exists():
            all_pairs = []
            with open(all_tsv, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        loanword = parts[0].strip()
                        native = parts[1].strip()
                        if loanword and native:
                            all_pairs.append({
                                'loanword': loanword,
                                'native': native,
                                'is_identical': loanword == native,
                                'language_code': self._get_language_code(language)
                            })
            data['all_pairs'] = all_pairs
            print(f"✅ Loaded {len(all_pairs)} total pairs for {language}")

        self.languages[language] = data
        return data

    def get_validation_pairs(self, language: str, use_replaced_only: bool = True,
                             max_pairs: int = None) -> List[Dict]:
        """
        Get loanword-native pairs for validation

        Args:
            language: Language name
            use_replaced_only: If True, only use pairs where native != loanword
            max_pairs: Maximum number of pairs to return

        Returns:
            List of dictionaries with loanword-native pairs
        """
        if language not in self.languages:
            self.load_language(language)

        if language not in self.languages:
            return []

        if use_replaced_only:
            pairs = self.languages[language]['replaced_pairs']
        else:
            pairs = [p for p in self.languages[language]['all_pairs']
                     if not p.get('is_identical', False)]

        if max_pairs and len(pairs) > max_pairs:
            # Sample evenly if we have many pairs
            import random
            random.seed(42)  # For reproducibility
            pairs = random.sample(pairs, max_pairs)

        return pairs

    def get_pairs_in_context(self, language: str) -> List[Dict]:
        """
        Get loanword-native pairs with sentence context

        Returns pairs with full sentence context from JSON data
        """
        if language not in self.languages:
            self.load_language(language)

        if not self.languages[language]['json_data']:
            return []

        pairs_with_context = []
        for entry in self.languages[language]['json_data']:
            if 'corresponding_words' in entry:
                for idx, (loan, native) in entry['corresponding_words'].items():
                    pairs_with_context.append({
                        'loanword': loan,
                        'native': native,
                        'context_loanword': entry.get('source_annotated_loanwords', ''),
                        'context_native': entry.get('source_annotated_loanwords_replaced', ''),
                        'language_code': self._get_language_code(language)
                    })

        return pairs_with_context

    def export_for_spectral(self, output_file: str, languages: List[str] = None):
        """
        Export ConLoan data in format suitable for SPECTRAL validation

        Args:
            output_file: Path to save CSV file
            languages: List of languages to include (None = all)
        """
        all_pairs = []

        if not languages:
            # Try to find all available languages
            languages = ['French', 'Spanish', 'German', 'Italian', 'Portuguese', 'Dutch']

        for lang in languages:
            pairs = self.get_validation_pairs(lang, use_replaced_only=True)
            for pair in pairs:
                all_pairs.append({
                    'language': pair['language_code'],
                    'loanword': pair['loanword'],
                    'native': pair['native'],
                    'source_language': 'en',  # Most ConLoan loanwords are from English
                    'dataset': 'ConLoan'
                })

        if all_pairs:
            df = pd.DataFrame(all_pairs)
            df.to_csv(output_file, index=False)
            print(f"✅ Exported {len(df)} pairs to {output_file}")
        else:
            print("❌ No pairs to export")

    def _get_language_code(self, language_name: str) -> str:
        """Map full language names to codes"""
        mapping = {
            'French': 'fr',
            'Spanish': 'es',
            'German': 'de',
            'Italian': 'it',
            'Portuguese': 'pt',
            'Dutch': 'nl',
            'English': 'en'
        }
        return mapping.get(language_name, language_name.lower()[:2])

    def get_statistics(self) -> Dict:
        """Get statistics about loaded data"""
        stats = {
            'languages': list(self.languages.keys()),
            'total_pairs': 0,
            'replaced_pairs': 0,
            'identical_pairs': 0,
            'per_language': {}
        }

        for lang, data in self.languages.items():
            replaced = len(data['replaced_pairs'])
            all_pairs = len(data['all_pairs'])
            identical = sum(1 for p in data['all_pairs'] if p.get('is_identical', False))

            stats['total_pairs'] += all_pairs
            stats['replaced_pairs'] += replaced
            stats['identical_pairs'] += identical
            stats['per_language'][lang] = {
                'total': all_pairs,
                'replaced': replaced,
                'identical': identical,
                'percentage_replaced': (replaced / all_pairs * 100) if all_pairs > 0 else 0
            }

        return stats


def main():
    """Test the ConLoan data loader"""
    print("""
╔════════════════════════════════════════════════════════════╗
║                ConLoan Data Loader v1.0                    ║
╚════════════════════════════════════════════════════════════╝
    """)

    # Initialize loader
    loader = ConLoanDataLoader()

    # Load French data
    print("\n📂 Loading French data...")
    french_data = loader.load_language('French')

    # Get statistics
    stats = loader.get_statistics()
    print(f"\n📊 Statistics:")
    print(f"  Total pairs: {stats['total_pairs']}")
    print(f"  Replaced (different) pairs: {stats['replaced_pairs']}")
    print(f"  Identical pairs: {stats['identical_pairs']}")

    if 'French' in stats['per_language']:
        fr_stats = stats['per_language']['French']
        print(f"\n  French specific:")
        print(f"    Total: {fr_stats['total']}")
        print(f"    Replaced: {fr_stats['replaced']} ({fr_stats['percentage_replaced']:.1f}%)")

    # Get some validation pairs
    print("\n🔍 Sample validation pairs:")
    pairs = loader.get_validation_pairs('French', max_pairs=5)
    for i, pair in enumerate(pairs, 1):
        print(f"  {i}. '{pair['loanword']}' → '{pair['native']}'")

    # Export for SPECTRAL
    output_file = "/home/claude/SPECTRAL_project/data/conloan_french.csv"
    print(f"\n💾 Exporting to SPECTRAL format...")
    loader.export_for_spectral(output_file, languages=['French'])

    print("\n✅ ConLoan data loaded successfully!")


if __name__ == "__main__":
    main()