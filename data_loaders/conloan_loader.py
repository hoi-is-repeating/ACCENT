# conloan_loader.py - FIXED VERSION
# !/usr/bin/env python3
"""
ConLoan Data Loader - Updated to use data/ directory
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import os


class ConLoanDataLoader:
    """
    Load and process actual ConLoan data files from data/ directory
    """

    def __init__(self, data_dir: str = None):
        """
        Initialize ConLoan data loader

        Args:
            data_dir: Directory containing ConLoan files (default: data/)
        """
        # Default to data/ directory in project root
        if data_dir is None:
            # Look for data/ in current directory or parent directory
            if Path('data').exists():
                data_dir = 'data'
            elif Path('../data').exists():
                data_dir = '../data'
            else:
                data_dir = os.getcwd()

        self.data_dir = Path(data_dir)
        print(f"📁 ConLoan data directory: {self.data_dir.absolute()}")
        self.languages = {}
        self.pairs = {}

    def load_language(self, language: str) -> Dict:
        """
        Load data for a specific language

        Args:
            language: Language name (e.g., 'Spanish')

        Returns:
            Dictionary with loaded data
        """
        # File paths - looking in data/ directory
        json_file = self.data_dir / f"{language}.json"
        replaced_tsv = self.data_dir / f"{language}_replaced_loanwords.tsv"
        all_tsv = self.data_dir / f"{language}_all_replacements.tsv"

        print(f"\n🔍 Looking for ConLoan files:")
        print(f"  - {json_file}")
        print(f"  - {replaced_tsv}")
        print(f"  - {all_tsv}")

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
                print(f"✅ Loaded {len(data['json_data'])} sentences from {json_file.name}")
        else:
            print(f"⚠️ Not found: {json_file}")

        # Load replaced loanwords TSV
        if replaced_tsv.exists():
            pairs = []
            with open(replaced_tsv, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        loanword = parts[0].strip()
                        native = parts[1].strip()
                        if loanword and native and loanword != native:
                            pairs.append({
                                'loanword': loanword,
                                'native': native,
                                'language_code': self._get_language_code(language)
                            })
            data['replaced_pairs'] = pairs
            print(f"✅ Loaded {len(pairs)} replaced pairs from {replaced_tsv.name}")
        else:
            print(f"⚠️ Not found: {replaced_tsv}")

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
            print(f"✅ Loaded {len(all_pairs)} total pairs from {all_tsv.name}")
        else:
            print(f"⚠️ Not found: {all_tsv}")

        self.languages[language] = data
        return data

    # Rest of the methods stay the same...
    def get_validation_pairs(self, language: str, use_replaced_only: bool = True,
                             max_pairs: int = None) -> List[Dict]:
        """Get loanword-native pairs for validation"""
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
            import random
            random.seed(42)
            pairs = random.sample(pairs, max_pairs)

        return pairs

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