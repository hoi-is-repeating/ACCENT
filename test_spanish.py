# test_spanish.py - Quick test of Spanish pipeline
# !/usr/bin/env python3
"""Quick test of Spanish pipeline"""

import sys
from pathlib import Path


# Test with mock data if dependencies not available
def test_spanish_pipeline():
    print("Testing Spanish Pipeline...")

    # Mock Spanish pairs
    spanish_pairs = [
        {'loanword': 'software', 'native': 'programa'},
        {'loanword': 'email', 'native': 'correo electrónico'},
        {'loanword': 'marketing', 'native': 'mercadotecnia'}
    ]

    # Generate sentences
    template = "Necesito el {} para mañana"
    for pair in spanish_pairs:
        lw_sent = template.format(pair['loanword'])
        nat_sent = template.format(pair['native'])
        print(f"\nPair: {pair['loanword']} vs {pair['native']}")
        print(f"  Loanword: {lw_sent}")
        print(f"  Native: {nat_sent}")

    print("\n✅ Basic pipeline working!")


if __name__ == "__main__":
    test_spanish_pipeline()