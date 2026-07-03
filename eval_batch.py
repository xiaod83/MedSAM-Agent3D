#!/usr/bin/env python
"""
MedSAM-Agent3D — Batch evaluation entry point.

Thin wrapper that delegates to ``medsam_agent3d.cli.main_batch``.

Usage:
    python eval_batch.py \\
        --ts-dir Totalsegmentator_dataset_v201 \\
        --model Qwen/Qwen3-VL-8B-Instruct \\
        --max-samples 10 \\
        --max-turns 5
"""

import os
import sys

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from medsam_agent3d.cli import main_batch

if __name__ == "__main__":
    main_batch()
