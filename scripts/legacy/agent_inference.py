#!/usr/bin/env python
"""
MedSAM-Agent3D â€” Single-case agentic inference entry point.

Thin wrapper that delegates to ``medsam_agent3d.cli.main_single``.

Usage:
    python agent_inference.py \\
        --volume path/to/ct.nii.gz \\
        --mask path/to/liver.nii.gz \\
        --target liver \\
        --model Qwen/Qwen3-VL-8B-Instruct \\
        --max-turns 5
"""

import os
import sys

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from medsam_agent3d.cli import main_single

if __name__ == "__main__":
    main_single()
