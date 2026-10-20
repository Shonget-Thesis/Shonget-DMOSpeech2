"""
Driver script to complete all 100 development utterances (200 condition-pairs)
and immediately generate the complete Objective 1 analysis report and summary.
"""

import os
import sys

HANDOVER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if HANDOVER_DIR not in sys.path:
    sys.path.insert(0, HANDOVER_DIR)

from phase1_dev_grid_100spk.run_grid import run_grid
from phase1_dev_grid_100spk.analyze import run_analysis

if __name__ == "__main__":
    print("=" * 80)
    print("LAUNCHING COMPLETE DEVELOPMENT RUN (ALL 100 UTTERANCES / 200 PAIRS)")
    print("=" * 80)

    # 1. Complete remaining pairs (from pair 62 through pair 200)
    run_grid(max_instances=None)

    # 2. Automatically compute comprehensive Chapter 3 Objective 1 metrics
    print("\n" + "=" * 80)
    print("GENERATING COMPREHENSIVE CHAPTER 3 OBJECTIVE 1 REPORT")
    print("=" * 80)
    run_analysis()
    print("\n[SUCCESS] All 100 development utterances processed and analyzed.")
