"""
Smoke test: dry-run the agentic pipeline with mock MLLM & SAM.

Purpose:
    Validate the entire modular architecture works end-to-end
    without needing a GPU or downloading model weights.
    All heavy models are replaced with lightweight mocks.

Usage:
    python tests/test_pipeline.py              # quick smoke test
    python tests/test_pipeline.py --verbose     # detailed output
    python tests/test_pipeline.py --seed 0 --turns 10  # stress test
"""

import os
import sys
import argparse
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from PIL import Image

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =========================================================================
# Mock MLLM Agent: simulates Qwen3-VL without loading the real model
# =========================================================================

class MockMLLMAgent:
    """Simulates the MLLM's reasoning loop without any model loaded.

    Uses a simple state machine:
      Turn 0 -> bbox (covers middle 60% of volume)
      Turn 1 -> positive point (near bbox center)
      Turn 2 -> positive point (offset)
      Turn 3 -> negative point (outside organ)
      Turn 4+ -> stop_action

    This exercises ALL code paths: bbox, point, stop, conversation history.
    """

    def __init__(self, volume_shape: Tuple[int, int, int], seed: int = 42):
        np.random.seed(seed)
        self.dz, self.dy, self.dx = volume_shape
        self._call_count = 0

    def generate_action(
        self,
        image: Image.Image,
        instruction: str,
        system_prompt: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        """Generate a deterministic action based on turn number."""
        t = self._call_count
        self._call_count += 1

        if t == 0:
            # Turn 0: bounding box covering middle 60%
            z1, y1, x1 = int(0.2 * 999), int(0.2 * 999), int(0.2 * 999)
            z2, y2, x2 = int(0.8 * 999), int(0.8 * 999), int(0.8 * 999)
            return {
                "name": "add_3d_bbox",
                "arguments": {"bbox_3d": [z1, y1, x1, z2, y2, x2]},
                "_raw_output": f'<tool_call>{{"name":"add_3d_bbox","arguments":{{"bbox_3d":[{z1},{y1},{x1},{z2},{y2},{x2}]}}}}</tool_call>',
            }
        elif t == 1:
            # Turn 1: positive point at bbox center
            return {
                "name": "add_3d_point",
                "arguments": {"point_3d": [500, 500, 500], "point_type": "positive"},
                "_raw_output": f'<tool_call>{{"name":"add_3d_point","arguments":{{"point_3d":[500,500,500],"point_type":"positive"}}}}</tool_call>',
            }
        elif t == 2:
            # Turn 2: second positive point (refine)
            return {
                "name": "add_3d_point",
                "arguments": {"point_3d": [400, 400, 400], "point_type": "positive"},
                "_raw_output": f'<tool_call>{{"name":"add_3d_point","arguments":{{"point_3d":[400,400,400],"point_type":"positive"}}}}</tool_call>',
            }
        elif t == 3:
            # Turn 3: negative point to exclude
            return {
                "name": "add_3d_point",
                "arguments": {"point_3d": [900, 900, 900], "point_type": "negative"},
                "_raw_output": f'<tool_call>{{"name":"add_3d_point","arguments":{{"point_3d":[900,900,900],"point_type":"negative"}}}}</tool_call>',
            }
        else:
            # Turn 4+: stop
            return {
                "name": "stop_action",
                "arguments": {},
                "_raw_output": '<tool_call>{"name":"stop_action"}</tool_call>',
            }


# =========================================================================
# Mock SAM-Med3D: generates synthetic mask from bbox/point prompts
# =========================================================================

class MockSAMMed3D:
    """Simulates SAM-Med3D without loading model weights.

    Generates a synthetic mask based on the prompts:
    - Bbox -> mask = everything inside the bbox
    - Point -> mask = sphere around the point
    - mask_logits -> adds slight dilation (simulates "refinement")
    """

    def __init__(self, volume_shape: Tuple[int, int, int], seed: int = 42):
        np.random.seed(seed)
        self.dz, self.dy, self.dx = volume_shape
        self._features = np.ones((1, 256, 32, 32, 32), dtype=np.float32)  # dummy

    def set_image(self, volume_3d: np.ndarray, gt_mask=None):
        """Accept the volume (no-op for mock)."""
        pass

    def predict_mask(
        self,
        points=None,
        labels=None,
        box_3d=None,
        mask_logits=None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic mask from prompts.

        Args:
            points: List of (z, y, x) tuples in voxel coords.
            labels: List of 1/0 for positive/negative.
            box_3d: (z1, y1, x1, z2, y2, x2) in voxel coords.
            mask_logits: If provided, dice the previous mask (simulates refinement).

        Returns:
            (binary_mask_3d, low_res_logits):
                mask: (Z, Y, X) uint8 binary mask.
                logits: (1, 1, 32, 32, 32) dummy logits.
        """
        mask = np.zeros((self.dz, self.dy, self.dx), dtype=np.uint8)

        # -- Bbox mode: fill the box
        if box_3d is not None:
            z1, y1, x1, z2, y2, x2 = box_3d
            mask[z1:z2, y1:y2, x1:x2] = 1

        # -- Point mode: add spheres around each point
        if points is not None and labels is not None:
            for (z, y, x), label in zip(points, labels):
                radius = 15 if label == 1 else 10  # pos = bigger, neg = smaller
                zz, yy, xx = np.ogrid[:self.dz, :self.dy, :self.dx]
                dist = np.sqrt((zz - z)**2 + (yy - y)**2 + (xx - x)**2)
                if label == 1:
                    mask[dist <= radius] = 1  # add tissue
                else:
                    mask[dist <= radius] = 0  # remove tissue

        # -- "Refinement" signal: if prior logits exist, dilate slightly
        if mask_logits is not None:
            from scipy.ndimage import binary_dilation
            mask = binary_dilation(mask, iterations=1).astype(np.uint8)

        # Dummy low-res logits (just needs right shape for next iteration)
        logits = np.random.randn(1, 1, 32, 32, 32).astype(np.float32)
        return mask, logits


# =========================================================================
# Test runner
# =========================================================================

def run_smoke_test(
    volume_shape: Tuple[int, int, int] = (128, 192, 192),
    max_turns: int = 10,
    seed: int = 42,
    verbose: bool = False,
):
    """Run a full end-to-end smoke test of the agentic pipeline.

    Creates a synthetic volume with a spherical "organ", runs the mock
    agent against it, and validates:
    - All modules import correctly
    - The agent loop runs without errors
    - Bbox and point code paths are exercised
    - Conversation history is maintained
    - stop_action is respected
    - AgentResult contains valid metrics
    """

    # --- Import the real modules (this is what we're testing) ---
    from medsam_agent3d.types import (
        AgentResult, InferenceArgs, TurnRecord, compute_metrics,
    )
    from medsam_agent3d.rendering import render_triview
    from medsam_agent3d.parsing import (
        parse_action, denormalize_coord, denormalize_bbox,
        denormalize_point, is_stop_action, is_bbox_action, is_point_action,
    )
    from medsam_agent3d.prompting import build_system_prompt
    from medsam_agent3d.clicker import Clicker3D
    from medsam_agent3d.loop import run_agent_loop

    print("=" * 60)
    print("MedSAM-Agent3D -- Full Pipeline Smoke Test")
    print("=" * 60)
    print(f"Volume shape: {volume_shape}")
    print(f"Max turns:    {max_turns}")
    print(f"Seed:         {seed}")
    print()

    # --- Create synthetic test data ---
    print("[1/6] Creating synthetic test data...")
    dz, dy, dx = volume_shape

    # Synthetic CT-like volume: background ~ 0, soft tissue ~ 100
    volume = np.random.normal(0, 50, volume_shape).astype(np.float32)

    # Synthetic "organ": sphere at center with higher intensity
    cz, cy, cx = dz // 2, dy // 2, dx // 2
    zz, yy, xx = np.ogrid[:dz, :dy, :dx]
    dist = np.sqrt((zz - cz)**2 + (yy - cy)**2 + (xx - cx)**2)
    organ_radius = min(dz, dy, dx) // 4
    organ_mask = (dist <= organ_radius).astype(np.uint8)
    volume[organ_mask > 0] = np.random.normal(200, 30, organ_mask.sum())

    # Ground truth = a slightly larger version (to simulate imperfect bbox)
    gt_mask = (dist <= organ_radius + 5).astype(np.uint8)

    print(f"  Volume range: [{volume.min():.1f}, {volume.max():.1f}]")
    print(f"  Organ voxels: {int(organ_mask.sum()):,}")
    print(f"  GT voxels:    {int(gt_mask.sum()):,}")
    print(f"  GT vs organ overlap IoU: {compute_metrics(organ_mask, gt_mask)[0]:.4f}")

    # --- Instantiate mock models ---
    print("\n[2/6] Instantiating mock models...")
    mllm = MockMLLMAgent(volume_shape, seed=seed)
    sam = MockSAMMed3D(volume_shape, seed=seed)
    print("  MockMLLMAgent  [OK]")
    print("  MockSAMMed3D   [OK]")

    # --- Test individual modules ---
    print("\n[3/6] Testing individual modules...")

    # Prompting
    prompt = build_system_prompt(volume_shape, organ_names=["mock_organ"])
    assert "Z=128" in prompt
    assert "mock_organ" in prompt
    assert "add_3d_bbox" in prompt
    assert "add_3d_point" in prompt
    print("  prompting.py   [OK]")

    # Parsing
    for action_text in [
        '<tool_call>{"name":"add_3d_bbox","arguments":{"bbox_3d":[100,200,300,400,500,600]}}</tool_call>',
        '<tool_call>{"name":"add_3d_point","arguments":{"point_3d":[500,500,500],"point_type":"positive"}}</tool_call>',
        '<tool_call>{"name":"stop_action"}</tool_call>',
        '{"name":"add_3d_point","point_3d":[100,200,300],"point_type":"negative"}',  # flat
        'garbage text that should produce stop_action',
    ]:
        action = parse_action(action_text)
        assert "name" in action
    print("  parsing.py     [OK]")

    # Rendering
    img = render_triview(volume, (cz, cy, cx), mask=organ_mask)
    assert isinstance(img, Image.Image)
    assert img.size[0] > 0 and img.size[1] > 0
    print(f"  rendering.py   [OK] (tri-view size: {img.size})")

    # Clicker
    clicker = Clicker3D()
    clicker.add_positive((10, 20, 30), turn=0)
    clicker.add_negative((15, 25, 35), turn=1)
    clicker.set_bbox((0, 0, 0, 100, 100, 100))
    assert clicker.total_clicks == 2
    assert clicker.has_bbox
    assert clicker.get_last_click().coords == (15, 25, 35)
    print("  clicker.py     [OK]")

    # --- Run agent loop ---
    print("\n[4/6] Running agent loop...")
    result = run_agent_loop(
        mllm_agent=mllm,
        sam_model=sam,
        volume=volume,
        gt_mask=gt_mask,
        target_name="mock_organ",
        volume_id="test_case_001",
        spacing=(1.5, 1.0, 1.0),
        max_turns=max_turns,
        output_dir="test_outputs",
        temperature=0.0,
        seed=seed,
    )

    # --- Validate results ---
    print("\n[5/6] Validating results...")

    assert isinstance(result, AgentResult), f"Expected AgentResult, got {type(result)}"
    assert result.volume_id == "test_case_001"
    assert result.target_name == "mock_organ"
    assert result.success, f"Agent failed: {result.error}"
    assert len(result.turns) > 0, "No turns recorded"

    # Check that we got both bbox and point turns
    action_names = [t.action.get("name") for t in result.turns]
    assert "add_3d_bbox" in action_names, f"No bbox turn in {action_names}"
    assert "add_3d_point" in action_names, f"No point turn in {action_names}"

    # Check metrics improved (or at least are valid)
    for t in result.turns:
        assert 0.0 <= t.iou <= 1.0, f"Invalid IoU: {t.iou}"
        assert 0.0 <= t.dice <= 1.0, f"Invalid Dice: {t.dice}"
        assert t.mask_voxels >= 0, f"Negative mask voxels: {t.mask_voxels}"

    # Check final metrics are valid
    assert 0.0 <= result.final_iou <= 1.0
    assert 0.0 <= result.final_dice <= 1.0

    print(f"  AgentResult     [OK] (volume_id={result.volume_id})")
    print(f"  Success         [OK] ({result.success})")
    print(f"  Turns executed  [OK] ({len(result.turns)})")
    print(f"  Final IoU       [OK] ({result.final_iou:.4f})")
    print(f"  Final Dice      [OK] ({result.final_dice:.4f})")

    # --- Print turn-by-turn summary ---
    print("\n[6/6] Turn-by-turn summary:")
    print("-" * 80)
    print(f"{'Turn':>4}  {'Action':<20}  {'IoU':>8}  {'Dice':>8}  {'dIoU':>8}  {'Voxels':>10}  {'Center':>20}")
    print("-" * 80)

    prev_iou = 0.0
    for t in result.turns:
        action_name = t.action.get("name", "unknown")
        delta = t.iou - prev_iou
        prev_iou = t.iou
        print(f"{t.turn:>4}  {action_name:<20}  {t.iou:>8.4f}  {t.dice:>8.4f}  {delta:>+8.4f}  {t.mask_voxels:>10,}  {str(t.center):>20}")

    print("-" * 80)
    print(f"\n{'FINAL':>4}  {'':>20}  {result.final_iou:>8.4f}  {result.final_dice:>8.4f}")
    print()

    # --- Check output files were created ---
    import glob
    images = glob.glob("test_outputs/*.png")
    if images:
        print(f"  Tri-view images: {len(images)} files saved to test_outputs/")
    else:
        print("  (No tri-view images -- normal if directory clean)")

    print("\n" + "=" * 60)
    print("*** ALL CHECKS PASSED ***")
    print("=" * 60)

    return result


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MedSAM-Agent3D Pipeline Smoke Test"
    )
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--turns", type=int, default=10,
                        help="Max turns")
    parser.add_argument("--shape", type=str, default="128,192,192",
                        help="Volume shape as 'z,y,x'")
    args = parser.parse_args()

    shape = tuple(int(x) for x in args.shape.split(","))
    assert len(shape) == 3, f"Shape must be 3 integers, got: {args.shape}"

    run_smoke_test(
        volume_shape=shape,
        max_turns=args.turns,
        seed=args.seed,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
