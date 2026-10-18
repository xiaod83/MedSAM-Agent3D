"""Failure and optional-ground-truth regressions using lightweight models."""
import numpy as np
from medsam_agent3d.loop import run_agent_loop

class Agent:
    def __init__(self, actions):
        self.actions = iter(actions)
    def generate_action(self, **kwargs):
        return next(self.actions)

class Sam:
    def set_image(self, volume, spacing=None, affine=None):
        self.volume = volume
    def predict_mask(self, **kwargs):
        mask = np.zeros_like(self.volume, dtype=np.uint8)
        if kwargs.get('points'):
            mask[tuple(kwargs['points'][0])] = 1
        return mask, None

def run(tmp_path, actions, mask=None):
    return run_agent_loop(Agent(actions), Sam(), np.ones((8, 10, 12)), mask,
                          'target', 'case', max_turns=3, output_dir=str(tmp_path))

POINT = {'name': 'add_3d_point', 'arguments': {'point_3d': [500, 500, 500], 'point_type': 'positive'}}
STOP = {'name': 'stop_action'}

def test_parse_failure_never_counts_as_success(tmp_path):
    result = run(tmp_path, [{'name': 'invalid_action'}, {'name': 'invalid_action'}])
    assert not result.success
    assert result.termination_reason == 'invalid_action'
    assert len(result.turns) == 2

def test_no_gt_inference_returns_mask_and_null_metrics(tmp_path):
    result = run(tmp_path, [POINT, STOP])
    assert result.success
    assert result.final_iou is None and not result.metrics_available
    assert result.predicted_mask.sum() == 1

def test_empty_bbox_prediction_keeps_three_coordinate_focus(tmp_path):
    bbox = {'name': 'add_3d_bbox', 'arguments': {'bbox_3d': [0, 0, 0, 999, 999, 999]}}
    result = run(tmp_path, [bbox, POINT, STOP])
    assert result.success
    assert all(len(turn.center) == 3 for turn in result.turns)

def test_retry_failure_preserves_last_mask_and_metrics(tmp_path):
    mask = np.zeros((8, 10, 12), dtype=np.uint8)
    mask[4, 5, 6] = 1
    result = run(tmp_path, [POINT, {'name': 'invalid_action'}, {'name': 'invalid_action'}], mask)
    assert not result.success
    assert result.predicted_mask.sum() == 1
    assert result.final_iou == 1.0


def test_navigation_changes_slices_without_sam_prediction(tmp_path):
    navigate = {"name": "navigate", "arguments": {"point_3d": [0, 999, 0]}}
    result = run(tmp_path, [navigate, POINT, STOP])
    assert result.success
    assert result.turns[0].next_center == (0, 9, 0)
    assert result.turns[0].mask_voxels == 0


def test_panel_point_resolves_on_current_slice_and_reports_no_overlap(tmp_path):
    action = {'name': 'add_view_point', 'arguments': {'view': 'axial', 'point_2d': [0, 999], 'point_type': 'positive'}}
    mask = np.zeros((8, 10, 12), dtype=np.uint8)
    mask[1, 1, 1] = 1
    result = run(tmp_path, [action, STOP], mask)
    assert result.success
    assert result.turns[0].action['resolved_point_zyx'] == [4, 9, 0]
    assert result.predicted_mask[4, 9, 0] == 1
    assert result.target_overlap is False
    assert result.quality_status == 'no_overlap'
