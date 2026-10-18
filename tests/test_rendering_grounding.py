"""Rendering regressions for grounded panel clicks (scientific deps optional)."""
import os
import sys
import unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import numpy as np
    from medsam_agent3d.rendering import render_triview, _anatomical_axis_directions
except ImportError as exc:
    raise unittest.SkipTest(f'Rendering dependencies unavailable: {exc}')


class GroundedRenderingTests(unittest.TestCase):
    def test_patient_directions_follow_affine_not_screen_convention(self):
        self.assertEqual(_anatomical_axis_directions(None), {})
        self.assertEqual(_anatomical_axis_directions(np.eye(4)), {'X': 'L->R', 'Y': 'P->A', 'Z': 'I->S'})
        self.assertEqual(_anatomical_axis_directions(np.diag([-1., 1., -1., 1.])), {'X': 'R->L', 'Y': 'P->A', 'Z': 'S->I'})

    def test_no_cyan_focus_lines_cover_visible_tissue(self):
        volume = np.zeros((64, 80, 100), dtype=np.float32)
        image = np.asarray(render_triview(volume, (32, 40, 50), affine=np.eye(4)))
        # Common legend is 30px; panel image origin is (29,52).
        widths = (100, 100, 80)
        heights = (80, 64, 64)
        left = 0
        for width, height in zip(widths, heights):
            anatomy = image[82:82 + height, left + 29:left + 29 + width]
            self.assertFalse(np.any(np.all(anatomy == (0, 220, 255), axis=-1)))
            left += width + 34 + 8
        self.assertTrue(np.any(np.all(image == (0, 220, 255), axis=-1)))

    def test_submillimeter_resizing_keeps_mask_and_image_aligned(self):
        volume = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        image = render_triview(volume, (1, 1, 2), spacing=(.1, .1, .1), mask=np.ones_like(volume))
        self.assertGreater(image.width, 0)
        self.assertGreater(image.height, 0)


if __name__ == '__main__':
    unittest.main()
