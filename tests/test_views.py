"""Panel click grounding regressions; runs with Python standard library only."""
import os
import sys
import unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from medsam_agent3d.views import view_point_to_voxel


class ViewPointTests(unittest.TestCase):
    shape = (179, 193, 295)
    center = (89, 96, 147)

    def test_each_panel_fixes_the_displayed_slice_axis(self):
        self.assertEqual(view_point_to_voxel('axial', [0, 999], self.center, self.shape), (89, 192, 0))
        self.assertEqual(view_point_to_voxel('coronal', [999, 0], self.center, self.shape), (0, 96, 294))
        self.assertEqual(view_point_to_voxel('sagittal', [999, 999], self.center, self.shape), (178, 192, 147))

    def test_femur_click_uses_visible_axial_location(self):
        # Known uploaded case: left femur lies near x=54,y=68 at displayed z=89.
        u, v = round(54 / 294 * 999), round(68 / 192 * 999)
        self.assertEqual(view_point_to_voxel('axial', [u, v], self.center, self.shape), (89, 68, 54))

    def test_full_panel_edges_and_singleton_dimensions(self):
        for view in ('axial', 'coronal', 'sagittal'):
            with self.subTest(view=view):
                self.assertEqual(view_point_to_voxel(view, [999, 999], (0, 0, 0), (1, 1, 1)), (0, 0, 0))
                mapped = view_point_to_voxel(view, [0, 0], self.center, self.shape)
                self.assertTrue(all(0 <= p < s for p, s in zip(mapped, self.shape)))

    def test_invalid_coordinates_are_rejected_before_mapping(self):
        for point in ([True, 0], [float('nan'), 0], [0, float('inf')], [-1, 0], [1000, 0], [10 ** 1000, 0], [10.5, 20], [1], '50,50'):
            with self.subTest(point=point), self.assertRaises(ValueError):
                view_point_to_voxel('axial', point, self.center, self.shape)
        with self.assertRaises(ValueError):
            view_point_to_voxel('unknown', [0, 0], self.center, self.shape)

    def test_invalid_focus_or_shape_is_rejected(self):
        for center, shape in (((-1, 0, 0), self.shape), ((179, 0, 0), self.shape), (self.center, (0, 10, 10)), ((1.5, 2, 3), self.shape), ((True, 2, 3), self.shape)):
            with self.subTest(center=center, shape=shape), self.assertRaises(ValueError):
                view_point_to_voxel('axial', [0, 0], center, shape)


if __name__ == '__main__':
    unittest.main()
