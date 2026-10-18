"""Physical geometry regressions; no checkpoint or GPU is required."""
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import numpy as np
    import nibabel as nib
    from medsam_agent3d.io import load_nifti, save_nifti, get_totalsegmentator_dataset
except ImportError as exc:
    raise unittest.SkipTest(f'NIfTI test dependencies unavailable: {exc}')


class NiftiGeometryTests(unittest.TestCase):
    def test_reflected_anisotropic_volume_and_mask_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, 'source.nii.gz')
            output = os.path.join(directory, 'output.nii.gz')
            data = np.zeros((9, 11, 13), dtype=np.uint8)
            data[2:5, 6:8, 3:9] = 1
            affine = np.array([[-2., 0, 0, 30], [0, 3., 0, -9], [0, 0, 4., 12], [0, 0, 0, 1]])
            nib.save(nib.Nifti1Image(data, affine), source)
            canonical, spacing, canonical_affine = load_nifti(source, return_affine=True)
            self.assertEqual(spacing, (4., 3., 2.))
            self.assertEqual(canonical.shape, (13, 11, 9))
            self.assertGreater(canonical_affine[0, 0], 0)
            save_nifti(canonical.astype(np.uint8), output, affine=canonical_affine, reference_path=source)
            restored = nib.load(output)
            np.testing.assert_allclose(restored.affine, affine)
            np.testing.assert_array_equal(restored.get_fdata(), data)

    def test_mask_is_aligned_to_reference_world_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = os.path.join(directory, 'reference.nii.gz')
            mask = os.path.join(directory, 'mask.nii.gz')
            data = np.zeros((8, 9, 10), dtype=np.uint8)
            data[3, 4, 5] = 1
            nib.save(nib.Nifti1Image(np.zeros_like(data), np.eye(4)), reference)
            shifted_affine = np.eye(4)
            shifted_affine[0, 3] = 1
            nib.save(nib.Nifti1Image(data, shifted_affine), mask)
            aligned, _ = load_nifti(mask, reference_path=reference, is_label=True)
            self.assertEqual(aligned[5, 4, 4], 1)
            self.assertEqual(int(aligned.sum()), 1)

    def test_split_filter_is_strict_and_sorted(self):
        with tempfile.TemporaryDirectory() as directory:
            for subject in ('s0002', 's0001'):
                root = os.path.join(directory, subject)
                os.makedirs(os.path.join(root, 'segmentations'))
                open(os.path.join(root, 'ct.nii.gz'), 'wb').close()
                open(os.path.join(root, 'segmentations', 'liver.nii.gz'), 'wb').close()
            with self.assertRaises(FileNotFoundError):
                get_totalsegmentator_dataset(directory, split='test')
            with open(os.path.join(directory, 'meta.csv'), 'w') as handle:
                handle.write('image_id;split\ns0002;train\ns0001;test\n')
            entries = get_totalsegmentator_dataset(directory, split='test')
            self.assertEqual([item['volume_id'] for item in entries], ['s0001'])
            self.assertEqual(entries[0]['volume_path'], entries[0]['ct_path'])


class CropGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            from medsam_agent3d.sam import FrozenSAMMed3D
        except ImportError as exc:
            raise unittest.SkipTest(f'SAM geometry dependencies unavailable: {exc}')
        cls.torch = torch
        cls.wrapper = FrozenSAMMed3D

    def make_wrapper(self, shape=(80, 100, 220), spacing=(2., 1.5, 1.)):
        wrapper = self.wrapper.__new__(self.wrapper)
        wrapper.device = self.torch.device('cpu')
        wrapper.model = SimpleNamespace(image_encoder=lambda image: image)
        wrapper.set_image(np.ones(shape, dtype=np.float32), spacing=spacing)
        return wrapper

    def test_auto_device_uses_cpu_for_small_gpu_and_explicit_cuda_is_checked(self):
        from medsam_agent3d.sam import _select_device
        with patch.object(self.torch.cuda, 'is_available', return_value=True), patch.object(self.torch.cuda, 'get_device_properties', return_value=SimpleNamespace(total_memory=2 * 1024 ** 3)):
            with self.assertWarns(RuntimeWarning):
                self.assertEqual(str(_select_device('auto')), 'cpu')
        with patch.object(self.torch.cuda, 'is_available', return_value=False):
            self.assertEqual(str(_select_device('auto')), 'cpu')
            with self.assertRaisesRegex(RuntimeError, 'explicitly requested'):
                _select_device('cuda')

    def test_constant_volume_padding_stays_finite(self):
        wrapper = self.make_wrapper(shape=(20, 30, 40), spacing=(1.5, 1.5, 1.5))
        self.assertTrue(bool(self.torch.isfinite(wrapper._features).all()))
        self.assertLess(float(wrapper._features.abs().max()), 100.)

    def test_prompt_maps_to_same_physical_location(self):
        wrapper = self.make_wrapper()
        point = np.array([21., 32., 100.])
        roi = wrapper._map_original_to_roi(point)
        world_from_roi = wrapper._roi_affine @ np.r_[roi, 1.]
        np.testing.assert_allclose(world_from_roi, wrapper._orig_affine @ np.r_[point[::-1], 1.], atol=1e-6)
        self.assertTrue(np.all((roi >= 0) & (roi < 128)))

    def test_original_edge_prompt_remains_inside_resampled_crop(self):
        wrapper = self.make_wrapper(shape=(140, 140, 401), spacing=(1.5, 1.5, 1.))
        point = np.array([139, 139, 400])
        wrapper._encode_crop(point)
        roi = wrapper._map_original_to_roi(point)
        self.assertTrue(np.all((roi >= 0) & (roi <= 127)), roi)

    def test_outside_prompt_is_recentered_and_prior_reprojected(self):
        wrapper = self.make_wrapper(shape=(140, 140, 400), spacing=(1.5, 1.5, 1.5))
        point = np.array([60, 65, 380])
        self.assertGreater(wrapper._map_original_to_roi(point)[0], 127)
        wrapper._global_mask[55:65, 60:70, 375:385] = 1
        wrapper._encode_crop(point)
        roi = wrapper._map_original_to_roi(point)
        self.assertTrue(np.all((roi >= 0) & (roi < 128)))
        prior = wrapper._prior_in_crop()
        self.assertEqual(tuple(prior.shape), (1, 1, 32, 32, 32))
        self.assertGreater(float(prior.max()), 0)
        self.assertLess(float(prior.min()), 0)

    def test_prediction_preserves_outside_crop_and_reprojects_prior(self):
        wrapper = self.make_wrapper(shape=(140, 140, 400), spacing=(1.5, 1.5, 1.5))
        wrapper._native_boxes = False
        seen_priors = []
        torch = self.torch
        class Prompt:
            def __call__(self, points, boxes, masks):
                seen_priors.append(masks)
                return torch.zeros(1, 1, 3), torch.zeros(1)
            def get_dense_pe(self):
                return torch.zeros(1)
        def decoder(**kwargs):
            logits = torch.full((1, 1, 32, 32, 32), -10.)
            logits[:, :, 14:18, 14:18, 14:18] = 10.
            return logits, torch.zeros(1)
        wrapper.model.prompt_encoder = Prompt()
        wrapper.model.mask_decoder = decoder
        first, logits = wrapper.predict_mask(points=[(64, 64, 20)], labels=[1])
        self.assertGreater(first.sum(), 0)
        second, _ = wrapper.predict_mask(points=[(64, 64, 380)], labels=[1], mask_logits=logits)
        np.testing.assert_array_equal(second[:, :, :100], first[:, :, :100])
        self.assertGreater(second[:, :, 300:].sum(), 0)
        self.assertIsNotNone(seen_priors[-1])
        self.assertEqual(tuple(seen_priors[-1].shape), (1, 1, 32, 32, 32))


if __name__ == '__main__':
    unittest.main()
