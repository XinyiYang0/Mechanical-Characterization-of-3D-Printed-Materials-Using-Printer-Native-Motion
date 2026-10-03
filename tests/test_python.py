"""Synthetic interface and numerical checks; no experimental data are required."""

from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np
from numpy.testing import assert_allclose

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'elastic'), str(ROOT / 'fracture')]
from tracking import TrackingConfig, beam_pixels, root_coordinates, continuous_pixels, tip_and_thickness, track_video
from curvature import analyze_image, fit_centerline, legacy_curvature_adjustment


def beam_frame(bend=10):
    image = np.zeros((200, 220, 3), np.uint8)
    rows = np.arange(30, 191)
    center = 100 + np.rint(bend * ((190-rows)/160)**2).astype(int)
    polygon = np.concatenate([np.column_stack([center-5, rows]),
                              np.column_stack([center[::-1]+5, rows[::-1]])])
    cv2.fillPoly(image, [polygon.astype(np.int32)], (0, 120, 0))
    return image


class TrackingChecks(unittest.TestCase):
    def test_profiles_and_no_beam(self):
        for profile in ('legacy', 'component'):
            config = TrackingConfig(profile=profile)
            points = continuous_pixels(root_coordinates(beam_pixels(beam_frame(), config)))
            axial, transverse, thickness = tip_and_thickness(points, profile)
            self.assertEqual(axial, 160)
            self.assertLess(transverse, -5)
            self.assertGreater(thickness, 0)
            self.assertEqual(beam_pixels(np.zeros((200,220,3), np.uint8), config), [])

    def test_translation_invariance(self):
        pixels = beam_pixels(beam_frame(), TrackingConfig())
        translated = [(x+13,y+21) for x,y in pixels]
        assert_allclose(root_coordinates(pixels), root_coordinates(translated))

    def test_video_preserves_missing_frame_and_acquisition_time(self):
        with tempfile.TemporaryDirectory() as folder:
            video = Path(folder) / 'sample.avi'
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 25, (220,200))
            self.assertTrue(writer.isOpened())
            for i in range(6):
                writer.write(np.zeros((200,220,3), np.uint8) if i == 2 else beam_frame(i*2))
            writer.release()
            output = Path(folder)/'tracking.csv'
            result = track_video(video, output, acquisition_fps=1000, preview_every=2)
            self.assertEqual(len(result), 6)
            assert_allclose(result.time_s, np.arange(6)/1000)
            self.assertEqual(result.loc[2,'tracking_status'], 'no_detection')
            self.assertTrue(np.isnan(result.loc[2,'tip_displacement_px']))
            self.assertTrue((output.parent/'tracking_previews/frame_000000.png').exists())
            with self.assertRaises(FileExistsError):
                track_video(video, output, acquisition_fps=1000)

    def test_invalid_rate_and_roi(self):
        with self.assertRaises(ValueError):
            track_video('unused.mp4', 'unused.csv', acquisition_fps=0)
        with self.assertRaises(ValueError):
            beam_pixels(beam_frame(), TrackingConfig(roi=(0,0,999,200)))


class CurvatureChecks(unittest.TestCase):
    def test_known_polynomial_and_clamp_constraint(self):
        y = np.linspace(80, 400, 200)
        x = 30 + 0.001*(y-80)**2 + 1e-8*(y-80)**4
        fit = fit_centerline(y, x, clamp_slope=True)
        assert_allclose(fit(y), x, atol=1e-9)
        self.assertAlmostEqual(fit.deriv()(80), 0, places=10)

    def test_metric_curvature_of_parabola(self):
        y = np.linspace(0,0.075,300)
        coefficient = 3.0
        fit = fit_centerline(y, coefficient*y**2)
        measured = abs(fit.deriv(2)(y)) / (1+fit.deriv()(y)**2)**1.5
        expected = 2*coefficient / (1+(2*coefficient*y)**2)**1.5
        assert_allclose(measured, expected, rtol=1e-9)

    def test_straight_image_pipeline_and_raw_preservation(self):
        with tempfile.TemporaryDirectory() as folder:
            image = np.zeros((240,240,3), np.uint8)
            # A cropped straight strip has no end caps: this isolates curvature
            # differentiation from the source detector's known end-cap bias.
            cv2.rectangle(image, (100,0), (110,239), (0,255,255), -1)
            source = Path(folder)/'straight.png'
            cv2.imwrite(str(source),image)
            profile = analyze_image(source, Path(folder)/'out', beam_length_m=0.075,
                                    include_legacy_adjustment=True)
            self.assertEqual(len(profile),3000)
            self.assertAlmostEqual(profile.arc_length_m.iloc[-1],0.075, places=6)
            self.assertLess(profile.curvature_per_m.max(),1e-6)
            self.assertIn('legacy_adjusted_curvature_per_m',profile)

    def test_end_cap_bias_remains_visible_in_legacy_method(self):
        with tempfile.TemporaryDirectory() as folder:
            image = np.zeros((240,240,3), np.uint8)
            cv2.rectangle(image, (100,20), (110,220), (0,255,255), -1)
            source = Path(folder)/'capped_strip.png'
            cv2.imwrite(str(source),image)
            profile = analyze_image(source, Path(folder)/'out', beam_length_m=0.075)
            # The first maximum in a flat distance-transform plateau can shift
            # towards an edge near an end cap. Do not claim zero curvature here.
            self.assertTrue(np.isfinite(profile.curvature_per_m).all())
            self.assertGreater(profile.curvature_per_m.max(), 1e-3)

    def test_heuristic_is_separate_and_not_strict_monotonicity(self):
        raw = np.ones(300)
        raw[150:] = 1.005
        copy = raw.copy()
        adjusted = legacy_curvature_adjustment(raw)
        assert_allclose(raw,copy)
        self.assertGreater(adjusted[150], adjusted[149])
        raw[150] = 2
        self.assertAlmostEqual(legacy_curvature_adjustment(raw)[150],0.99)


if __name__ == '__main__':
    unittest.main(verbosity=2)
