"""Check the public workflow using generated signals and saved aggregate counts."""

import csv
from pathlib import Path
import unittest

import numpy as np
import torch

from examples.synthetic_pipeline import run_demo
from src.evaluation import metrics
from src.features import feature_names, features48, lagged_gram
from src.models import ScaleCNN, make_cnn_gru
from src.preprocessing import Settings, Stream, interpolate, prepare_pair

ROOT = Path(__file__).resolve().parents[1]


class SignalTests(unittest.TestCase):
    def test_forbidden_gap_is_nan(self):
        stream = Stream(np.array([0, 20, 400, 420]), np.ones((4, 3)), np.arange(4), {})
        result = interpolate(stream, np.arange(0, 421, 20), Settings(bracket_ms=200))
        self.assertFalse(result["valid"][2:20].any())
        self.assertTrue(np.isnan(result["xyz"][2:20]).all())

    def test_no_extrapolation(self):
        stream = Stream(np.array([20, 40]), np.ones((2, 3)), np.arange(2), {})
        result = interpolate(stream, np.array([0, 20, 30, 40, 60]), Settings())
        np.testing.assert_array_equal(result["valid"], [False, True, True, True, False])
        self.assertEqual(result["available_after_ms"][2], 40)

    def test_same_rotation_preserves_only_the_lag_block(self):
        windows = np.random.default_rng(2).normal(size=(2, 125, 6)).astype(np.float32)
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        rotated = windows.copy()
        rotated[:, :, :3] = windows[:, :, :3] @ rotation.T
        rotated[:, :, 3:] = windows[:, :, 3:] @ rotation.T
        np.testing.assert_allclose(lagged_gram(windows), lagged_gram(rotated), atol=1e-14)
        self.assertFalse(np.array_equal(features48(windows), features48(rotated)))
        self.assertEqual(len(set(feature_names(60))), 60)

    def test_recording_identity_must_match(self):
        metadata = dict(subject=1, site="waist", activity="walking", trial=1, key="toy")
        time = np.array([0, 20])
        first = Stream(time, np.ones((2, 3)), np.arange(2), metadata)
        second = Stream(time, np.ones((2, 3)), np.arange(2), dict(metadata, activity="running"))
        with self.assertRaisesRegex(ValueError, "different recording identities"):
            prepare_pair(first, second)


class EvaluationTests(unittest.TestCase):
    def test_unavailable_is_a_miss(self):
        result = metrics(np.arange(7), [0, 1, 2, 3, 4, 5, 7])
        self.assertEqual(result["confusion"].shape, (7, 8))
        self.assertAlmostEqual(result["accuracy"], 6 / 7)
        self.assertAlmostEqual(result["macro_f1"], 6 / 7)

    def test_demo_runs_with_missing_predictions(self):
        shape, scheduled, unavailable, result = run_demo()
        self.assertEqual(shape[1], 60)
        self.assertGreater(unavailable, 0)
        self.assertEqual(result["confusion"].sum(), scheduled)
        self.assertEqual(result["confusion"][:, 7].sum(), unavailable)

    def test_d60_confusion_matches_eligible_count(self):
        with (ROOT / "results/model_comparison.csv").open(newline="") as data:
            d60 = next(row for row in csv.DictReader(data) if row["model"] == "d60")
        confusion = np.loadtxt(ROOT / "results/d60_confusion.csv", delimiter=",", dtype=int)
        self.assertEqual(confusion.shape, (7, 7))
        self.assertEqual(confusion.sum(), int(d60["eligible_queries"]))


class ModelTests(unittest.TestCase):
    def test_network_capacity_and_output(self):
        values = torch.zeros(2, 125, 6)
        with torch.inference_mode():
            for model, parameters in (
                (make_cnn_gru(), 6215),
                (ScaleCNN("control"), 3983),
                (ScaleCNN("candidate"), 3983),
            ):
                self.assertEqual(sum(p.numel() for p in model.parameters()), parameters)
                self.assertEqual(model(values).shape, (2, 7))
