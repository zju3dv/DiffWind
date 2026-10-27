"""CPU regression checks: python -m unittest discover -s tests -v."""
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from reconstruction.evaluation import score
from reconstruction.io import export_image, read_json, run_lock, save_array, write_json
from reconstruction.config import sha256
from reconstruction.optimizer import TransactionalAdam, optimize_window
from reconstruction.runner import committed_frames, evaluate_run


class ReconstructionTests(unittest.TestCase):
    def test_adam_matches_torch_and_rejects_without_advancing(self):
        initial = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        parameter = torch.tensor(initial, requires_grad=True)
        reference = torch.optim.Adam([parameter], lr=0.001)
        actual, x = TransactionalAdam(initial.shape), initial.copy()
        for gradient in (
            np.array([0.3, -0.1, 0.5], np.float32),
            np.array([-0.2, 0.7, 0.1], np.float32),
        ):
            previous = (actual.m.copy(), actual.v.copy(), actual.t)
            for lr in (0.1, 0.01):
                actual.propose(gradient, lr)  # rejected candidates
            np.testing.assert_array_equal(actual.m, previous[0])
            np.testing.assert_array_equal(actual.v, previous[1])
            self.assertEqual(actual.t, previous[2])
            step, candidate = actual.propose(gradient, 0.001)
            x += step
            actual.commit(candidate)
            parameter.grad = torch.tensor(gradient)
            reference.step()
            reference.zero_grad()
        np.testing.assert_allclose(x, parameter.detach().numpy(), atol=2e-7, rtol=1e-6)

    def test_backtracking_decreases_actual_objective(self):
        class Quadratic:
            image_domain = "clipped"

            def evaluate(self, state, force, frame, gradient=False, **kwargs):
                delta = force - np.array([0.4, -0.7], np.float32)
                loss = float(np.sum(delta**2))
                metric = dict(
                    loss=loss,
                    train_psnr=-10 * math.log10(max(loss, 1e-30)),
                    state_diagnostics={},
                )
                return metric, 2 * delta if gradient else None, force.copy()

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "window"
            _, _, result = optimize_window(
                Quadratic(),
                None,
                np.zeros(2, np.float32),
                1,
                output,
                initial_lr=8,
                min_updates=3,
                max_updates=8,
            )
            self.assertLess(result["best"]["loss"], result["initial"]["loss"])
            rows = [
                json.loads(line)
                for line in (output / "training.jsonl").read_text().splitlines()
            ]
            self.assertTrue(any(len(row["attempts"]) > 1 for row in rows))
            for row in rows:
                self.assertLessEqual(row["loss_after"], row["loss_before"])

    def test_png_metric_averages_per_image_and_excludes_frame_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = []
            for view in range(4):
                label = "train_view%d" % view
                for branch in ("gt", "render"):
                    (root / branch / label).mkdir(parents=True)
                for frame, offset in ((0, 255), (1, 1), (2, 8)):
                    truth = np.full((2, 3, 3), 100, np.uint8)
                    actual = np.full_like(truth, 100 + offset if frame else 0)
                    for branch, image in (("gt", truth), ("render", actual)):
                        Image.fromarray(image).save(
                            str(root / branch / label / ("%05d.png" % frame))
                        )
                    if frame:
                        error = np.square(
                            truth.astype(np.float32) / 255
                            - actual.astype(np.float32) / 255
                        )
                        expected.append(
                            -10 * math.log10(float(error.mean(dtype=np.float64)))
                        )
            result = score(
                root / "gt",
                root / "render",
                root / "metrics",
                "potted_plant",
                frames=2,
                split="train",
            )
            self.assertEqual(result["image_count"], 8)
            self.assertAlmostEqual(result["psnr"], float(np.mean(expected)), places=12)
            with self.assertRaises(ValueError):
                score(
                    root / "gt",
                    root / "render",
                    root / "test",
                    "potted_plant",
                    frames=2,
                )

    def test_current_point_adjoint_failure_rolls_back_and_recovers(self):
        class TransientFailure:
            image_domain = "clipped"
            adjoints = 0

            def evaluate(self, state, force, frame, gradient=False, **kwargs):
                if gradient:
                    self.adjoints += 1
                    if self.adjoints == 2:
                        raise FloatingPointError(
                            "simulated CFL failure on recomputation"
                        )
                delta = force - 1.0
                loss = float(np.sum(delta**2))
                return (
                    dict(
                        loss=loss,
                        train_psnr=-10 * math.log10(max(loss, 1e-30)),
                        state_diagnostics={},
                    ),
                    2 * delta if gradient else None,
                    force.copy(),
                )

        with tempfile.TemporaryDirectory() as directory:
            engine = TransientFailure()
            force, state, result = optimize_window(
                engine,
                None,
                np.zeros(1, np.float32),
                1,
                Path(directory) / "window",
                initial_lr=0.05,
                min_updates=4,
                max_updates=4,
            )
            self.assertEqual(result["numerical_recoveries"], 1)
            self.assertLess(result["best"]["loss"], result["initial"]["loss"])
            np.testing.assert_array_equal(force, state)
            self.assertGreater(result["accepted_updates"], 0)

    def test_persistent_invalid_adjoint_is_not_reported_as_success(self):
        class InvalidAdjoint:
            image_domain = "clipped"

            def evaluate(self, state, force, frame, gradient=False, **kwargs):
                if gradient:
                    raise FloatingPointError("persistent nonfinite adjoint")
                return dict(loss=1.0, train_psnr=0.0, state_diagnostics={}), None, force

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FloatingPointError, "No valid adjoint"):
                optimize_window(
                    InvalidAdjoint(),
                    None,
                    np.zeros(1, np.float32),
                    1,
                    Path(directory) / "window",
                    min_updates=1,
                    max_updates=1,
                )

    def test_checkpoint_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_array(root / "force.npy", np.zeros(2, np.float32))
            save_array(root / "state.npz", dict(x=np.zeros(3, np.float32)))
            record = dict(
                frame=1,
                force="force.npy",
                state="state.npz",
                force_sha256=sha256(root / "force.npy"),
                state_sha256=sha256(root / "state.npz"),
            )
            write_json(root / "frames/00001.json", record)
            self.assertEqual(len(committed_frames(root)), 1)
            save_array(root / "force.npy", np.ones(2, np.float32))
            with self.assertRaises(ValueError):
                committed_frames(root)

    def test_frozen_evaluation_scores_continuous_replay(self):
        class Engine:
            def __init__(self, gt):
                self.spec = dict(gt=gt)
                self.initial = dict(x=np.array([0.0], np.float32))
                self.incoming = []

            def forward(self, state, force):
                self.incoming.append(float(state["x"][0]))
                return dict(x=state["x"] + force)

            def render(self, state, view):
                return torch.full((3, 2, 3), float(state["x"][0]) / 255), None, None

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gt = root / "gt"
            records = []
            for frame in range(1, 49):
                force = "forces/%05d.npy" % frame
                state = "states/%05d.npz" % frame
                save_array(root / force, np.ones(1, np.float32))
                save_array(root / state, {
                    k: np.array([-1000.0], np.float32) for k in ("x", "v", "F", "C")
                })
                record = dict(frame=frame, force=force, state=state,
                              force_sha256=sha256(root / force),
                              state_sha256=sha256(root / state))
                records.append(record)
                write_json(root / "frames" / ("%05d.json" % frame), record)
                for view in range(4):
                    path = gt / ("test_view%d" % view) / ("%05d.png" % frame)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(np.full((2, 3, 3), 100, np.uint8)).save(str(path))
            run = dict(scene="dress", frames=48)
            write_json(root / "run.json", run)
            write_json(root / "frozen.json", dict(
                frames=records, run_sha256=sha256(root / "run.json")))
            engine = Engine(gt)
            with patch("reconstruction.runner.verify_run"), patch("builtins.print"):
                result = evaluate_run(engine, root, run, "test-binary")
            self.assertEqual(engine.incoming, list(range(48)))
            self.assertEqual(result["image_count"], 192)
            expected = []
            for frame in range(1, 49):
                difference = np.float32(100) / np.float32(255) - np.float32(frame) / np.float32(255)
                expected.append(-10 * math.log10(float(difference ** np.float32(2))))
            self.assertAlmostEqual(result["psnr"], float(np.mean(expected)), places=10)
            with Image.open(root / "render/test_view0/00048.png") as image:
                np.testing.assert_array_equal(np.asarray(image), np.full((2, 3, 3), 48, np.uint8))
            diagnostics = read_json(root / "replay.json")
            self.assertEqual(len(diagnostics), 48)
            self.assertEqual(diagnostics[-1]["max_abs"]["x"], 1048.0)
            self.assertEqual(read_json(root / "status.json")["phase"], "complete")

    def test_export_rejects_nonfinite_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "render.png"
            for value in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(value=value):
                    with self.assertRaises(FloatingPointError):
                        export_image(torch.full((3, 2, 3), value), path)
                    self.assertFalse(path.exists())

    def test_duplicate_run_lock_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with run_lock(Path(directory)):
                with self.assertRaises(RuntimeError):
                    with run_lock(Path(directory)):
                        pass


if __name__ == "__main__":
    unittest.main()
