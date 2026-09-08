import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import evaluate_radaz_paired_pilot as evaluator
from run_radaz_paired_pilot import checkpoint_epoch, process_identity, previous_child_is_alive
from prepare_radaz_paired_pilot import source_manifest


class PilotTests(unittest.TestCase):
    def test_source_manifest_supplies_all_loader_splits_without_mutating_parent(self):
        parent = dict(normalization=dict(low=[-1], high=[1]), cases=[
            dict(case_key="source", role="source", splits=["train", "val"]),
            dict(case_key="held", role="holdout", splits=["test"])])
        result = source_manifest(parent)
        self.assertEqual([c["case_key"] for c in result["cases"]], ["source"])
        self.assertEqual(result["cases"][0]["splits"], ["train", "val", "test"])
        self.assertEqual(parent["cases"][0]["splits"], ["train", "val"])
        self.assertEqual(result["normalization"], parent["normalization"])

    def test_last_checkpoint_requires_valid_epoch_and_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.ckpt"
            self.assertIsNone(checkpoint_epoch(path))
            for epoch in (0, 58, 59):
                torch.save(dict(epoch=epoch, state_dict={}, optimizer_states=[{}]), path)
                self.assertEqual(checkpoint_epoch(path), epoch)
            for state in (dict(epoch=60, state_dict={}, optimizer_states=[{}]),
                          dict(epoch=59, state_dict={}, optimizer_states=[])):
                torch.save(state, path)
                with self.assertRaises(RuntimeError):
                    checkpoint_epoch(path)

    def test_process_identity_guards_pid_reuse(self):
        identity = process_identity(os.getpid())
        self.assertIsInstance(identity, int)
        self.assertTrue(previous_child_is_alive(dict(child_pid=os.getpid(), child_create_time=identity)))
        self.assertFalse(previous_child_is_alive(dict(child_pid=os.getpid(), child_create_time=identity+1)))

    def test_baseline_must_have_matching_targets(self):
        truth = np.arange(24.).reshape(2, 3, 4)
        with tempfile.TemporaryDirectory() as directory, patch.object(evaluator, "BASELINES", Path(directory)):
            path = Path(directory) / "source_test_example_gamma_predictions.npz"
            np.savez(path, truth=truth, ar10_validation_selected=truth+1)
            np.testing.assert_equal(evaluator.matched_baseline("gamma", "source_test", "example", truth), truth+1)
            with self.assertRaises(AssertionError):
                evaluator.matched_baseline("gamma", "source_test", "example", truth[:, ::-1])


if __name__ == "__main__":
    unittest.main()
