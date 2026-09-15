"""Model and data tests for the released niche components."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from cellworldmodel.data.niche_context import NicheTransitionData, neighborhood_composition, shared_composition
from cellworldmodel.model.niche_conditioning import NicheConditionedTransition, NichePotentialCorrection
from cellworldmodel.model.waddington_dit_1d import WaddingtonDiT1D


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()

class NicheModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.base = WaddingtonDiT1D(
            dim=8, hidden_dim=32, depth=1, num_heads=4,
            num_register_tokens=2, time_emb_dim=32, curl_rank=2,
        ).eval()
        self.z = torch.randn(4, 8)
        self.delta = torch.tensor([0., 1., 2., 3.])
        self.epsilon = torch.randn(4, 2, 8)
        self.context = torch.tensor([[1., 0., 0.], [0., 1., 0.],
                                     [.5, .5, 0.], [0., 0., 1.]])
        self.original_state = state_digest(self.base)
        self.before = self.base(self.z, self.delta, self.epsilon)
        self.correction = NichePotentialCorrection(
            8, 3, delta_scale=3., programs=2, hidden_dim=8,
        )
        self.model = NicheConditionedTransition(self.base, self.correction).eval()

    def test_disabled_and_null_skip_correction_exactly(self):
        def must_not_run(*args):
            raise AssertionError("disabled correction was evaluated")
        self.correction.forward = must_not_run
        self.assertTrue(torch.equal(self.before, self.model(self.z, self.delta, self.epsilon)))
        self.model.enabled = False
        self.assertTrue(torch.equal(self.before, self.model(
            self.z, self.delta, self.epsilon, self.context)))
        self.assertEqual(self.original_state, state_digest(self.base))

    def test_zero_initialization_and_zero_duration_identity(self):
        with torch.no_grad():
            prediction = self.model(self.z, self.delta, self.epsilon, self.context)
        self.assertTrue(torch.equal(self.before, prediction))
        with torch.no_grad():
            self.correction.coefficients[-1].weight.fill_(.1)
        prediction = self.model(self.z, self.delta, self.epsilon, self.context)
        self.assertTrue(torch.equal(prediction[0], self.z[0].expand_as(prediction[0])))

    def test_condition_changes_prediction_and_basis_gets_gradient(self):
        self.model.train()
        self.assertFalse(self.base.training)
        optimizer = torch.optim.AdamW(self.correction.parameters(), lr=.01)
        # This is an autograd check, not an experimental training objective.
        for _ in range(2):
            optimizer.zero_grad()
            prediction = self.model(self.z, self.delta, self.epsilon, self.context)
            loss = prediction.square().mean()
            loss.backward()
            self.assertGreater(self.correction.coefficients[-1].weight.grad.abs().sum(), 0)
            optimizer.step()
        self.assertGreater(sum(p.grad.abs().sum() for p in self.correction.basis.parameters()
                               if p.grad is not None), 0)
        self.model.eval()
        actual = self.model(self.z, self.delta, self.epsilon, self.context)
        replaced = self.model(self.z, self.delta, self.epsilon, self.context.roll(1, 0))
        self.assertFalse(torch.equal(actual, replaced))
        self.assertTrue(all(p.grad is None for p in self.base.parameters()))
        self.assertEqual(self.original_state, state_digest(self.base))
        self.model.enabled = False
        self.assertTrue(torch.equal(self.before, self.model(
            self.z, self.delta, self.epsilon, self.context)))

    def test_invalid_context_fails_loud(self):
        with self.assertRaisesRegex(ValueError, "one vector"):
            self.model(self.z, self.delta, self.epsilon, self.context[:1])
        with self.assertRaisesRegex(ValueError, "finite"):
            self.model(self.z, self.delta, self.epsilon, self.context * float("nan"))

class NeighborhoodTests(unittest.TestCase):
    def setUp(self):
        self.xy = np.array([[0., 0.], [1., 0.], [4., 0.], [4., 3.]])
        self.labels = np.array(["a", "b", "b", "new"])

    def test_excludes_self_and_keeps_unknown_category(self):
        composition = neighborhood_composition(self.xy, self.labels, ["a", "b"], neighbors=1)
        np.testing.assert_array_equal(composition[0], [0., 1., 0.])
        np.testing.assert_array_equal(composition[1], [1., 0., 0.])
        np.testing.assert_array_equal(composition[3], [0., 1., 0.])
        all_neighbors = neighborhood_composition(self.xy, self.labels, ["a", "b"], neighbors=10)
        self.assertAlmostEqual(float(all_neighbors[0, 2]), 1 / 3, places=6)
        np.testing.assert_allclose(all_neighbors.sum(1), 1)

    def test_translation_rotation_and_scale_invariance(self):
        before = neighborhood_composition(self.xy, self.labels, ["a", "b"], neighbors=2)
        rotation = np.array([[0., -1.], [1., 0.]])
        transformed = 3 * (self.xy @ rotation) + np.array([31., -17.])
        after = neighborhood_composition(transformed, self.labels, ["a", "b"], neighbors=2)
        np.testing.assert_array_equal(before, after)

    def test_ties_and_blocks_are_deterministic(self):
        xy = np.array([[0., 0.], [-1., 0.], [1., 0.]])
        labels = np.array(["a", "b", "a"])
        before = neighborhood_composition(xy, labels, ["a", "b"], neighbors=1, block_size=1)
        after = neighborhood_composition(xy, labels, ["a", "b"], neighbors=1, block_size=4)
        np.testing.assert_array_equal(before, after)
        np.testing.assert_array_equal(before[0], [0., 1., 0.])

    def test_shared_control_uses_training_composition(self):
        np.testing.assert_allclose(shared_composition(np.array(["a", "a", "b"]), ["a", "b"]),
                                   [2 / 3, 1 / 3, 0])

class NicheDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        rows = []
        for time in (1., 2.):
            for split in ("train", "val", "test"):
                for index, label in enumerate(("a", "a", "b")):
                    rows.append({"cell_id": len(rows), "time": time, "split": split,
                                 "spatial_x": index, "spatial_y": index * index,
                                 "cell_type": label, "slice": str(time)})
        self.meta = pd.DataFrame(rows)
        self.z = np.random.default_rng(42).normal(size=(len(rows), 8)).astype(np.float32)
        self.meta.to_csv(self.root / "metadata.tsv", sep="\t", index=False)
        np.savez(self.root / "representations.npz", scvi128=self.z)
        train = self.z[self.meta["split"] == "train"]
        np.savez(self.root / "metric_standardization.npz", mean=train.mean(0), std=train.std(0))
        (self.root / "split_manifest.json").write_text(json.dumps(
            {"encoder": {"vae_sha256": "a" * 64, "gene_vocab_sha256": "b" * 64}}))

    def load(self):
        return NicheTransitionData(self.root, source_time=1., target_time=2.,
                                   label_column="cell_type", neighbors=1)

    def test_target_coordinates_and_annotations_cannot_enter_source_context(self):
        before = self.load()
        target = self.meta["time"] == 2.
        self.meta.loc[target, "spatial_x"] = 10000
        self.meta.loc[target, "cell_type"] = "future_only"
        self.meta.to_csv(self.root / "metadata.tsv", sep="\t", index=False)
        after = self.load()
        np.testing.assert_array_equal(before.context, after.context)
        np.testing.assert_array_equal(before.shared, after.shared)
        self.assertEqual(before.categories, after.categories)

    def test_validation_categories_do_not_expand_training_vocabulary(self):
        before = self.load()
        validation_source = (self.meta["time"] == 1.) & (self.meta["split"] == "val")
        self.meta.loc[validation_source, "cell_type"] = "unseen"
        self.meta.to_csv(self.root / "metadata.tsv", sep="\t", index=False)
        after = self.load()
        self.assertEqual(before.categories, after.categories)
        np.testing.assert_array_equal(before.shared, after.shared)
        np.testing.assert_array_equal(before.context[before.source_ids["train"]],
                                      after.context[after.source_ids["train"]])
        np.testing.assert_array_equal(after.context[after.source_ids["val"], -1], 1.)

    def test_stale_statistics_and_missing_encoder_are_rejected(self):
        np.savez(self.root / "metric_standardization.npz", mean=np.zeros(8), std=np.ones(8))
        with self.assertRaisesRegex(ValueError, "mean"):
            self.load()
        (self.root / "split_manifest.json").write_text('{"encoder": null}')
        with self.assertRaisesRegex(ValueError, "encoder"):
            self.load()

    def test_independent_source_coordinate_frames_cannot_mix(self):
        self.meta.loc[0, "slice"] = "another_section"
        self.meta.to_csv(self.root / "metadata.tsv", sep="\t", index=False)
        with self.assertRaisesRegex(ValueError, "coordinate frame"):
            self.load()
