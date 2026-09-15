from __future__ import annotations

import copy
import unittest

import torch
from torch import nn

from cellworldmodel.model.spatial_state_heads import (
    EquivariantPositionHead,
    LogMassHead,
    SpatialStateHeads,
)
from cellworldmodel.model.waddington_dit_1d import WaddingtonDiT1D


class SpatialHeadTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(817)
        self.z = torch.randn(5, 3, dtype=torch.float64)
        self.x = torch.randn(5, 2, dtype=torch.float64)
        self.neighbor_z = torch.randn(5, 4, 3, dtype=torch.float64)
        self.relative_x = torch.randn(5, 4, 2, dtype=torch.float64)
        self.delta = torch.tensor([0., .4, 1., 2., 3.], dtype=torch.float64)
        self.alpha = -torch.expm1(-self.delta / 2)
        self.log_mass = torch.randn(5, dtype=torch.float64)
        self.context = torch.randn(5, 7, dtype=torch.float64)
        self.heads = SpatialStateHeads(
            3, 7, delta_scale=3., distance_scale=2., hidden_dim=16, init_seed=19,
        ).double()

    def inputs(self):
        return {"z": self.z, "delta": self.delta, "alpha": self.alpha,
                "x": self.x, "neighbor_z": self.neighbor_z,
                "relative_x": self.relative_x, "log_mass": self.log_mass,
                "context": self.context}

    def activate(self):
        with torch.no_grad():
            self.heads.position_head.messages[-1].weight.fill_(.3)
            self.heads.mass_head.growth[-1].weight.fill_(.2)
            self.heads.mass_head.growth[-1].bias.fill_(.1)

    def test_zero_initialization_is_exact_identity(self):
        before = {name: value.clone() for name, value in self.inputs().items()}
        x, mass = self.heads(**self.inputs())
        self.assertTrue(torch.equal(x, self.x))
        self.assertTrue(torch.equal(mass, self.log_mass))
        for name, value in self.inputs().items():
            self.assertTrue(torch.equal(before[name], value), name)

    def test_first_and_later_steps_have_effective_gradients(self):
        optimizer = torch.optim.SGD(self.heads.parameters(), lr=.3)
        target_x = self.x + self.relative_x[:, 0]
        target_mass = self.log_mass + .7
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            x, mass = self.heads(**self.inputs())
            loss = (x - target_x).square().mean() + (mass - target_mass).square().mean()
            loss.backward()
            for network in (self.heads.position_head.messages, self.heads.mass_head.growth):
                self.assertGreater(network[-1].weight.grad.abs().sum().item(), 0)
                if step == 0:
                    self.assertEqual(network[0].weight.grad.abs().sum().item(), 0)
                else:
                    self.assertGreater(network[0].weight.grad.abs().sum().item(), 0)
            optimizer.step()
        x, mass = self.heads(**self.inputs())
        self.assertFalse(torch.equal(x, self.x))
        self.assertFalse(torch.equal(mass, self.log_mass))
        self.heads.enabled = False
        x, mass = self.heads(**self.inputs())
        self.assertIs(x, self.x)
        self.assertIs(mass, self.log_mass)

    def test_zero_delta_stays_exact_after_training_and_wrong_alpha_fails(self):
        self.activate()
        x, mass = self.heads(**self.inputs())
        self.assertTrue(torch.equal(x[0], self.x[0]))
        self.assertTrue(torch.equal(mass[0], self.log_mass[0]))
        args = self.inputs()
        args["alpha"] = torch.ones_like(self.alpha)
        with self.assertRaisesRegex(ValueError, "alpha must be zero"):
            self.heads(**args)

    def test_disabled_nonzero_heads_skip_both_branch_hooks(self):
        self.activate()
        def must_not_run(*args):
            raise AssertionError("disabled branch ran")
        self.heads.position_head.register_forward_pre_hook(must_not_run)
        self.heads.mass_head.register_forward_pre_hook(must_not_run)
        self.heads.enabled = False
        args = self.inputs()
        args.update(z=None, delta=None, alpha=None, neighbor_z=None, context=None)
        x, mass = self.heads(**args)
        self.assertIs(x, self.x)
        self.assertIs(mass, self.log_mass)

    def test_missing_modalities_skip_their_hooks_independently(self):
        self.activate()
        position_calls, mass_calls = [], []
        self.heads.position_head.register_forward_pre_hook(lambda *args: position_calls.append(1))
        self.heads.mass_head.register_forward_pre_hook(lambda *args: mass_calls.append(1))
        args = self.inputs()
        args.update(x=None, neighbor_z=None, relative_x=None)
        x, mass = self.heads(**args)
        self.assertIsNone(x)
        self.assertFalse(torch.equal(mass, self.log_mass))
        self.assertEqual((len(position_calls), len(mass_calls)), (0, 1))
        args = self.inputs()
        args.update(log_mass=None, context=None)
        x, mass = self.heads(**args)
        self.assertIsNone(mass)
        self.assertFalse(torch.equal(x, self.x))
        self.assertEqual((len(position_calls), len(mass_calls)), (1, 1))
        self.assertEqual(self.heads(None, None, None), (None, None))
        self.assertEqual((len(position_calls), len(mass_calls)), (1, 1))

    def test_per_branch_disabled_returns_original_object(self):
        self.activate()
        self.heads.position_enabled = False
        x, mass = self.heads(**self.inputs())
        self.assertIs(x, self.x)
        self.assertFalse(torch.equal(mass, self.log_mass))
        self.heads.position_enabled, self.heads.mass_enabled = True, False
        x, mass = self.heads(**self.inputs())
        self.assertFalse(torch.equal(x, self.x))
        self.assertIs(mass, self.log_mass)

    def test_translation_rotation_and_reflection_equivariance(self):
        self.activate()
        x, mass = self.heads(**self.inputs())
        angle = torch.tensor(.731, dtype=self.z.dtype)
        c, s = torch.cos(angle), torch.sin(angle)
        rotation = torch.stack((torch.stack((c, -s)), torch.stack((s, c))))
        for transform in (rotation, torch.diag(torch.tensor([-1., 1.], dtype=self.z.dtype))):
            args = self.inputs()
            translation = torch.tensor([11., -7.], dtype=self.z.dtype)
            args["x"] = self.x @ transform + translation
            args["relative_x"] = self.relative_x @ transform
            transformed_x, transformed_mass = self.heads(**args)
            torch.testing.assert_close(transformed_x, x @ transform + translation, atol=1e-12, rtol=1e-12)
            self.assertTrue(torch.equal(mass, transformed_mass))

    def test_batch_and_neighbor_permutations(self):
        self.activate()
        x, mass = self.heads(**self.inputs())
        order = torch.tensor([4, 1, 0, 3, 2])
        args = {name: value[order] for name, value in self.inputs().items()}
        permuted_x, permuted_mass = self.heads(**args)
        torch.testing.assert_close(permuted_x, x[order], atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(permuted_mass, mass[order], atol=1e-12, rtol=1e-12)
        args = self.inputs()
        order = torch.tensor([2, 0, 3, 1])
        args["neighbor_z"] = self.neighbor_z[:, order]
        args["relative_x"] = self.relative_x[:, order]
        permuted_x, permuted_mass = self.heads(**args)
        torch.testing.assert_close(permuted_x, x, atol=1e-12, rtol=1e-12)
        self.assertTrue(torch.equal(mass, permuted_mass))

    def test_null_subtracts_neighbor_independent_additive_terms(self):
        self.activate()
        with torch.no_grad():
            self.heads.position_head.messages[0].weight[:, 3:6].zero_()
        x, _ = self.heads(**self.inputs())
        self.assertTrue(torch.equal(x, self.x))

    def test_zero_neighbor_genes_are_exact_null(self):
        self.activate()
        args = self.inputs()
        args["neighbor_z"] = torch.zeros_like(self.neighbor_z)
        x, _ = self.heads(**args)
        self.assertTrue(torch.equal(x, self.x))

    def test_sum_not_mean_and_no_coordinate_components_enter_scalar(self):
        self.activate()
        args = self.inputs()
        x, _ = self.heads(**args)
        args["neighbor_z"] = self.neighbor_z.repeat(1, 2, 1)
        args["relative_x"] = self.relative_x.repeat(1, 2, 1)
        doubled_x, _ = self.heads(**args)
        torch.testing.assert_close(doubled_x - self.x, 2 * (x - self.x), atol=1e-12, rtol=1e-12)
        actual_inputs = []
        hook = self.heads.position_head.messages.register_forward_pre_hook(
            lambda module, inputs: actual_inputs.append(inputs[0].detach()))
        self.heads(**self.inputs())
        hook.remove()
        self.assertEqual(len(actual_inputs), 2)
        self.assertEqual(actual_inputs[0].shape, (5, 4, 8))
        torch.testing.assert_close(actual_inputs[0][..., -2],
                                   torch.linalg.vector_norm(self.relative_x / 2, dim=-1))
        torch.testing.assert_close(actual_inputs[0][..., -1], self.delta[:, None].expand(-1, 4) / 3)
        self.assertTrue(torch.equal(actual_inputs[0][..., :3], actual_inputs[1][..., :3]))
        self.assertEqual(actual_inputs[1][..., 3:6].count_nonzero().item(), 0)

    def test_empty_neighbors_and_masked_padding(self):
        self.activate()
        args = self.inputs()
        args.update(neighbor_z=self.neighbor_z[:, :0], relative_x=self.relative_x[:, :0])
        calls = []
        hook = self.heads.position_head.messages.register_forward_pre_hook(lambda *args: calls.append(1))
        x, _ = self.heads(**args)
        self.assertIs(x, self.x)
        self.assertEqual(calls, [])
        hook.remove()
        args = self.inputs()
        args["neighbor_mask"] = torch.zeros(5, 4, dtype=torch.bool)
        x, _ = self.heads(**args)
        self.assertTrue(torch.equal(x, self.x))
        args["neighbor_mask"][:, :2] = True
        x, _ = self.heads(**args)
        args.update(neighbor_z=self.neighbor_z[:, :2], relative_x=self.relative_x[:, :2], neighbor_mask=None)
        cropped_x, _ = self.heads(**args)
        torch.testing.assert_close(cropped_x, x, atol=1e-12, rtol=1e-12)

    def test_empty_batch(self):
        args = {name: value[:0] for name, value in self.inputs().items()}
        x, mass = self.heads(**args)
        self.assertEqual(x.shape, (0, 2))
        self.assertEqual(mass.shape, (0,))

    def test_context_is_optional_and_configurable(self):
        self.activate()
        args = self.inputs()
        args["context"] = None
        _, absent = self.heads(**args)
        args["context"] = torch.zeros_like(self.context)
        _, explicit_zero = self.heads(**args)
        self.assertTrue(torch.equal(absent, explicit_zero))
        _, present = self.heads(**self.inputs())
        self.assertFalse(torch.equal(present, absent))
        no_context = LogMassHead(3, delta_scale=3.).double()
        self.assertTrue(torch.equal(no_context(self.z, self.log_mass, self.delta, self.alpha), self.log_mass))

    def test_shapes_dtypes_devices_and_nonfinite_inputs_are_rejected(self):
        bad_values = {"z": self.z[:, :2], "delta": self.delta[:, None],
                      "alpha": self.alpha.float(), "x": torch.zeros(5, 3, dtype=self.z.dtype),
                      "neighbor_z": None, "relative_x": self.relative_x[:, :2],
                      "log_mass": self.log_mass[:, None], "context": self.context[:, :4],
                      "neighbor_mask": torch.ones(5, 4, dtype=torch.float64)}
        for name, value in bad_values.items():
            with self.subTest(name=name):
                args = self.inputs()
                args[name] = value
                with self.assertRaises(ValueError):
                    self.heads(**args)
        for name, value in self.inputs().items():
            for nonfinite in (float("nan"), float("inf")):
                with self.subTest(name=name, nonfinite=nonfinite):
                    args = self.inputs()
                    args[name] = torch.full_like(value, nonfinite)
                    with self.assertRaisesRegex(ValueError, "finite"):
                        self.heads(**args)
        args = self.inputs()
        args["context"] = torch.empty_like(self.context, device="meta")
        with self.assertRaisesRegex(ValueError, "dtype and device"):
            self.heads(**args)
        self.heads.float()
        with self.assertRaisesRegex(ValueError, "head parameters"):
            self.heads(**self.inputs())

    def test_nonfinite_padding_and_parameter_outputs_are_rejected(self):
        args = self.inputs()
        args["neighbor_z"] = self.neighbor_z.clone()
        args["neighbor_z"][:, -1] = float("nan")
        args["neighbor_mask"] = torch.zeros(5, 4, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "finite, including masked padding"):
            self.heads(**args)
        with torch.no_grad():
            self.heads.mass_head.growth[-1].weight.fill_(float("inf"))
        with self.assertRaisesRegex(ValueError, "log_mass update is nonfinite"):
            self.heads(**self.inputs())

    def test_all_masked_finite_padding_overflow_is_rejected_before_backward(self):
        head = EquivariantPositionHead(3, delta_scale=1., distance_scale=1., hidden_dim=4)
        with torch.no_grad():
            for layer in (head.messages[0], head.messages[2], head.messages[4]):
                layer.weight.fill_(4.)
                if layer.bias is not None:
                    layer.bias.zero_()
        z, x = torch.zeros(1, 3), torch.zeros(1, 2)
        neighbors = torch.full((1, 1, 3), torch.finfo(torch.float32).max / 4)
        self.assertTrue(torch.isfinite(neighbors).all())
        self.assertTrue(all(torch.isfinite(p).all() for p in head.parameters()))
        with self.assertRaisesRegex(ValueError, "position messages are nonfinite before masking"):
            head(z, x, neighbors, torch.ones(1, 1, 2), torch.ones(1), torch.ones(1),
                 torch.zeros(1, 1, dtype=torch.bool))
        self.assertTrue(all(p.grad is None for p in head.parameters()))

    def test_all_masked_nonfinite_position_parameters_are_rejected(self):
        head = EquivariantPositionHead(3, delta_scale=1., distance_scale=1., hidden_dim=4)
        with torch.no_grad():
            head.messages[-1].weight.fill_(float("inf"))
        with self.assertRaisesRegex(ValueError, "position messages are nonfinite before masking"):
            head(torch.zeros(1, 3), torch.zeros(1, 2), torch.ones(1, 1, 3),
                 torch.ones(1, 1, 2), torch.ones(1), torch.ones(1), torch.zeros(1, 1, dtype=torch.bool))
        self.assertTrue(all(p.grad is None for p in head.parameters()))

    def test_invalid_constructor_configuration(self):
        base = {"gene_dim": 3, "context_dim": 7, "delta_scale": 3., "distance_scale": 2.}
        for name, values in (("gene_dim", (0, -1, 2.5, True)), ("context_dim", (-1, 1.5)),
                             ("hidden_dim", (0, -1)), ("init_seed", (-1, 2**63, 1.5)),
                             ("delta_scale", (0., -1., float("nan"), float("inf"))),
                             ("distance_scale", (0., -1., float("nan"), float("inf")))):
            for value in values:
                with self.subTest(name=name, value=value):
                    config = dict(base, **{name: value})
                    with self.assertRaises(ValueError):
                        SpatialStateHeads(**config)


class BackboneCompatibilityTests(unittest.TestCase):
    def test_initialization_preserves_global_rng_and_is_seeded(self):
        torch.manual_seed(5678)
        state = torch.get_rng_state().clone()
        first = SpatialStateHeads(128, 11, delta_scale=19., distance_scale=1., init_seed=14)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        torch.randn(91)
        other_state = torch.get_rng_state().clone()
        second = SpatialStateHeads(128, 11, delta_scale=19., distance_scale=1., init_seed=14)
        self.assertTrue(torch.equal(other_state, torch.get_rng_state()))
        for name, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[name]), name)

    def test_construction_ignores_default_dtype_without_changing_it(self):
        default_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            head = EquivariantPositionHead(3, delta_scale=1., distance_scale=1.)
            self.assertEqual(next(head.parameters()).dtype, torch.float32)
            self.assertEqual(torch.get_default_dtype(), torch.float64)
        finally:
            torch.set_default_dtype(default_dtype)

    def test_128dim_gene_and_base_parameters_untouched_with_active_heads(self):
        torch.manual_seed(42)
        base = WaddingtonDiT1D(dim=128, hidden_dim=32, depth=1, num_heads=4,
                              num_register_tokens=2, time_emb_dim=32, curl_rank=2).eval()
        z, delta, epsilon = torch.randn(4, 128), torch.ones(4), torch.randn(4, 2, 128)
        reference = base(z, delta, epsilon).detach().clone()
        states = copy.deepcopy(base.state_dict())
        z_before = z.clone()
        heads = SpatialStateHeads(128, 9, delta_scale=19., distance_scale=1., hidden_dim=16)
        optimizer = torch.optim.SGD(heads.parameters(), lr=.1)
        x, neighbors, relative, context = torch.randn(4, 2), torch.randn(4, 3, 128), torch.randn(4, 3, 2), torch.randn(4, 9)
        # A detached gate is the runner's explicit choice when the base is frozen.
        alpha = base.alpha_gate(delta).detach()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            new_x, log_mass = heads(z, delta, alpha, x=x, neighbor_z=neighbors, relative_x=relative,
                                   log_mass=torch.zeros(4), context=context)
            loss = (new_x - x - 1).square().mean() + (log_mass - 1).square().mean()
            loss.backward()
            optimizer.step()
        self.assertEqual(z.shape, (4, 128))
        self.assertTrue(torch.equal(z, z_before))
        self.assertTrue(all(parameter.grad is None for parameter in base.parameters()))
        for name, value in states.items():
            self.assertTrue(torch.equal(value, base.state_dict()[name]), name)
        self.assertTrue(torch.equal(reference, base(z, delta, epsilon)))

    def test_before_off_and_missing_base_short_training_is_bitwise_equal(self):
        def train(mode):
            torch.manual_seed(312)
            base = WaddingtonDiT1D(dim=128, hidden_dim=32, depth=1, num_heads=4,
                                  num_register_tokens=2, time_emb_dim=32, curl_rank=2)
            initial_state = copy.deepcopy(base.state_dict())
            optimizer = torch.optim.SGD(base.parameters(), lr=1e-3)
            heads = None if mode == "before" else SpatialStateHeads(
                128, delta_scale=19., distance_scale=1., enabled=(mode != "off"))
            history, predictions = [], []
            for _ in range(2):
                z, delta, epsilon = torch.randn(3, 128), torch.ones(3), torch.randn(3, 2, 128)
                optimizer.zero_grad(set_to_none=True)
                prediction = base(z, delta, epsilon)
                if heads is not None:
                    self.assertEqual(heads(z, delta, base.alpha_gate(delta)), (None, None))
                loss = (prediction - z[:, None]).square().mean()
                loss.backward()
                optimizer.step()
                history.append(loss.detach())
                predictions.append(prediction.detach())
            self.assertTrue(any(not torch.equal(value, base.state_dict()[name])
                                for name, value in initial_state.items()))
            return base.state_dict(), torch.stack(history), predictions, torch.get_rng_state()
        before = train("before")
        for mode in ("off", "missing"):
            after = train(mode)
            for name, value in before[0].items():
                self.assertTrue(torch.equal(value, after[0][name]), (mode, name))
            self.assertTrue(torch.equal(before[1], after[1]), mode)
            for reference, actual in zip(before[2], after[2]):
                self.assertTrue(torch.equal(reference, actual), mode)
            self.assertTrue(torch.equal(before[3], after[3]), mode)
        self.assertNotEqual(before[1][0].item(), before[1][1].item())


if __name__ == "__main__":
    unittest.main()
