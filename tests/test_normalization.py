"""Joint ragged statistics must agree with PyTorch and keep AD connected."""

import pytest
import torch
from torch import nn

from pinn_engine import normalization as norm


@pytest.fixture(autouse=True)
def small_cpu_workloads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(11)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("dimensions", (0, 1, 2, 3))
@pytest.mark.parametrize("momentum", (None, .1))
def test_joint_list_batchnorm_outputs_buffers_and_gradients_match_native(dimensions, momentum):
    kind = getattr(norm, f"BatchNorm{max(1, dimensions)}d")
    shapes = [(2, 3) + (4,) * dimensions, (1, 3) + (2,) * dimensions]
    actual_inputs = [torch.randn(s, dtype=torch.double, requires_grad=True) for s in shapes]
    expected_inputs = [t.detach().clone().requires_grad_() for t in actual_inputs]
    actual = kind(3, momentum=momentum).double()
    expected = nn.BatchNorm1d(3, momentum=momentum).double()
    expected.load_state_dict(actual.state_dict())
    for _ in range(2):
        result = actual(actual_inputs)
        # One native BN on real samples is an independent reference for all
        # spatial ranks; there is no padded data in this reference.
        rows = [t.movedim(1, -1).reshape(-1, 3) for t in expected_inputs]
        reference = expected(torch.cat(rows))
        for a, b in zip(result, reference.split([r.shape[0] for r in rows])):
            torch.testing.assert_close(a.movedim(1, -1).reshape(-1, 3), b)
        for key, value in actual.state_dict().items():
            torch.testing.assert_close(value, expected.state_dict()[key])
    sum(t.sin().sum() for t in result).backward()
    reference.sin().sum().backward()
    for a, b in zip(actual_inputs, expected_inputs):
        torch.testing.assert_close(a.grad, b.grad)
    torch.testing.assert_close(actual.weight.grad, expected.weight.grad)
    torch.testing.assert_close(actual.bias.grad, expected.bias.grad)


def test_masked_batchnorm_ignores_padding_nan_and_restores_zero_outputs():
    tensors = [torch.randn(2, 3, 5, dtype=torch.double), torch.randn(1, 3, 8, dtype=torch.double)]
    masks = [torch.tensor([[1, 1, 0, 1, 0], [1, 0, 1, 1, 1]], dtype=torch.bool),
             torch.tensor([[1, 1, 1, 1, 1, 0, 1, 0]], dtype=torch.bool)]
    for t, m in zip(tensors, masks):
        t.masked_fill_(~m[:, None], float("nan"))
        t.requires_grad_()
    bn = norm.BatchNorm1d(3).double()
    native = nn.BatchNorm1d(3).double()
    native.load_state_dict(bn.state_dict())
    valid = torch.cat([t.movedim(1, -1)[m] for t, m in zip(tensors, masks)])
    expected = native(valid)
    result = bn(tensors, masks)
    joined = torch.cat([t.movedim(1, -1)[m] for t, m in zip(result, masks)])
    torch.testing.assert_close(joined, expected)
    for t, m in zip(result, masks):
        assert torch.count_nonzero(t.movedim(1, -1)[~m]) == 0
    joined.square().sum().backward()
    for t, m in zip(tensors, masks):
        assert torch.isfinite(t.grad).all()
        assert torch.count_nonzero(t.grad.movedim(1, -1)[~m]) == 0
    assert bn.num_batches_tracked == 1
    torch.testing.assert_close(bn.running_mean, native.running_mean)
    torch.testing.assert_close(bn.running_var, native.running_var)


@pytest.mark.parametrize("training,track", ((True, False), (False, False), (False, True)))
def test_list_batchnorm_eval_and_missing_running_statistics(training, track):
    bn = norm.BatchNorm1d(2, track_running_stats=track).double().train(training)
    native = nn.BatchNorm1d(2, track_running_stats=track).double().train(training)
    native.load_state_dict(bn.state_dict())
    tensors = [torch.randn(1, 2, 3, dtype=torch.double), torch.randn(2, 2, 5, dtype=torch.double)]
    actual = bn(tensors)
    rows = [t.movedim(1, -1).reshape(-1, 2) for t in tensors]
    expected = native(torch.cat(rows))
    torch.testing.assert_close(torch.cat([t.movedim(1, -1).reshape(-1, 2) for t in actual]), expected)


def test_list_batchnorm_supports_second_derivatives():
    bn = norm.BatchNorm1d(2, track_running_stats=False).double()
    tensors = (torch.randn(1, 2, 3, dtype=torch.double, requires_grad=True),
               torch.randn(1, 2, 2, dtype=torch.double, requires_grad=True))
    function = lambda *args: tuple(bn(list(args)))
    assert torch.autograd.gradcheck(function, tensors)
    assert torch.autograd.gradgradcheck(function, tensors)


def test_point_instance_norm_is_per_case_masked_and_vmap_compatible():
    layer = norm.InstanceNorm1d(3, affine=True).double()
    tensors = [torch.randn(2, 3, 4, dtype=torch.double), torch.randn(1, 3, 7, dtype=torch.double)]
    native = nn.InstanceNorm1d(3, affine=True).double()
    native.load_state_dict(layer.state_dict())
    for actual, tensor in zip(layer(tensors), tensors):
        torch.testing.assert_close(actual, native(tensor))
    mask = torch.tensor([[True, True, False, True], [True, False, True, True]])
    masked = tensors[0].masked_fill(~mask[:, None], float("nan")).requires_grad_()
    actual = torch.func.vmap(lambda t, m: layer(t[None], m[None])[0])(masked, mask)
    for i in range(2):
        expected = native(masked[i:i+1, :, mask[i]])
        torch.testing.assert_close(actual[i, :, mask[i]], expected[0])
    actual.square().sum().backward()
    assert torch.isfinite(masked.grad).all()


def test_norm_rejects_invalid_lists_and_restores_mask_context():
    bn = norm.BatchNorm1d(2)
    with pytest.raises(TypeError):
        bn([])
    with pytest.raises(ValueError):
        bn([torch.randn(1, 2, 3), torch.randn(1, 3, 4)])
    with pytest.raises(ValueError):
        bn([torch.randn(1, 2, 3)], [torch.ones(1, 4, dtype=torch.bool)])
    with pytest.raises(ValueError):
        bn([torch.randn(1, 2, 3)], [torch.zeros(1, 3, dtype=torch.bool)])
    assert not norm.has_normalization_mask()
    with pytest.raises(RuntimeError):
        with norm.normalization_mask(torch.ones(1, 3, dtype=torch.bool)):
            assert norm.has_normalization_mask()
            raise RuntimeError("test scope cleanup")
    assert not norm.has_normalization_mask()
