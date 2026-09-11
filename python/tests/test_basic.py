"""Basic tests for the hypercube_lcn package.

Small cubes (dim 4-5) keep every test fast; the goal is contract
coverage, not benchmark scores.
"""

import numpy as np
import pytest

import hypercube_lcn as hl


def make_net(**kw):
    args = dict(dim=4, gather_span=2)
    args.update(kw)
    return hl.LCN(**args)


# ── Construction / validation ──

def test_version():
    assert isinstance(hl.__version__, str) and hl.__version__
    assert hl.__version__ == hl._core.__version__


def test_sizes():
    net = make_net(dim=5, z_max=3, gather_span=2)
    assert net.dim == 5
    assert net.N == 32
    assert net.z_max == 3
    assert net.gather_span == 2
    assert net.num_weights == 32 * 5 * 2 * 3


def test_z_max_zero_resolves_to_dim():
    net = make_net(dim=4, z_max=0)
    assert net.z_max == 4


def test_dim_bounds():
    with pytest.raises(ValueError):
        hl.LCN(dim=3)
    with pytest.raises(ValueError):
        hl.LCN(dim=25)


def test_bad_config_throws():
    with pytest.raises(ValueError):
        make_net(gather_span=1)
    with pytest.raises(ValueError):
        make_net(gather_span=7)
    with pytest.raises(ValueError):
        make_net(z_max=1)
    with pytest.raises(ValueError):
        make_net(lr=-1.0)
    with pytest.raises(ValueError):
        make_net(lr_min_frac=2.0)
    with pytest.raises(ValueError):
        make_net(beta1=1.0)
    with pytest.raises(ValueError):
        make_net(beta2=-0.1)
    with pytest.raises(ValueError):
        make_net(eps=0.0)
    with pytest.raises(ValueError):
        make_net(lr_decay_epochs=-1)
    # NaN must be rejected even under fast-math (bit-level guard)
    with pytest.raises(ValueError):
        make_net(lr=float("nan"))
    with pytest.raises(ValueError):
        make_net(lr_min_frac=float("nan"))
    with pytest.raises(ValueError):
        make_net(eps=float("inf"))


# ── Forward ──

def test_forward_shape_and_determinism():
    net1 = make_net(seed=7)
    net2 = make_net(seed=7)
    x = np.linspace(-1, 1, net1.N, dtype=np.float32)
    y1 = net1.forward(x)
    y2 = net2.forward(x)
    assert y1.shape == (net1.N,)
    assert y1.dtype == np.float32
    np.testing.assert_array_equal(y1, y2)


def test_forward_seed_changes_output():
    x = np.linspace(-1, 1, 16, dtype=np.float32)
    y1 = make_net(seed=1).forward(x)
    y2 = make_net(seed=2).forward(x)
    assert not np.array_equal(y1, y2)


def test_forward_wrong_length_throws():
    net = make_net()
    with pytest.raises(ValueError):
        net.forward(np.zeros(net.N + 1, dtype=np.float32))


def test_tanh_last_bounds_output():
    x = 100.0 * np.ones(16, dtype=np.float32)
    y = make_net(tanh_last=True).forward(x)
    assert np.all(np.abs(y) <= 1.0)


# ── Training ──

def test_fit_reduces_loss():
    rng = np.random.default_rng(0)
    net = make_net(dim=4, tanh_last=False, lr=1e-2)
    n = net.N
    fields = rng.standard_normal((64, n)).astype(np.float32)
    targets = np.roll(fields, 1, axis=1) * 0.5  # a learnable local-ish map

    first = 0.0
    for i in range(64):
        net.forward(fields[i])
        first += net.loss(targets[i])

    net.fit(fields, targets, epochs=30, batch_size=16)

    last = 0.0
    for i in range(64):
        net.forward(fields[i])
        last += net.loss(targets[i])
    assert last < first * 0.5


def test_masked_loss_width():
    net = make_net(dim=4)
    x = np.zeros(net.N, dtype=np.float32)
    net.forward(x)
    assert net.loss(np.zeros(4, dtype=np.float32)) >= 0.0
    with pytest.raises(ValueError):
        net.loss(np.zeros(net.N + 1, dtype=np.float32))
    with pytest.raises(ValueError):
        net.loss(np.zeros(0, dtype=np.float32))


def test_grad_accumulates_and_zeroes():
    net = make_net(dim=4)
    x = np.ones(net.N, dtype=np.float32)
    t = np.zeros(net.N, dtype=np.float32)
    net.zero_grad()
    net.forward(x)
    net.loss(t)
    net.backward()
    g1 = net.grad.copy()
    assert np.any(g1 != 0.0)
    net.forward(x)
    net.loss(t)
    net.backward()
    g2 = net.grad
    np.testing.assert_allclose(g2, 2.0 * g1, rtol=1e-4)
    net.zero_grad()
    assert not np.any(net.grad)


def test_accumulate_grad_merges_workers():
    # Data-parallel contract: worker gradients merged into the master
    # match the same samples run serially on one instance.
    master = make_net(dim=4)
    workers = [make_net(dim=4) for _ in range(2)]
    rng = np.random.default_rng(7)
    xs = rng.standard_normal((4, master.N)).astype(np.float32)
    ts = rng.standard_normal((4, master.N)).astype(np.float32)

    serial = make_net(dim=4)
    serial.zero_grad()
    for i in range(4):
        serial.forward(xs[i])
        serial.loss(ts[i])
        serial.backward()
    g_serial = serial.grad

    for k, w in enumerate(workers):
        w.weights = master.weights
        w.zero_grad()
        for i in range(k, 4, 2):  # strided share
            w.forward(xs[i])
            w.loss(ts[i])
            w.backward()
    master.zero_grad()
    for w in workers:
        master.accumulate_grad(w.grad)
    np.testing.assert_allclose(master.grad, g_serial, rtol=1e-4, atol=1e-6)


def test_accumulate_grad_wrong_length_throws():
    net = make_net(dim=4)
    with pytest.raises(ValueError):
        net.accumulate_grad(np.zeros(net.num_weights - 1, dtype=np.float32))


def test_backward_requires_fresh_loss():
    net = make_net(dim=4)
    x = np.zeros(net.N, dtype=np.float32)
    t = np.zeros(net.N, dtype=np.float32)
    net.forward(x)
    with pytest.raises(ValueError):
        net.backward()          # no loss against this forward
    net.loss(t)
    net.backward()              # in order: fine
    with pytest.raises(ValueError):
        net.backward()          # seed consumed: needs a fresh loss
    net.forward(x)
    with pytest.raises(ValueError):
        net.backward()          # stale: new forward, no new loss


def test_backward_rejected_before_any_forward():
    net = make_net(dim=4)
    with pytest.raises(ValueError):
        net.backward()


def test_backward_rejected_after_weight_mutation():
    net = make_net(dim=4)
    x = np.zeros(net.N, dtype=np.float32)
    t = np.zeros(net.N, dtype=np.float32)

    # adam between loss and backward: activations no longer match weights
    net.zero_grad()
    net.forward(x)
    net.loss(t)
    net.adam()
    with pytest.raises(ValueError):
        net.backward()

    # same for a weights load
    net.forward(x)
    net.loss(t)
    net.weights = net.weights.copy()
    with pytest.raises(ValueError):
        net.backward()


def test_restore_best():
    rng = np.random.default_rng(1)
    net = make_net(dim=4, restore_best=True)
    w0 = net.weights
    net.observe(1.0, 0)
    assert net.has_best and net.best_epoch == 0
    # worsen the weights, then a worse metric must NOT snapshot
    net.weights = rng.standard_normal(net.num_weights).astype(np.float32)
    net.observe(2.0, 1)
    assert net.best_epoch == 0
    net.restore_best()
    np.testing.assert_array_equal(net.weights, w0)


def test_reset_clears_optimizer():
    net = make_net(dim=4, restore_best=True)
    net.observe(1.0, 3)
    assert net.has_best
    net.reset()
    assert not net.has_best
    assert net.best_epoch == -1


# ── Weights / persistence ──

def test_weights_roundtrip():
    net = make_net(dim=4)
    w = net.weights
    assert w.size == net.num_weights
    w2 = np.arange(w.size, dtype=np.float32) / w.size
    net.weights = w2
    np.testing.assert_array_equal(net.weights, w2)
    with pytest.raises(ValueError):
        net.weights = w2[:-1]


def test_pickle_roundtrip(tmp_path):
    rng = np.random.default_rng(2)
    net = make_net(dim=5, z_max=3, gather_span=3, tanh_last=False, seed=99)
    net.weights = rng.standard_normal(net.num_weights).astype(np.float32)
    x = rng.standard_normal(net.N).astype(np.float32)
    y_before = net.forward(x)

    path = tmp_path / "model.pkl"
    net.save(path)
    loaded = hl.LCN.load(path)

    assert loaded.dim == 5 and loaded.z_max == 3 and loaded.gather_span == 3
    assert loaded.tanh_last is False and loaded.seed == 99
    np.testing.assert_array_equal(loaded.forward(x), y_before)


def test_repr():
    r = repr(make_net(dim=4))
    assert "LCN(dim=4" in r
