"""Smallest end-to-end hypercube_lcn run: learn a synthetic field map.

The task: given a random input field, predict a smoothed copy shifted one
vertex along axis 0 — local structure an LCN picks up quickly. Prints the
train loss falling and a held-out MSE at the end.
"""

import numpy as np

import hypercube_lcn as hl

DIM = 6
EPOCHS = 40
rng = np.random.default_rng(0)

net = hl.LCN(dim=DIM, gather_span=3, tanh_last=False,
             lr=1e-2, lr_min_frac=0.05, restore_best=True)
n = net.N

def make_set(count):
    x = rng.standard_normal((count, n)).astype(np.float32)
    # target: average of each vertex with its axis-0 neighbor
    y = 0.5 * (x + x[:, np.arange(n) ^ 1])
    return x, y.astype(np.float32)

fields, targets = make_set(256)
test_fields, test_targets = make_set(64)

net.fit(fields, targets, epochs=EPOCHS, batch_size=16, verbose=True)

pred = np.stack([net.forward(f) for f in test_fields])
mse = float(np.mean((pred - test_targets) ** 2))
print(f"held-out MSE: {mse:.6f}")
print(net)
