# Training

`Core` is the network: it owns the weights and runs the forward pass.
`Training` is a friend class that walks the same loop nest backwards,
accumulates gradients, and steps the weights with Adam. This note
follows [Training.cpp](../Training.cpp); the forward pass it inverts is
[forward.md](forward.md), and the notation (`n`, `span`, `z_max`,
slots, taps) is that note's.

## Two classes, one nest

`Core` keeps its internals private; `Training` is declared a friend so
the reverse pass can read `s_` and `w_` directly instead of going
through a pile of getters. A `Training` holds a reference to its
`Core` — it does not own the network — plus four buffers of its own:

| Buffer | Mirrors | Holds |
|--------|---------|-------|
| `ds_`  | `s_`    | Gradient of the loss with respect to each history slot |
| `dw_`  | `w_`    | Gradient with respect to each weight, summed over a minibatch |
| `m_`, `v_` | `w_` | Adam's first and second moment estimates |

The knobs live in `TrainingConfig` at the head of
[Training.h](../Training.h):

| Knob | Default | Meaning |
|------|---------|---------|
| `lr` | `5e-3` | Adam step size; peak of the cosine schedule |
| `lr_min_frac` | `1` | Schedule floor as a fraction of `lr`; 1 = constant |
| `lr_decay_epochs` | `0` | Cosine horizon; 0 = use `SetEpoch`'s `num_epochs` |
| `restore_best` | `false` | Snapshot weights on a new best score; restore at end |
| `beta1`, `beta2` | `0.9`, `0.999` | Adam moment decay rates |
| `eps` | `1e-8` | Floor under the Adam denominator |

## The training loop

The dataset is pairs of fields `(x, y)`: `x` goes into `Forward`, `y`
is what `Output()` should have been. Since `Forward` fully rewrites the
history stack on every call, samples are independent — there is no
state to reset between them. The host (the Raman example, for one)
drives this loop:

```
for each epoch:
    train.SetEpoch(epoch, num_epochs)          // pick this epoch's lr
    for each minibatch:
        train.ZeroGrad()
        for each (x, y) in the minibatch:
            core.Forward(x)
            train.Loss(y)                      // seed ds_ from the output error
            train.Backward()                   // accumulate into dw_
        train.Adam()                           // one step on the summed gradient
    train.Observe(score, epoch)                // maybe snapshot the weights
train.RestoreBest()                            // maybe roll back to the snapshot
```

`Loss` and `Backward` run one sample at a time; `dw_` sums across the
minibatch and `Adam` takes one step per minibatch. The gradient is
summed, not averaged — but Adam divides each step by the gradient's own
running magnitude, so the batch-size factor largely cancels out of the
step length.

## Loss: seeding the backward pass

The loss is half the summed squared error over the output field:
`0.5 · Σ (o[v] − y[v])²`. `Loss` returns that number for logging, but
its real job is seeding `ds_`: it zeroes the whole buffer, then writes
the output error into the output slot —

```
ds[(z_max + span − 1) * n + v] = o[v] − y[v]
```

— which is exactly the derivative of the loss with respect to the
output (the `0.5` exists to make that derivative clean). Everything
else in `ds_` starts at zero and is filled in by `Backward` as the
error flows toward the input. The full zeroing matters: `Backward`
accumulates into `ds_` with `+=`, so each sample must start from a
clean slate.

## Backward: the nest in reverse

`Forward` at depth `z` read slots `z .. z+span−1` and wrote slot
`z + span`. `Backward` visits the depths in the opposite order,
`z = z_max−1` down to `0`, and inverts each write.

At vertex `v`, depth `z`, the first step is passing the gradient back
through the activation. `Forward` wrote `y = tanh(acc)` into slot
`z + span`; the derivative of `tanh` at that point is `1 − y²`,
computable from the stored output itself — no need to have kept `acc`:

```
y        = s[(z + span) * n + v]
incoming = ds[(z + span) * n + v] * (1 − y²)
```

(On the last depth with `tanh_last` false, `Forward` wrote `acc` raw,
so the factor is 1.)

`incoming` is the gradient at the accumulator. The accumulator was a
plain weighted sum, so the gradient splits by the product rule into the
same axis × tap traversal `Forward` made, with two accumulations per
weight — one for the weight, one for the history value it read:

```
for each axis, v_nn = v XOR (1 << axis), for each tap k:
    dw[z, v, axis, k]      += incoming * s[(z + k) * n + v_nn]
    ds[(z + k) * n + v_nn] += incoming * w[z, v, axis, k]
```

The first line is the gradient for this vertex's own table — tables are
private, so no other vertex touches these slots; the sum in `dw_` is
only across taps that read the same value, and across minibatch
samples. The second line sends gradient back to the neighbors' history,
where it waits to be picked up when the loop reaches the depth that
wrote that slot.

The ordering works for the same reason it worked forward, mirrored:
depth `z` reads gradient from slot `z + span` and deposits into slots
`z .. z+span−1`, and since `z` is descending, every deeper depth has
already finished depositing into slot `z + span` before depth `z`
consumes it.

Taps that land in the zero prefix get the honest treatment for free:
their `dw` contribution is `incoming × 0`, so the dead weights
identified in [forward.md](forward.md) stay frozen at their initial
values, and their `ds` deposits land in prefix slots that no depth
reads — spilled into a strip of zeros that exists partly for this
purpose.

`Backward` reads `s_` throughout, so the history from the sample's
`Forward` must still be intact — run `Forward`, `Loss`, `Backward` as
an unbroken triple per sample. A serial-number guard enforces it:
`Backward` throws unless a fresh `Loss` matches the most recent
`Forward`, each `Loss` seeds exactly one `Backward` (the pass dirties
`ds_` on the way down), and mutating the weights (`Adam`,
`LoadWeights`) invalidates a pending seed.

## Adam

`Adam()` is one optimizer over the packed weight vector, the standard
bias-corrected form. Per weight, with `g` the summed gradient from
`dw_`:

```
m = beta1 * m + (1 − beta1) * g          // running mean of g
v = beta2 * v + (1 − beta2) * g²         // running mean of g²
w −= lr * (m / bc1) / (sqrt(v / bc2) + eps)
```

`bc1` and `bc2` are the usual startup corrections (`1 − beta^t`, with
`t` counting optimizer steps), which keep the first steps from being
dwarfed by the zero-initialized moments. The division by `sqrt(v)`
normalizes each weight's step by its own gradient scale — that is what
makes the summed (unaveraged) minibatch gradient and any depth-to-depth
gradient imbalance tolerable without hand-tuning per-layer rates.

The step uses `Lr()`, the scheduled rate — never a raw read of
`cfg.lr`.

## The learning-rate schedule

`SetEpoch(epoch, num_epochs)` — called once at the top of each epoch —
sets the rate from a half-cosine that starts at `lr` and lands on
`lr * lr_min_frac`:

```
progress = epoch / (horizon − 1)
Lr()     = lr_min + 0.5 * (lr_max − lr_min) * (1 + cos(pi * progress))
```

The horizon is `lr_decay_epochs` when positive, otherwise the
`num_epochs` argument. Edge behavior: a horizon of 1 or less stays at
the peak, epochs past the horizon clamp to the floor, and the default
`lr_min_frac = 1` makes the whole schedule an identity. Call `SetEpoch`
even when the schedule is flat — then turning the schedule on later is
a config change, not a code change.

The Raman run banner shows the working configuration: `lr = 5e-3`
decaying to 5% of peak (`lr_min_frac = 0.05`) over the full run.

## Best-weight restore

Validation scores are noisy, and the last epoch is not reliably the
best one. With `restore_best` on, `Observe(metric, epoch)` snapshots
the entire weight vector whenever the metric makes a new strict low
(ties keep the earlier epoch; non-finite metrics are ignored), and
`RestoreBest()` writes the winning snapshot back at the end of the run.
The metric is whatever the host passes — the Raman example uses the
held-out RMSE printed on each epoch line, which is why its final
weights correspond to the `restore_best epoch=...` line in the log.

Two deliberate limits: with `restore_best` off (the default) both calls
are no-ops and the last-epoch weights stand, and the Adam moments are
*not* snapshotted — this is an end-of-run pick, not a rewind. Resuming
training after a restore would take a few steps to re-estimate the
moments.

For gradient inspection, `Grad()` exposes the current `dw_` read-only.
