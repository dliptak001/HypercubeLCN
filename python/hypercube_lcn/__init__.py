"""HypercubeLCN: a locally connected network on a Boolean hypercube.

A deep feedforward network whose connectivity is the cube's own edges and
whose weights are all trained: every vertex owns a private weight table at
every depth, neighbors are one bit-flip away, and a lookback window keeps
the input visible to the early depths.

Quick start::

    import numpy as np
    import hypercube_lcn as hl

    # fields: (num_samples, N) float32, targets: (num_samples, N) float32
    net = hl.LCN(dim=11, gather_span=4, tanh_last=False,
                 lr=5e-3, lr_min_frac=0.05, restore_best=True)
    net.fit(fields_train, targets_train, epochs=30, batch_size=48)
    baseline = net.forward(fields_test[0])
"""

from __future__ import annotations

import pathlib
import pickle

import numpy as np

from ._core import _Core, _Training
from ._version import __version__

__all__ = ["LCN", "__version__"]

# Valid hypercube dimensions (matches the C++ Core constructor [4, 24] check).
_DIM_MIN = 4
_DIM_MAX = 24


def _to_float32(arr):
    """Ensure array is C-contiguous float32."""
    return np.ascontiguousarray(arr, dtype=np.float32)


class LCN:
    """A trainable locally connected network on a hypercube.

    N = 2^dim is the field length: input, every intermediate field, and
    output all live on the same N vertices. Each sample is a full length-N
    field (you pack domain data on the host; pad unused vertices). Targets
    may be narrower than N — then only vertices ``0 .. width-1`` carry loss
    and the rest of the cube is free hidden units.

    Typical lifecycle: construct, :meth:`fit`, :meth:`forward`, or drive a
    custom loop with :meth:`zero_grad` / :meth:`loss` / :meth:`backward` /
    :meth:`adam`.

    Parameters
    ----------
    dim : int
        Hypercube dimension (4-24). N = 2^dim field length.
    seed : int
        Weight-init seed (normal draw, Xavier-style scale).
    z_max : int
        Depth count; 0 (default) = use ``dim``, else must be >= 2.
    gather_span : int
        Lookback window width in fields, 2-6. Default: 2.
    tanh_last : bool
        If True the last depth applies tanh, confining the output to
        (-1, 1). Default False: the raw accumulator — what squared-error
        training wants, even against ±1 targets.
    lr : float
        Adam step size; cosine peak. Default: 5e-3.
    lr_min_frac : float
        Cosine floor as a fraction of ``lr``, in [0, 1]; 1 = constant lr.
    lr_decay_epochs : int
        Cosine horizon; 0 = use fit's ``epochs``.
    restore_best : bool
        Snapshot weights on a new low epoch loss during :meth:`fit` and
        restore the best at the end. Default: False.
    beta1, beta2, eps : float
        Adam moments and denominator floor.

    Notes
    -----
    This class is **not thread-safe** for concurrent calls from multiple
    host threads.

    The per-batch training cycle (what :meth:`fit` runs for you) is::

        net.zero_grad()
        for x, y in batch:
            net.forward(x)
            net.loss(y)
            net.backward()
        net.adam()

    The gradient is a **sum** over the batch, so the effective step scales
    with batch size.
    """

    def __init__(
        self,
        dim: int,
        *,
        seed: int = 934791766227647176,
        z_max: int = 0,
        gather_span: int = 2,
        tanh_last: bool = False,
        lr: float = 5e-3,
        lr_min_frac: float = 1.0,
        lr_decay_epochs: int = 0,
        restore_best: bool = False,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        if not isinstance(dim, int) or not (_DIM_MIN <= dim <= _DIM_MAX):
            raise ValueError(
                f"dim must be an integer in [{_DIM_MIN}, {_DIM_MAX}], got {dim!r}"
            )
        self._ctor = {
            "dim": dim,
            "seed": seed,
            "z_max": z_max,
            "gather_span": gather_span,
            "tanh_last": tanh_last,
            "lr": lr,
            "lr_min_frac": lr_min_frac,
            "lr_decay_epochs": lr_decay_epochs,
            "restore_best": restore_best,
            "beta1": beta1,
            "beta2": beta2,
            "eps": eps,
        }
        self._core = _Core(
            dim=dim, seed=seed, z_max=z_max,
            gather_span=gather_span, tanh_last=tanh_last,
        )
        self._train = _Training(
            self._core, lr=lr, lr_min_frac=lr_min_frac,
            lr_decay_epochs=lr_decay_epochs, restore_best=restore_best,
            beta1=beta1, beta2=beta2, eps=eps,
        )

    # ── Inference ──

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Run the forward pass on one field and return the output field.

        Parameters
        ----------
        x : ndarray
            Length-N field (or shape that ravel-flattens to N). Converted
            to float32.

        Returns
        -------
        ndarray
            Length-N float32 output field (a copy).
        """
        self._core.forward(_to_float32(np.ravel(x)))
        return self._core.output()

    # ── Training (one-shot) ──

    def fit(
        self,
        fields: np.ndarray,
        targets: np.ndarray,
        *,
        epochs: int,
        batch_size: int = 32,
        shuffle_seed: int = 1,
        verbose: bool = False,
    ) -> "LCN":
        """Train on a sample set with the standard batch cycle.

        Runs ``epochs`` passes: shuffle, then per batch zero_grad /
        forward / loss / backward per sample / adam. With ``restore_best``
        the mean epoch loss is observed and the best weights are restored
        at the end. Calling fit again continues from the current weights
        (use :meth:`reset` first for a fresh optimizer).

        Parameters
        ----------
        fields : ndarray
            Shape ``(count, N)`` float32 input fields.
        targets : ndarray
            Shape ``(count, width)`` with width <= N. width == N is
            full-field regression; a smaller width masks the loss to
            vertices ``0 .. width-1`` (free hidden vertices).
        epochs : int
            Number of passes over the set.
        batch_size : int
            Samples per Adam step. A short tail that cannot fill a batch
            is dropped each epoch (every step sees ``batch_size`` samples).
        shuffle_seed : int
            Seed for the per-epoch shuffle.
        verbose : bool
            Print ``epoch  train_loss`` lines.

        Returns
        -------
        LCN
            Self, for method chaining.
        """
        fields = _to_float32(fields)
        targets = _to_float32(targets)
        n = self.N
        if fields.ndim != 2 or fields.shape[1] != n:
            raise ValueError(
                f"fields must be shape (count, {n}), got {fields.shape}"
            )
        if targets.ndim != 2 or targets.shape[0] != fields.shape[0]:
            raise ValueError(
                f"targets must be shape ({fields.shape[0]}, width<= {n}), "
                f"got {targets.shape}"
            )
        if not (1 <= targets.shape[1] <= n):
            raise ValueError(
                f"target width must be in [1, {n}], got {targets.shape[1]}"
            )
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        count = fields.shape[0]
        n_use = count - (count % batch_size)
        if n_use == 0:
            n_use = count
        rng = np.random.default_rng(shuffle_seed)

        for epoch in range(epochs):
            self._train.set_epoch(epoch, epochs)
            order = rng.permutation(count)
            loss_sum = 0.0
            for start in range(0, n_use, batch_size):
                self._train.zero_grad()
                for i in order[start:start + batch_size]:
                    self._core.forward(fields[i])
                    loss_sum += self._train.loss(targets[i])
                    self._train.backward()
                self._train.adam()
            mean_loss = loss_sum / n_use
            if verbose:
                print(f"epoch={epoch}/{epochs} train_loss={mean_loss:.6f}")
            self._train.observe(mean_loss, epoch)
        self._train.restore_best()
        return self

    # ── Training (custom loop) ──

    def zero_grad(self) -> None:
        """Clear the accumulated gradient. Call at the start of each batch."""
        self._train.zero_grad()

    def loss(self, target: np.ndarray) -> float:
        """0.5 x SSE against the most recent :meth:`forward`, seeding the
        gradient for :meth:`backward`.

        ``len(target) == N`` is full-width; shorter masks the loss to
        vertices ``0 .. len-1``.
        """
        return float(self._train.loss(_to_float32(np.ravel(target))))

    def backward(self) -> None:
        """Backprop the most recent :meth:`loss`, accumulating (summing)
        into the gradient. Call once per sample."""
        self._train.backward()

    def adam(self) -> None:
        """One Adam step on the accumulated gradient."""
        self._train.adam()

    def accumulate_grad(self, g: np.ndarray) -> None:
        """Add an externally accumulated gradient into this network's
        gradient, elementwise.

        The data-parallel merge: worker networks (clones sharing this
        network's constructor knobs) build private gradients with
        :meth:`zero_grad` / :meth:`loss` / :meth:`backward`, then one
        master accumulates each worker's :attr:`grad` and takes the
        single :meth:`adam` step. Same layout and length as
        :attr:`weights`.
        """
        g = _to_float32(np.ravel(g))
        if g.size != self.num_weights:
            raise ValueError(
                f"grad length ({g.size}) must equal num_weights "
                f"({self.num_weights})"
            )
        self._train.accumulate_grad(g)

    def set_epoch(self, epoch: int, num_epochs: int = 0) -> None:
        """Apply the cosine learning-rate schedule for this epoch."""
        self._train.set_epoch(epoch, num_epochs)

    def observe(self, metric: float, epoch: int) -> None:
        """Snapshot weights on a new low metric (restore_best only)."""
        self._train.observe(float(metric), int(epoch))

    def restore_best(self) -> None:
        """Write the best-metric snapshot back into the network."""
        self._train.restore_best()

    def reset(self) -> None:
        """Forget the optimizer run: Adam moments, step count, lr, best
        snapshot. Weights are untouched."""
        self._train.reset()

    # ── Weights / gradient ──

    @property
    def weights(self) -> np.ndarray:
        """Copy of all trainable weights: depth, axis, tap, vertex.
        Length N * dim * gather_span * z_max. Vertex is the fastest index."""
        return self._core.weights()

    @weights.setter
    def weights(self, w: np.ndarray) -> None:
        w = _to_float32(np.ravel(w))
        if w.size != self.num_weights:
            raise ValueError(
                f"weights length ({w.size}) must equal num_weights "
                f"({self.num_weights})"
            )
        self._core.load_weights(w)

    @property
    def grad(self) -> np.ndarray:
        """Copy of the accumulated gradient, same layout as :attr:`weights`."""
        return self._train.grad()

    # ── Properties ──

    @property
    def dim(self) -> int:
        """Hypercube dimension."""
        return int(self._core.dim)

    @property
    def N(self) -> int:
        """Field length: 2^dim."""
        return int(self._core.N)

    @property
    def seed(self) -> int:
        """Weight-init seed."""
        return int(self._core.seed)

    @property
    def z_max(self) -> int:
        """Resolved depth count (0 in the constructor becomes dim)."""
        return int(self._core.z_max)

    @property
    def gather_span(self) -> int:
        """Lookback window width in fields."""
        return int(self._core.gather_span)

    @property
    def tanh_last(self) -> bool:
        return bool(self._core.tanh_last)

    @property
    def num_weights(self) -> int:
        """Total trainable weights: N * dim * gather_span * z_max."""
        return int(self._core.num_weights)

    @property
    def lr(self) -> float:
        """Learning rate currently in effect (after set_epoch)."""
        return float(self._train.lr)

    @property
    def has_best(self) -> bool:
        """True once observe() has snapshotted a best."""
        return bool(self._train.has_best)

    @property
    def best_epoch(self) -> int:
        return int(self._train.best_epoch)

    @property
    def best_metric(self) -> float:
        return float(self._train.best_metric)

    def __repr__(self) -> str:
        return (
            f"LCN(dim={self.dim}, N={self.N}, z_max={self.z_max}, "
            f"gather_span={self.gather_span}, tanh_last={self.tanh_last}, "
            f"weights={self.num_weights})"
        )

    # ── Persistence ──

    _PERSISTENCE_VERSION = 2

    def __getstate__(self) -> dict:
        """Serialize constructor config + weights.

        Optimizer state (Adam moments, best snapshot) is **not** saved.
        """
        return {
            "_version": self._PERSISTENCE_VERSION,
            "ctor": dict(self._ctor),
            "weights": self._core.weights(),
        }

    def __setstate__(self, state: dict) -> None:
        version = state.get("_version", 0)
        if version > self._PERSISTENCE_VERSION:
            raise ValueError(
                f"Model was saved with persistence version {version}, "
                f"but this version only supports up to "
                f"{self._PERSISTENCE_VERSION}. Upgrade hypercube-lcn."
            )
        if version < 2:
            raise ValueError(
                "Model was saved with persistence version "
                f"{version} (weight layout depth, vertex, axis, tap). "
                "This version stores depth, axis, tap, vertex. "
                "Retrain; old pickles are not loaded."
            )
        self.__init__(**dict(state["ctor"]))
        self._core.load_weights(_to_float32(state["weights"]))

    def save(self, path) -> None:
        """Save config + weights to a pickle file (optimizer state is not
        saved)."""
        with open(pathlib.Path(path), "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path) -> "LCN":
        """Load a model saved by :meth:`save`.

        .. warning::

            Uses ``pickle.load``. Never load untrusted files.
        """
        with open(pathlib.Path(path), "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected LCN, got {type(obj).__name__}")
        return obj
