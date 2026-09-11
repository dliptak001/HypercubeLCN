# Python examples

Demo scripts for the `hypercube-lcn` API. They call only the installed
package.

| Script | What it shows |
|--------|----------------|
| [`synthetic_regression.py`](synthetic_regression.py) | Toy length-N field map → `fit` with verbose loss → held-out MSE |

## How to run

These files live in the GitHub repository. They are not part of `pip install`
— use the [Quick start](../README.md#quick-start) if you only want a snippet.

From a clone of HypercubeLCN (repository root), after installing the
package:

```bash
pip install hypercube-lcn
# or from this tree:  pip install ./python
python python/examples/synthetic_regression.py
```

## What these are not

- **Not** the C++ demos (`lcn_raman`, `lcn_mnist`, and friends) or the study
  write-ups. Those live under [`examples/`](../../examples/).
- **Not** automated tests. Package tests are [`tests/test_basic.py`](../tests/test_basic.py).
- **Not** hard tasks. Synthetic field maps are easy onboarding so the
  training API is obvious; do not cite their metrics as research results.

## Going further

| Want… | See… |
|-------|------|
| Full Python API | [`docs/Python_SDK.md`](../../docs/Python_SDK.md) |
| C++ product guide | [`docs/CPP_SDK.md`](../../docs/CPP_SDK.md) |
| Package readme | [`README.md`](../README.md) |
| The forward pass, loop by loop | [`docs/forward.md`](../../docs/forward.md) |
| Backprop through the same loops | [`docs/training.md`](../../docs/training.md) |
