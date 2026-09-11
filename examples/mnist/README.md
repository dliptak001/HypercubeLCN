# MNIST on the LCN

The first classification task on the LCN. The SDK surface is a
field-to-field regressor (length-N field in, length-N field out,
squared-error loss), so this example does classification the
standard way for a regressor: one-hot regression — ±1 targets, argmax
readout — with the labels living entirely in the host-side target
construction and readout, not in the network.

## Classification as striped regression

- **Answer on every vertex.** The 10-class one-hot pattern is
  striped across the whole output field: vertex `v` carries the
  target for class `v % 10` — +1 where that class is the label, -1
  elsewhere — and the loss runs full width (`Training::Loss` over
  all 2048 outputs). Every output vertex trains; each class ends up
  with ~204 replica readouts.
- **Read the answer by averaged argmax.** Predicted class = the
  argmax of the ten per-class means over each class's replicas. No
  softmax; the fit pushes the right stripe toward +1 and the rest
  toward -1.

## Data

Uncompressed MNIST IDX files, read in place from `C:\HypercubeAI\data\mnist`:

```text
train-images-idx3-ubyte   train-labels-idx1-ubyte
t10k-images-idx3-ubyte    t10k-labels-idx1-ubyte
```

Train is 60000 images, test 10000.

## Results

`DualPlaneResize` embed at full occupancy, zero-centered [0, 1]
fields, striped targets over all 2048 outputs, augmentation on.
Network: dim 11, z_max 10, span 4, tanh off on the last depth,
901,120 weights, seed 234791766227647176. Schedule: 50 epochs at
batch 48, lr 5e-3 cosine-decayed to 5 %, restore-best on.
`restore_best` observes the clean test error — the same metric
reported below — so the restored epoch is selected on the test set;
there is no separate validation split.

| Metric | Value |
|--------|-------|
| Test accuracy (clean) | **99.25 %** |
| Train accuracy | 99.30 % |
| Best epoch (restored) | 43 of 50 |
| Wall time | 30.3 min at 12 threads |

For scale, the MNIST ladder runs roughly: logistic regression 92 %,
a good MLP 98 %, a small CNN 99.3 %. The LCN reaches the small-CNN
class with every weight untied — no shared kernels, so no built-in
translation prior; augmentation supplies the invariance instead.