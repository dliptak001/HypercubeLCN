# Forward

**Status: Core.cpp as of 1.2.0. Weights are packed depth, axis, tap, vertex.**

## Definitions

| Symbol / term | Meaning |
|---------------|---------|
| Core | The network: owns the weights and runs the forward pass. |
| dim | Hypercube dimension. Valid range [4, 24]. |
| N, n | Vertex count, N = 2ᵈⁱᵐ. Also the length of one field. |
| v | Vertex address, a dim-bit integer 0 .. N−1. |
| axis | One of the dim bit positions. Flipping it walks to a neighbor. |
| neighbor, v_nn | The vertex reached by v XOR (1 << axis). Every vertex has dim neighbors. |
| field | N floats, one per vertex, laid out by address. Input, output, and every intermediate are fields. |
| depth, z | One gather-and-write over the cube. z runs 0 .. z_max−1. |
| z_max | Depth count. Config 0 means use dim. Must be ≥ 2 otherwise. |
| gather_span, span | Lookback window width in fields. Valid range [2, 6]. |
| tap, k | One slot in the window. Tap k at depth z reads slot z+k. k runs 0 .. span−1. |
| s_ | History stack: (z_max + span) consecutive slots of N floats. |
| slot | One field in s_. Slot t holds s[t·N + v] for each vertex v. |
| prefix | The first span−1 slots, zeroed at construction and never written again. |
| w_ | All trainable weights. Layout depth, axis, tap, vertex. Length N × dim × span × z_max. |
| TapOffset | Base of the N-float table for (z, axis, k): ((z·dim + axis)·span + k)·N. Vertex is the fastest index. |
| o_ | Output field of the most recent Forward: a copy of the last slot written. |
| tanh_last | If false (default), the last depth writes the raw accumulator instead of tanh. |
| acc | Pre-activation at one vertex: the weighted sum of neighbor taps before tanh. |
| seed | RNG seed for the initial weight draw (normal, Xavier-style scale). |
| fan-in | Number of products in one accumulator: dim × span. |

One field goes in, one field comes out. In between, the network writes a
stack of new fields on the same hypercube, one per depth. Each depth
looks at a short window of recently written fields, mixes each vertex's
neighbors through that vertex's own weights, and writes the next
field. The last field written is the output.

This note follows [Core.cpp](../Core.cpp). Training — how the weights
get their values — is [training.md](training.md). The worked examples
use dim = 4 (16 vertices) so you can check every bit by hand; the
constructor accepts dim 4 through 24, and the Raman example runs
dim = 11.

## The cube: vertices are bit addresses

A dim-dimensional hypercube has n = 2ᵈⁱᵐ vertices. There is no
adjacency list, because the address *is* the geometry: label each vertex
with a dim-bit integer, and two vertices are neighbors exactly when
their labels differ in one bit. To walk to a neighbor, flip a bit:

```
v_nn = v XOR (1 << axis)
```

Every vertex has exactly dim neighbors, one per axis. Take dim = 4
and vertex 11, which is `1011` in binary:

| axis | mask   | neighbor | neighbor bits |
|-----:|-------:|---------:|---------------|
| 0    | `0001` | 10       | `1010` |
| 1    | `0010` | 9        | `1001` |
| 2    | `0100` | 15       | `1111` |
| 3    | `1000` | 3        | `0011` |

A **field** is one float per vertex — n floats laid out by address.
The input is a field, the output is a field, and everything the network
writes in between is a field.

Two properties of this graph shape everything that follows. First, it
is sparse: a vertex sees only its dim neighbors, never the whole
cube, and there is no self-edge — a vertex cannot read its own value
directly. Second, it is small in diameter: any two vertices differ in
at most dim bits, so information can cross the entire cube in dim
neighbor-hops. Depth buys hops.

## The history stack

The network keeps its fields in one flat buffer s_, laid out as
consecutive slots of n floats:

```
s[t * n + v]     vertex v at slot t
```

The stack has z_max + gather_span slots, and its structure is the
one genuinely clever piece of bookkeeping in the design:

- The first gather_span − 1 slots are a **zero prefix**. They are
  zeroed when the network is built and nothing ever writes them again.
- The input field is copied into slot gather_span − 1, right after
  the prefix.
- Depth z writes slot z + gather_span. The last depth writes the
  last slot, which Forward copies into o_ (readable via
  `Output()`).

With gather_span = 2 (abbreviated *span* from here on) and
z_max = 4:

```
slot 0   zeros, forever            (prefix)
slot 1   input
slot 2   written by depth 0
slot 3   written by depth 1
slot 4   written by depth 2
slot 5   written by depth 3  →  copied to o_
```

Why the zero prefix exists: every depth reads a window of span
consecutive slots (next section), and the earliest depths would
otherwise reach past the beginning of time. Instead of clipping the
window or giving early depths a different table shape, the stack simply
starts with slots that say "before the input there was nothing." Every
depth then has the same window width, the same table shape, and the
same loop — the special case is data, not code.

Note also what Forward does *not* do: it never clears the stack.
Every non-prefix slot is fully rewritten on each call — the input slot
from the argument, each later slot by its depth — so consecutive calls
are independent. Each Forward is a fresh sample; nothing persists
between calls except the weights.

## The gather window

Depth z reads slots z through z + span − 1 and writes slot
z + span. Tap k (for k = 0 .. span−1) reads slot z + k, so the
newest tap reads the slot that depth z − 1 just finished, and the
oldest tap reads span − 1 slots further back.

The window is the same width at every depth. What changes is what it
lands on:

| z   | span 2 reads   | span 3 reads          |
|-----|----------------|-----------------------|
| 0   | prefix, input  | prefix, prefix, input |
| 1   | input, slot 2  | prefix, input, slot 3 |
| 2   | slots 2, 3     | input, slots 3, 4     |
| 3+  | slots z, z+1   | slots z, z+1, z+2     |

Two things to read off this table. Early depths waste taps on the
prefix — those taps multiply zeros and contribute nothing. And after
the first few depths the window **slides off the input**: from
z = span onward, no depth sees the raw input field again. The bet is
that the input has already been mixed into the recent fields by then,
and later depths work from that mixture. A wider span keeps the input
visible longer and gives every depth more temporal context, at a
proportional cost in weights.

span is `CoreConfig.gather_span`, valid from 2 to 6.

## Private tables

Every vertex owns its own weight at every (depth, axis, tap), and
nothing is shared — the only coupling between vertices is the gather
itself, where a vertex reads its neighbors' history. Each accumulator
sums dim axes × span taps products.

Uniform shape makes the packing pure arithmetic. w_ is laid out
depth, then axis, then tap, then vertex — vertex is the fastest index,
so the N weights of one tap sit in a contiguous block:

```
w index = ((z * dim + axis) * span + k) * n  +  v
|w_|    = n * dim * span * z_max
TapOffset(z, axis, k) = ((z * dim + axis) * span + k) * n
```

For the Raman configuration (dim = 11, n = 2048, span = 4,
z_max = 8) that is 720 896 floats — the `|w|` printed in the run
banner.

At construction every weight is drawn from one normal distribution:
mean 0, standard deviation 1 / √(dim × span). That is Xavier-style
scaling for a fan-in of dim × span — each accumulator sums exactly
that many products, so the pre-activation lands near unit scale where
tanh is neither saturated nor trivially linear. The draw is seeded by
`CoreConfig.seed`, so a given config always starts from the same
weights. The generator fills w_ in storage order, so a layout change
is a different net at the same seed.

Two honest consequences of keeping every table the same shape:

- **Dead weights.** A tap that lands in the zero prefix multiplies zero
  forever. Across the early depths that is
  n × dim × span × (span−1) / 2 weights — a fraction
  (span−1) / (2 × z_max) of the total (about 19% for the Raman
  config). They are inert, not just redundant: their gradient is
  exactly zero, so training leaves them frozen at their random initial
  values and no special handling is needed.
- **Depth 0 starts cool.** Its live fan-in is only dim (the other
  taps read the prefix), but its weights are scaled for dim × span,
  so its output starts a factor √span small. Accepted: it sits
  in tanh's linear region and is one depth out of many.

## One vertex at one depth

Everything above assembles into a small kernel. To compute the new
value at vertex v, depth z:

1. `acc = 0`.
2. For each axis, flip that bit to get the neighbor
   v_nn = v XOR (1 << axis); for each tap k, add
   `w[TapOffset(z, axis, k) + v] * s[(z + k) * n + v_nn]`.
3. Write `s[(z + span) * n + v] = tanh(acc)`.

Each (axis, tap) pair has its own weight — the network does not sum a
neighbor's window first and then apply one scale; it weighs every slot
of every neighbor independently. tanh runs once per vertex, after the
whole sum, and there is no bias term.

The one exception is the last depth when `CoreConfig.tanh_last` is
false (the default): it writes acc raw instead of tanh(acc). A
tanh output is trapped in (−1, 1); a regression target normalized to
roughly unit scale can still exceed that, so squared-error training
wants the final depth linear (the Raman baselines and the MNIST run
all use it that way). Set tanh_last true only to confine the output
to (−1, 1) like every other depth.

Within one depth, order does not matter: every vertex reads only
finished slots (z + span − 1 and older) and writes only its own entry
of slot z + span, so vertices never see each other's new values.
Across depths, order does matter — depth z needs depth z − 1's slot
— which is why z runs upward and the stack is written left to right.

The code does not walk vertex-outer. It zeros the output field, then
for each (axis, tap) adds a length-N strip of contiguous weights
times a bit-flipped read of the source slot, then applies tanh as a
field pass. Same sum; vertex innermost so the weights of one tap are
adjacent.

## Worked example: dim 4, span 2, vertex 11

n = 16, z_max = 4 (the z_max = 0 default resolves to dim).
Six slots: prefix, input, four written. Vertex 11, neighbors
10, 9, 15, 3 from the table above. Writing w[axis][k] for vertex
11's weights at the current depth:

**Depth 0** reads slots 0 and 1. Slot 0 is the prefix, so every k = 0
tap multiplies zero; only the input taps land.

```
acc  = w[0][0]*0 + w[0][1]*s[1][10]
     + w[1][0]*0 + w[1][1]*s[1][9]
     + w[2][0]*0 + w[2][1]*s[1][15]
     + w[3][0]*0 + w[3][1]*s[1][3]

s[2][11] = tanh(acc)
```

**Depth 1** reads slots 1 and 2 — the input and the field depth 0 just
wrote. Both taps are live from here on.

```
acc  = w[0][0]*s[1][10] + w[0][1]*s[2][10]
     + w[1][0]*s[1][9]  + w[1][1]*s[2][9]
     + w[2][0]*s[1][15] + w[2][1]*s[2][15]
     + w[3][0]*s[1][3]  + w[3][1]*s[2][3]

s[3][11] = tanh(acc)
```

**Depth 2** reads slots 2 and 3: the window has slid off the input.
**Depth 3** reads slots 3 and 4 and writes slot 5, the output — raw if
tanh_last is false.

Notice what vertex 11 never reads: itself. Its own input s[1][11]
reaches it only by a round trip — a neighbor folds it into slot 2, and
vertex 11 reads that neighbor at depth 1. This is why the constructor
requires z_max ≥ 2: with a single depth the output at a vertex could
not depend on that vertex's own input at all. At the other end of the
ladder, z_max = dim is the cube's diameter — the depth at which even
antipodal vertices (all bits different) can influence one another.
Between those bounds, depth is a fit knob: the task decides how much
reach is worth paying for.

## Configuration summary

From `CoreConfig` in [Core.h](../Core.h), enforced in the constructor:

| Knob | Default | Constraint | Meaning |
|------|---------|------------|---------|
| dim | 8 | 4 .. 24 | Address bits; n = 2ᵈⁱᵐ vertices |
| seed | fixed | — | RNG seed for the weight draw |
| z_max | 0 | 0, or ≥ 2 | Number of depths; 0 means "use dim" |
| gather_span | 2 | 2 .. 6 | Window width in slots |
| tanh_last | false | — | If true, the last depth applies tanh like the rest |

Sizes that follow: s_ is (z_max + span) × n floats, w_ is
z_max × dim × span × n floats, o_ is n floats.

## The loop nest

The whole of Forward, in the order the code runs it:

```
s[span - 1][v] = input_field[v]        for all v
                                       (prefix slots stay zero; every
                                        other slot is about to be written)

for z = 0 .. z_max-1:
    out = s[z + span]                  (N floats; zeroed)
    for axis = 0 .. dim-1:
        mask = 1 << axis
        for k = 0 .. span-1:
            src = s[z + k]             (N floats)
            wv  = w + TapOffset(z, axis, k)
            for v = 0 .. n-1:
                out[v] += wv[v] * src[v XOR mask]
    if not (last depth and tanh_last false):
        out[v] = tanh(out[v])          for all v

o[v] = s[z_max + span - 1][v]          for all v
```

[training.md](training.md) walks this same nest in reverse.
