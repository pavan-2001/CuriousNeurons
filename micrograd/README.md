# micrograd (practice)

A practice re-implementation of Andrej Karpathy's
[micrograd](https://github.com/karpathy/micrograd): a tiny scalar-valued
autograd engine plus a small neural-net library built on top of it.

Built purely for learning: to understand how forward passes, the
computation graph, backprop, and SGD fit together from scratch.

## Files

| File | What it contains |
|---|---|
| `engine.py` | `Value` : a scalar that records every op into a DAG and supports `backward()`. |
| `nn.py`     | `Neuron`, `Layer`, `MLP` built on top of `Value`. |
| `demo.py`   | Runnable example: trains a small MLP and visualizes it. |

## Run the demo

From the folder that **contains** `micrograd/`:

```bash
python -m micrograd.demo
```

## What `demo.py` does

1. Builds an MLP: `MLP(3, [4, 4, 1])` : 3 inputs, two ReLU hidden layers, 1 linear output.
2. Defines a tiny dummy dataset (4 samples, targets in `{-1, +1}`).
3. Trains for 100 SGD steps on MSE loss and prints the loss every 10 steps.
4. Shows three plots:
   - **Training loss** over steps.
   - **MLP architecture** (input / hidden / output layers).
   - **Computation graph** of a small expression, with each node's `data` and `grad`
     after `backward()` , the picture of what autograd actually walks.

## Dependencies

`matplotlib`, `networkx` (only used by `demo.py` for the plots).
