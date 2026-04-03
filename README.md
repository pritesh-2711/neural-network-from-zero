# Deep Learning Basics with Python

A self-contained study path from math fundamentals to Transformers, using NumPy and PyTorch.

---

## Prerequisites

```text
numpy matplotlib torch
```

---

## Study Path

Work through the notebooks in order. Each one builds on the last.

---

### Stage 0 — PyTorch Reference

**[torch/pytorch_deeplearning_complete.ipynb](torch/pytorch_deeplearning_complete.ipynb)**

A reference notebook to keep open while working through the rest. Covers:

- Tensors: shapes, dtypes, device placement
- Tensor operations: matrix math, broadcasting
- Reshaping, indexing, slicing
- Automatic differentiation (`autograd`)
- `nn.Module`, loss functions, optimizers
- Training loop skeleton
- `Dataset` + `DataLoader`
- GPU usage, mixed precision
- Custom layers and custom autograd functions
- Common architectures, debugging tips

You do not need to read this front-to-back first. Use it as a lookup when you hit something unfamiliar.

---

### Stage 1 — The Math

**[maths/00_math_concepts_nn.ipynb](maths/00_math_concepts_nn.ipynb)**

Builds the minimum math you need before writing any code.

| Section | What you learn |
| ------- | -------------- |
| Scalars, vectors, matrices | Shape tracking; `z = Wx + b` |
| Dot product | Weighted sum as neuron pre-activation |
| Sigmoid and ReLU | Formulas, derivatives, saturation behavior |
| Loss functions | MSE and binary cross-entropy with derivatives |
| Chain rule | The entire secret of backprop |
| Manual backprop | `dL/dw` and `dL/db` for a single neuron, by hand |
| Numerical gradient check | Finite differences as a sanity check |
| Loss landscape | Visualize loss vs a single weight |

**Goal:** you can compute gradients by hand for a single neuron and explain what gradient descent is doing.

---

### Stage 2 — MLP from Scratch (NumPy only)

**[scratchpad/01_numpy_mlp_from_scratch.ipynb](scratchpad/01_numpy_mlp_from_scratch.ipynb)**

Builds a 2-layer MLP using only NumPy. No autograd. Every gradient is written by hand.

| Section | What you learn |
| ------- | -------------- |
| XOR dataset | Why it is a hard problem for linear models |
| Logistic regression baseline | Proves linear models fail on XOR |
| Forward pass | `z1 = XW1 + b1`, `h = sigmoid(z1)`, `p = sigmoid(hW2 + b2)` |
| Backprop | Manual chain rule through two layers |
| Training loop | Gradient descent step-by-step |
| Diagnostics | Gradient norms, activation saturation |
| Numerical gradient check | Verify your analytic gradients |

**Goal:** you can implement backprop without autograd and know what "vanishing gradients" looks like in practice.

---

### Stage 3 — PyTorch Autograd (bare-metal)

**[scratchpad/02_pytorch_bare_metal_autograd.ipynb](scratchpad/02_pytorch_bare_metal_autograd.ipynb)**

Rebuilds the same XOR MLP in PyTorch, but without `nn.Module` or `torch.optim` — just tensors and `.backward()`.

| Section | What you learn |
| ------- | -------------- |
| `requires_grad=True` | How PyTorch tracks computation |
| `.backward()` + manual SGD | Gradient flow without an optimizer object |
| `torch.no_grad()` | When and why to disable grad tracking |
| Gradient diagnostics | Grad norms and saturation in PyTorch |
| Finite-difference check | Verify autograd against numerical gradients |
| Device placement | CPU vs GPU, all-on-same-device rule |
| `nn.Module` + `optim` | The standard approach — earned after understanding the low level |

**Goal:** you understand what `.backward()` actually does and can debug gradient issues.

---

### Stage 4 — Proper PyTorch Engineering

**[scratchpad/03_pytorch_engineering_training_pipeline.ipynb](scratchpad/03_pytorch_engineering_training_pipeline.ipynb)**

Same tiny model, but with production-quality scaffolding. This is the reusable skeleton for every future project.

| Section | What you learn |
| ------- | -------------- |
| `seed_everything` | Reproducibility (CPU + CUDA) |
| `@dataclass` Config | Single place for all hyperparameters |
| `Dataset` + `DataLoader` | Proper batching pattern |
| `nn.Module` model | `BCEWithLogitsLoss` (numerically stable) |
| Weight initialization | Xavier for tanh; Kaiming for ReLU |
| `train_one_epoch` / `eval_one_epoch` | `model.train()` vs `model.eval()`, correct zero-grad placement |
| Gradient clipping | `clip_grad_norm_` |
| LR scheduler | `StepLR` |
| AMP mixed precision | `torch.amp.autocast` + `GradScaler` |
| Checkpointing | Save/load model + optimizer + scheduler state |
| Common failure modes | Checklist of things that go wrong |

**Goal:** you have a training skeleton you can copy into any new project.

---

### Stage 5 — Deep Networks: Stability and Scaling

**[scratchpad/04_depth_stability_scaling.ipynb](scratchpad/04_depth_stability_scaling.ipynb)**

Goes from 2 layers to 20 layers on a two-moons classification task. Runs four controlled experiments.

| Section | What you learn |
| ------- | -------------- |
| Two-moons generator | Nonlinear boundary; harder than XOR |
| Train/val split | Detecting overfitting |
| Deep MLP builder | Configurable depth, width, activation, norm, residual |
| Activation stats hook | Forward hook to track mean/std/min/max per layer |
| Experiment A: naive baseline | What goes wrong with a deep net out of the box |
| Experiment B: Kaiming + grad clip | How proper init and clipping stabilize training |
| Experiment C: LayerNorm | Normalizing activations across the network |
| Experiment D: Residual connections | Skip connections enable much deeper nets |
| Regularization (dropout + weight decay) | Controlling overfitting |
| Decision boundary plot | Visualize what the network learned |

**Goal:** you can diagnose a failing deep net and know which fixes to try first.

---

### Stage 6 — Convolutional Neural Networks

**[scratchpad/05_cnn_convolutions_and_training.ipynb](scratchpad/05_cnn_convolutions_and_training.ipynb)**

Starts from the convolution operation itself, then builds and trains a CNN classifier.

| Section | What you learn |
| ------- | -------------- |
| Naive `conv2d` (NumPy) | What convolution actually computes; verified against PyTorch |
| Stride and padding shapes | Formula: `out = (in + 2p - k) / s + 1` |
| Synthetic image dataset | 4-class (vertical / horizontal / diagonal / plus) 28×28 images |
| `SimpleCNN` | `Conv2d → ReLU → MaxPool → Linear` |
| Training + metrics | CrossEntropyLoss, AdamW |
| Feature map visualization | See what conv1 filters respond to |
| CNN vs MLP comparison | Weight sharing gives better generalization on images |

**Goal:** you understand what a convolution is and why CNNs outperform MLPs on spatial data.

---

### Stage 7 — Sequence Models and Transformers

**[scratchpad/06_sequence_models_attention_transformer.ipynb](scratchpad/06_sequence_models_attention_transformer.ipynb)**

Progression from RNN to Transformer on a next-token prediction task.

| Section | What you learn |
| ------- | -------------- |
| RNN (manual cell) | `h_t = tanh(Wx * x_t + Wh * h_{t-1})` |
| GRU | Gating as a fix for vanishing gradients over long sequences |
| Scaled dot-product attention | `softmax(QK^T / sqrt(d)) V` |
| Causal mask | Why autoregressive models need it |
| Multi-head self-attention | Parallel attention heads |
| `TransformerBlock` | Pre-LN: `x + Attn(LN(x))`, `x + FF(LN(x))` |
| Positional embeddings | Why Transformers need explicit position information |
| Attention visualization | Lower-triangular structure of causal attention |
| No-position ablation | What breaks when you remove positional embeddings |
| RNN vs GRU vs Transformer | Side-by-side accuracy comparison |

**Goal:** you can implement a tiny Transformer from scratch and explain why each component exists.

---

## What each stage unlocks

```text
Stage 0  PyTorch tensors and APIs — reference throughout
Stage 1  Math intuition — gradient descent makes sense
Stage 2  Backprop — you own it, not just autograd
Stage 3  Autograd — confident PyTorch debugging
Stage 4  Training pipeline — reusable skeleton for real projects
Stage 5  Deep networks — you can stabilize training
Stage 6  CNNs — image data
Stage 7  Transformers — sequence data, attention, modern architectures
```
