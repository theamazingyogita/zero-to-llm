# Phase 01 — Single Neuron

## What is a neuron?

A neuron takes inputs, multiplies each by a weight, adds a bias, then squeezes through sigmoid.

```
output = sigmoid( w1*x1 + w2*x2 + w3*x3 + b )
```

## Our example

Predict if a student will pass or fail based on:
- x1 = hours studied
- x2 = hours slept
- x3 = ate food (1=yes, 0=no)

## Key concepts learned

- `x` values = DATA — fixed, given to us
- `w` values = WEIGHTS — start random, learned by training
- `b` = BIAS — base score before any inputs
- `sigmoid` = squeezes any number into 0 to 1
- `loss` = how wrong our prediction is
- `forward pass` = input → prediction
- `backward pass` = prediction → fix weights

## How to run

```bash
pip install numpy
python neuron.py
```

## What the output looks like

```
step      0  loss = 0.232  predictions = [0.51, 0.50, 0.55, 0.49]
step   2000  loss = 0.013  predictions = [0.89, 0.18, 0.97, 0.08]
step  10000  loss = 0.001  predictions = [0.97, 0.05, 0.99, 0.02]
```

Loss drops from 0.232 to 0.001 — the neuron learned!
