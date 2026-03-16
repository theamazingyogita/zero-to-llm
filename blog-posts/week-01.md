# Week 1 — I built a neuron from scratch. Here is what I learned.

I started learning AI from zero this week. No PyTorch, no TensorFlow. Just Python and numpy.

## What even is a neuron?

A neuron takes numbers in, multiplies each by a weight, adds a bias, then squeezes the result through sigmoid.

```
output = sigmoid( w1*x1 + w2*x2 + w3*x3 + b )
```

That is literally it.

## My example

I built a neuron that predicts if a student will pass an exam based on:
- Hours studied
- Hours slept
- Whether they ate food

The neuron starts with random weights — it knows nothing. After training on 4 students for 10,000 steps, it predicts correctly every time.

## The moment it clicked

Weights are just guesses that get corrected over time.

`x` = your data (fixed — you give it)
`w` = weights (start random, learned by training)

Every training step = make a prediction → measure how wrong → fix weights a tiny bit → repeat.

## What I built

A working neuron in 40 lines of Python. No libraries except numpy.

Loss started at 0.232 (basically random guessing). After 10,000 steps: 0.001.

See the code: [github.com/yourusername/zero-to-llm/01-single-neuron](https://github.com)

## Next week

Backpropagation — the chain rule that makes weights actually learn.
