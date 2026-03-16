# zero-to-llm

> Building AI from absolute zero — neuron → MLP → Transformer → LLM → Agent.
> Every concept explained line by line. No shortcuts. No CS degree needed.

---

## What is this?

I am learning AI and Machine Learning from scratch and documenting every single step publicly.

Starting from a **single neuron in pure Python** — no PyTorch, no TensorFlow, nothing.
Just numpy and math.

The goal: build a small working LLM and AI agent by the end.

---

## Roadmap

| Phase | Topic | Status |
|-------|-------|--------|
| 01 | Single neuron — forward pass | Done |
| 02 | Backpropagation by hand | Done |
| 03 | Full MLP model class | Done |
| 04 | Rebuild in PyTorch | Coming soon |
| 05 | Transformer from scratch | Coming soon |
| 06 | Train a mini LLM | Coming soon |
| 07 | Build an AI Agent | Coming soon |

---

## How to run

```bash
pip install numpy
python 01-single-neuron/neuron.py
python 02-backprop/backprop.py
python 03-mlp/model.py
```

---

## Folder structure

```
zero-to-llm/
├── README.md
├── 01-single-neuron/
│   ├── README.md
│   └── neuron.py
├── 02-backprop/
│   ├── README.md
│   └── backprop.py
├── 03-mlp/
│   ├── README.md
│   └── model.py
└── blog-posts/
    └── week-01.md
```

---

*Built in public. Learning in public. One neuron at a time.*
