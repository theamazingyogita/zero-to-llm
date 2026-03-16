# Phase 01 — Single Neuron

> Teaching a computer to predict if a student will pass or fail.

## What problem are we solving?

We have 4 students. We know 3 things about each of them.
We want to predict: **will they pass or fail?**

```
Student A → studied 2hrs, slept 8hrs, ate food   → PASS
Student B → studied 1hr,  slept 3hrs, no food     → FAIL
Student C → studied 5hrs, slept 7hrs, ate food    → PASS
Student D → studied 0hrs, slept 2hrs, no food     → FAIL
```

A neuron looks at those 3 numbers and gives us an answer between 0 and 1.
**0 = definitely fail. 1 = definitely pass.**

---

## What is a Neuron?

A neuron is just **3 operations** done in order:

```
1. Multiply each input by its weight     →   weighted sum
2. Add a bias                            →   z
3. Squeeze through sigmoid               →   output (0 to 1)
```

The full formula in one line:

```
output = sigmoid( w1×x1 + w2×x2 + w3×x3 + b )
```

That is the entire neuron. Nothing more.

---

## The Data — x values

```
         x1           x2           x3
      (studied)    (slept)     (ate food)      answer
         |            |            |              |
         v            v            v              v
A  →  [  2    ,      8    ,       1   ]    →    y = 1  (PASS)
B  →  [  1    ,      3    ,       0   ]    →    y = 0  (FAIL)
C  →  [  5    ,      7    ,       1   ]    →    y = 1  (PASS)
D  →  [  0    ,      2    ,       0   ]    →    y = 0  (FAIL)
         ^            ^            ^
         |            |            |
    This is X = our training data
    These numbers are FIXED. We never change them.
    y = the correct answers we already know.
```

---

## The Weights — w values

Weights are the neuron's guess about **how important each input is.**

```
w1 = how much does studying matter?
w2 = how much does sleep matter?
w3 = how much does food matter?
b  = base difficulty (bias)
```

### Where do weights come from?

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   x values  =  DATA       →  you give them                 │
│                               they never change             │
│                                                             │
│   w values  =  MODEL      →  start as random small numbers  │
│                               they CHANGE during training   │
│                               they are what the model learns│
│                                                             │
└─────────────────────────────────────────────────────────────┘

At the START (random guesses):           After 10,000 training steps:
   w1 =  0.0497  (random)                  w1 =  0.54  ← studying matters
   w2 = -0.0138  (random)                  w2 =  0.57  ← sleep matters
   w3 =  0.0648  (random)                  w3 =  2.93  ← food very important!
   b  =  0.0000                            b  = -5.15

The model DISCOVERED these values by seeing the data.
We never told it "studying is important". It figured that out.
```

### Why random and not zero?

```
If all weights start at zero:

   z = 0×2 + 0×8 + 0×1 = 0   (for every student)

   Every student gets the same score.
   Every gradient is the same.
   Every weight update is the same.
   After 10,000 steps: w1 = w2 = w3  (still all equal!)
   The model learns nothing useful.

If weights start random:

   z = 0.05×2 + (-0.01)×8 + 0.06×1 = 0.02  (different per student)

   Each weight gets a different gradient.
   Each weight learns something different.
   w1 learns "studying matters", w2 learns "sleep matters", etc.
```

---

## Step 1 — Weighted Sum

Multiply each input by its weight. Add them all up.

```
For Student A:  x1=2,  x2=8,  x3=1
                w1=0.5, w2=0.3, w3=0.2, b=-0.4

   x1 × w1  =  2 × 0.5  =  1.0
   x2 × w2  =  8 × 0.3  =  2.4
   x3 × w3  =  1 × 0.2  =  0.2
                            ---
   sum                   =  3.6
   + bias                = -0.4
                            ---
   z                     =  3.2    ← this is the "raw score"
```

```
z = w1×x1 + w2×x2 + w3×x3 + b
z = (0.5×2) + (0.3×8) + (0.2×1) + (-0.4)
z = 1.0 + 2.4 + 0.2 - 0.4
z = 3.2
```

`z` can be any number — it could be 100, it could be -50.
We need to squeeze it into 0 to 1.
That is what sigmoid does.

---

## Step 2 — Sigmoid Function

Sigmoid takes any number and squeezes it between 0 and 1.

```
          1
σ(z) = ───────
       1 + e⁻ᶻ
```

### What is e — Euler's Number?

```
Full name  :  Euler's Number
Named after:  Leonhard Euler (Swiss mathematician, 1700s)
Pronounced :  "Oiler" — not "Yooler"
Value      :  e = 2.71828182845904523536...
```

e is not something someone invented. It appears naturally in mathematics — like π = 3.14159...

Just like π always shows up whenever there is a circle,
**e always shows up whenever there is growth or decay.**

It appears in:

- Population growth
- Radioactive decay
- Compound interest
- Neural networks (sigmoid uses e)
- Probability and statistics

Leonhard Euler did not invent e — he discovered it and gave it the name `e` in 1736.
Before him, mathematicians had noticed this number but had no name for it.

**Where does e come from?**

Imagine you put ₹100 in a bank. The bank gives 100% interest per year.

```
Compound once a year:    (1 + 1/1)¹   = 2.000
Compound every 6 months: (1 + 1/2)²   = 2.250
Compound every month:    (1 + 1/12)¹² = 2.613
Compound every day:      (1 + 1/365)³⁶⁵ = 2.714
Compound every second:   (1 + 1/n)ⁿ  → 2.71828...

The more often you compound, the closer you get to e.
e is the limit — the maximum growth possible.
That limit is e = 2.71828...
```

### How to calculate e^x by hand

There is a formula — add up these terms forever (but they get tiny fast):

```
e^x = 1  +  x  +  x²/2  +  x³/6  +  x⁴/24  +  x⁵/120  + ...
          term1  term2     term3     term4      term5
```

**Example: e^1 (just finding the value of e)**

```
Term 1:  1                        =  1.000000
Term 2:  1                        =  1.000000
Term 3:  1² / 2  = 1/2            =  0.500000
Term 4:  1³ / 6  = 1/6            =  0.166667
Term 5:  1⁴ / 24 = 1/24           =  0.041667
Term 6:  1⁵ / 120 = 1/120         =  0.008333
Term 7:  1⁶ / 720                 =  0.001389
Term 8:  1⁷ / 5040                =  0.000198
                                    ---------
               SUM so far         =  2.718254  ≈ e ✓
```

Each term gets smaller and smaller. After ~10 terms you have e.

**Example: e^3.2 (what we need for Student A)**

```
Term 1:  1                              =   1.0000
Term 2:  3.2                            =   3.2000
Term 3:  3.2² / 2  = 10.24 / 2         =   5.1200
Term 4:  3.2³ / 6  = 32.768 / 6        =   5.4613
Term 5:  3.2⁴ / 24 = 104.857 / 24      =   4.3690
Term 6:  3.2⁵ / 120 = 335.54 / 120     =   2.7962
Term 7:  3.2⁶ / 720 = 1073.7 / 720     =   1.4913
Term 8:  3.2⁷ / 5040                   =   0.6816
Term 9:  ...                            =   0.2726
Term 10: ...                            =   0.0969
                                         --------
                         SUM → e^3.2   ≈  24.5325 ✓
```

### Now calculate sigmoid(3.2) by hand

```
         1              1              1
σ(3.2) = ────────  =  ────────  =  ────────  =  0.9608
         1 + e⁻³·²    1 + 1/e³·²   1 + 0.0408

Step by step:
   1.  -z          =  -3.2
   2.  e^(-3.2)    =  1 / e^3.2  =  1 / 24.53  =  0.0408
   3.  1 + 0.0408  =  1.0408
   4.  1 / 1.0408  =  0.9608

σ(3.2) = 0.96 = 96% chance Student A will PASS ✓
```

### Sigmoid for different z values

```
   z = -5   →   σ = 0.007   ≈  0%    definitely FAIL
   z = -3   →   σ = 0.047   ≈  5%    very likely FAIL
   z = -1   →   σ = 0.269   ≈  27%   probably FAIL
   z =  0   →   σ = 0.500   =  50%   no idea (50/50)
   z =  1   →   σ = 0.731   ≈  73%   probably PASS
   z =  3   →   σ = 0.952   ≈  95%   very likely PASS
   z =  5   →   σ = 0.993   ≈  99%   definitely PASS

No matter how big or small z is — sigmoid always gives 0 to 1.
That is its only job.
```

---

## Step 3 — Loss (How Wrong Are We?)

We compare our prediction to the real answer.

```
Loss formula:  L = (prediction - real answer)²

Why square it?
   If pred=0.7, y=1:   0.7 - 1.0 = -0.3  →  (-0.3)² = 0.09
   If pred=0.3, y=1:   0.3 - 1.0 = -0.7  →  (-0.7)² = 0.49
   Squaring removes the negative sign so all errors are positive.
   Big errors get punished more (0.7² = 0.49 vs 0.3² = 0.09).
```

```
For all 4 students (before training, step 0):

   Student A:  pred=0.51, y=1  →  (0.51-1.0)² = 0.2401
   Student B:  pred=0.50, y=0  →  (0.50-0.0)² = 0.2500
   Student C:  pred=0.55, y=1  →  (0.55-1.0)² = 0.2025
   Student D:  pred=0.49, y=0  →  (0.49-0.0)² = 0.2401

   Average = (0.2401 + 0.2500 + 0.2025 + 0.2401) / 4
           = 0.9327 / 4
           = 0.2332  ← total wrongness at step 0

After 10,000 training steps:
   All predictions close to correct → loss = 0.001 ✓
```

---

## Step 4 — Backpropagation (Fixing the Weights)

We know the loss. Now we need to find:
**"Which weight caused this mistake? By how much?"**

The answer is the **Chain Rule** from calculus.

```
dL/dW = dL/dpred  ×  dpred/dz  ×  dz/dW
           │               │           │
           │               │           └──→ = x  (just the input value!)
           │               └──────────────→ = pred × (1 - pred)
           └──────────────────────────────→ = 2 × (pred - y) / n
```

### What does d mean?

`dL/dW` means "if I change W by a tiny amount, how much does L change?"

```
Think of it like a hill:
   - If you are on a hill and take a tiny step right, do you go up or down?
   - The gradient tells you the direction of uphill.
   - We want to go DOWNHILL to reduce loss.
   - So we move weights OPPOSITE to the gradient.
```

### Piece 1 — dL/dpred

"How does loss change when prediction changes?"

```
Loss = (pred - y)²

Derivative = 2 × (pred - y) / n

Example: pred=0.51, y=0, n=4
   = 2 × (0.51 - 0) / 4
   = 2 × 0.51 / 4
   = 0.255

Positive → prediction is too HIGH → needs to go DOWN
Negative → prediction is too LOW  → needs to go UP
```

### Piece 2 — dpred/dz

"How does prediction change when z changes?"

```
pred = sigmoid(z) = 1 / (1 + e^-z)

The derivative of sigmoid is beautiful:
   dpred/dz = pred × (1 - pred)

Example: pred = 0.51
   = 0.51 × (1 - 0.51)
   = 0.51 × 0.49
   = 0.2499

Why pred×(1-pred)?
   When pred=0.5  →  0.5×0.5 = 0.25  (maximum — very uncertain, big change)
   When pred=0.99 →  0.99×0.01 = 0.009 (tiny — already confident, small change)
   When pred=0.01 →  0.01×0.99 = 0.009 (tiny — already confident other way)
```

### Piece 3 — dz/dW

"How does z change when W changes?"

```
z = w1×x1 + w2×x2 + w3×x3 + b

If we change w1 by a tiny amount:
   z changes by exactly x1

So dz/dw1 = x1
   dz/dw2 = x2
   dz/dw3 = x3
   dz/db  = 1
```

### Chain Rule — all together

```
Gradient for w1 = Piece1 × Piece2 × Piece3
                = 2×(pred-y)/n  ×  pred×(1-pred)  ×  x1

Example for Student B: pred=0.50, y=0, x1=1, n=4

   Piece 1 = 2×(0.50-0)/4    = 0.250
   Piece 2 = 0.50×(1-0.50)   = 0.250
   Piece 3 = x1               = 1.0

   gradient for w1 = 0.250 × 0.250 × 1.0 = 0.0625
```

### Weight Update

```
w_new = w_old - learning_rate × gradient

learning_rate = 0.1  (how big each step is)

w1_new = 0.0497 - 0.1 × 0.0625
w1_new = 0.0497 - 0.00625
w1_new = 0.0435  ← w1 moved!

┌─────────────────────────────────────────────────────┐
│  gradient positive → weight goes DOWN               │
│  gradient negative → weight goes UP                 │
│  always moving to reduce the loss                   │
│  this is called gradient descent                    │
└─────────────────────────────────────────────────────┘
```

---

## Step 5 — The Training Loop

Do steps 1–4 over and over. Each time, loss gets smaller.

```
┌──────────────────────────────────────────────────┐
│                                                  │
│  REPEAT 10,000 TIMES:                            │
│                                                  │
│   1. z    = X @ W + b          ← forward pass   │
│   2. pred = sigmoid(z)         ← prediction     │
│   3. loss = mean((pred-y)²)    ← how wrong?     │
│   4. delta = chain rule        ← backprop       │
│   5. W = W - lr × gradient     ← fix weights    │
│                                                  │
└──────────────────────────────────────────────────┘

What you see in the terminal:

   step      0  loss=0.2327  preds=[0.51, 0.50, 0.55, 0.49]  ← random
   step   2000  loss=0.0130  preds=[0.89, 0.18, 0.97, 0.08]  ← learning
   step   4000  loss=0.0027  preds=[0.95, 0.08, 0.98, 0.03]  ← better
   step   6000  loss=0.0017  preds=[0.96, 0.07, 0.99, 0.02]  ← good
   step   8000  loss=0.0013  preds=[0.96, 0.06, 0.99, 0.02]  ← great
   step  10000  loss=0.0010  preds=[0.97, 0.05, 0.99, 0.02]  ← done!
                                       ↑     ↑     ↑     ↑
                                     PASS  FAIL  PASS  FAIL  ← all correct
```

---

## Step 6 — Saving the Model

```
PROBLEM:
   After training, W lives in Python memory (RAM).
   Close Python → W is deleted → next user has to retrain 10,000 steps again.
   That is useless.

SOLUTION:
   np.save("model_W.npy", W)   ← writes W to a file on disk
   np.save("model_b.npy", b)   ← writes b to a file on disk

   These files stay on disk forever.
   Any user can load them:

   W = np.load("model_W.npy")   ← reads W back instantly
   predict(new_student)          ← no training needed

FILE SIZE:
   Our neuron   = 152 bytes    (3 weights + 1 bias)
   GPT-2        = 548 MB       (117 million weights)
   GPT-4        = ~100s of GB  (~1 trillion weights)

   Same exact idea. Just different scale.
   The .npy file IS the brain of the model.
```

---

## The Code — Every Single Line Explained

```python
import numpy as np
```

Bring in the numpy library — our maths tool.
`as np` = short name so we type `np` instead of `numpy` every time.

---

```python
X = np.array([
    [2, 8, 1],
    [1, 3, 0],
    [5, 7, 1],
    [0, 2, 0],
], dtype=float)
```

Our training data. 4 rows = 4 students. 3 columns = 3 features.
`dtype=float` = store as decimal numbers (2.0 not 2).

---

```python
y = np.array([[1],[0],[1],[0]], dtype=float)
```

Correct answers. 1=PASS, 0=FAIL. Shape (4,1) = 4 rows, 1 column.

---

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

The sigmoid function. `np.exp(-z)` = e to the power of -z.
Python knows e = 2.71828 already — that is what `np.exp` uses.

---

```python
np.random.seed(42)
W = np.random.randn(3, 1) * 0.1
b = 0.0
```

`seed(42)` = makes random numbers the same every run (so you get my results).
`randn(3, 1)` = 3 random numbers in a column shape.
`* 0.1` = keep them small (like 0.05, -0.01, 0.06).
`b = 0.0` = bias starts at zero.

---

```python
lr = 0.1
```

Learning rate = how big each weight update step is.
Too big (like 5.0) → weights bounce around, never learn.
Too small (like 0.00001) → learns but takes forever.
0.1 is a safe starting point.

---

```python
for step in range(10001):
```

Repeat everything inside 10,001 times. `step` counts 0, 1, 2 ... 10000.

---

```python
    z    = X @ W + b
    pred = sigmoid(z)
```

`@` = matrix multiply. X is (4×3), W is (3×1), result is (4×1).
One z value per student. Then sigmoid gives one prediction per student.

---

```python
    loss = np.mean((pred - y) ** 2)
```

Subtract real answer from prediction. Square it. Average all 4 students.
One number — our total wrongness. We want this to shrink every step.

---

```python
    n      = X.shape[0]
    delta  = 2*(pred - y)/n * pred*(1 - pred)
    dL_dW  = X.T @ delta
    dL_db  = np.sum(delta)
```

`X.shape[0]` = number of rows = 4 students.
`delta` = Piece1 × Piece2 of chain rule combined into one line.
`X.T @ delta` = matrix multiply transposed X with delta = gradient for W.
`np.sum(delta)` = add up all deltas = gradient for b.

---

```python
    W -= lr * dL_dW
    b -= lr * dL_db
```

**The most important lines in all of machine learning.**
Move W opposite to gradient. Gradient = uphill. We go downhill.
`W -= lr * dL_dW` means `W = W - (0.1 × gradient)`.
This is gradient descent.

---

```python
    if step % 2000 == 0:
        print(f"  step {step}  loss = {loss:.6f}")
```

`%` = remainder. `step % 2000 == 0` = "is step divisible by 2000?"
So it prints only at 0, 2000, 4000, 6000, 8000, 10000.
`:.6f` = show exactly 6 decimal places.

---

## Final Results

```
Student A: [███████████████████░] 0.97  →  PASS  ✓
Student B: [█░░░░░░░░░░░░░░░░░░░] 0.05  →  FAIL  ✓
Student C: [███████████████████░] 0.99  →  PASS  ✓
Student D: [░░░░░░░░░░░░░░░░░░░░] 0.02  →  FAIL  ✓

New student — studied 3hrs, slept 6hrs, ate food:
→ PASS  (94.4% confident)
```

---

## Key Takeaways

```
1. A neuron = multiply → sum → sigmoid. Three operations.

2. x = data      → FIXED. You give it. Never changes.
   w = weights   → start random, LEARNED during training.
   b = bias      → base score, also learned.

3. e = 2.71828   → natural number, appears in sigmoid formula.
   sigmoid       → squeezes any number to 0-1.

4. loss          → how wrong we are. We want it small.

5. chain rule    → dL/dW = piece1 × piece2 × piece3
   tells us which direction to move each weight.

6. gradient descent → W = W - lr × gradient
   move downhill, loss decreases, model learns.

7. .npy file     → the brain. Save it. Share it. Load it.
   users never need to retrain.
```

---

## How to Run

```bash
pip install numpy
python neuron.py
```

---

*Week 1 complete. Next: Backpropagation in detail — watching every gradient.*