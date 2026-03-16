import numpy as np

# ================================================================
#  PHASE 01 — SINGLE NEURON FROM SCRATCH
#
#  Task: predict if a student passes or fails
#  Input:  x1=hours studied, x2=hours slept, x3=ate food
#  Output: 1=pass, 0=fail
#
#  Formula: output = sigmoid(w1*x1 + w2*x2 + w3*x3 + b)
# ================================================================


# ── STEP 1: DATA ─────────────────────────────────────────────────
# X = inputs  (4 students, 3 features each)
# y = correct answers

X = np.array([
    [2, 8, 1],   # studied 2hrs, slept 8, ate food  → PASS
    [1, 3, 0],   # studied 1hr,  slept 3, no food   → FAIL
    [5, 7, 1],   # studied 5hrs, slept 7, ate food  → PASS
    [0, 2, 0],   # studied 0hrs, slept 2, no food   → FAIL
], dtype=float)

y = np.array([[1], [0], [1], [0]], dtype=float)


# ── STEP 2: SIGMOID FUNCTION ─────────────────────────────────────
# Squeezes any number into 0 to 1
# Formula: 1 / (1 + e^(-z))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ── STEP 3: STARTING WEIGHTS ─────────────────────────────────────
# Random small numbers — just guesses to start

np.random.seed(42)
W = np.random.randn(3, 1) * 0.1   # shape (3,1) — one weight per feature
b = 0.0                            # bias starts at zero
lr = 0.1                           # learning rate — how big each fix step is


# ── STEP 4: TRAINING LOOP ────────────────────────────────────────
print("=" * 55)
print("  TRAINING")
print("=" * 55)
print(f"  {'Step':>6}   {'Loss':>10}   Predictions")
print("  " + "-" * 50)

for step in range(10001):

    # Forward pass — make predictions
    z    = X @ W + b          # weighted sum
    pred = sigmoid(z)         # squeeze to 0-1

    # Loss — how wrong are we?
    loss = np.mean((pred - y) ** 2)

    # Backward pass — chain rule
    n      = X.shape[0]
    delta  = 2*(pred - y)/n * pred*(1 - pred)
    dL_dW  = X.T @ delta
    dL_db  = np.sum(delta)

    # Update weights — move opposite to gradient
    W -= lr * dL_dW
    b -= lr * dL_db

    if step % 2000 == 0:
        print(f"  {step:>6}   {loss:>10.6f}   {pred.flatten().round(2)}")


# ── STEP 5: RESULTS ───────────────────────────────────────────────
print()
print("=" * 55)
print("  FINAL PREDICTIONS")
print("=" * 55)
names = ['Student A', 'Student B', 'Student C', 'Student D']
final = sigmoid(X @ W + b)
for name, p, real in zip(names, final.flatten(), y.flatten()):
    bar     = "█" * int(p*20) + "░" * (20 - int(p*20))
    verdict = "CORRECT" if round(p) == real else "WRONG"
    label   = "PASS" if round(p) == 1 else "FAIL"
    print(f"  {name}: [{bar}] {p:.2f} → {label} ({verdict})")


# ── STEP 6: PREDICT NEW STUDENT ──────────────────────────────────
print()
print("=" * 55)
print("  NEW STUDENT (never seen before)")
print("=" * 55)
new = np.array([[3, 6, 1]])
result = sigmoid(new @ W + b)[0][0]
print(f"  Studied: 3hrs  |  Slept: 6hrs  |  Ate: yes")
print(f"  Prediction: {result:.4f} → {'PASS' if result >= 0.5 else 'FAIL'} ({result*100:.1f}% confident)")
