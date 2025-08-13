#!/usr/bin/env python
"""
Minimize Vc subject to a lower bound on normalized buckling load Nx*,
with all physical bounds on lamination parameters enforced.
"""
import numpy as np
from scipy.optimize import minimize
from Defs import (
    U1l, U2l, U3l, U4l, U5l,
    F1, F2, F3, F4,
    Nx_star
)

# Aspect ratio
R = 1

# Compute reference max buckling at known optimum
from test2 import Nx_max
# Required minimum buckling
Nx_min = 0.9 * Nx_max

# Helper: cap function a(Vc)
def a_of(Vc):
    return 1.0 - (1.0 - Vc) ** 3

# Objective: minimize Vc
def objective(x):
    return x[5]

# Constraint: Nx* >= Nx_min
def constraint_Nxmin(x):
    W1, W3, Wf0, Wf1, Wf3, Vc = x
    return Nx_star(
        U1l, U2l, U3l, U4l, U5l,
        F1, F2, F3, F4,
        W1, W3, Wf0, Wf1, Wf3,
        R
    ) - Nx_min

# Physical bounds constraints (g(x) >= 0)
def c_W3_low(x):   return x[1] - (2.0 * x[0]**2 - 1.0)
def c_W3_high(x):  return 1.0 - x[1]
def c_Wf0_low(x):  return x[2] - x[5]**3
def c_Wf0_high(x): return a_of(x[5]) - x[2]
def c_Wf1_low(x):  return x[3] + a_of(x[5])
def c_Wf1_high(x): return a_of(x[5]) - x[3]
def c_Wf3_roof(x): return a_of(x[5]) - x[4]
def c_Wf3_floor(x):
    a = a_of(x[5])
    return x[4] - (2.0/a) * x[3]**2 + a if a > 1e-12 else 1e-12 - abs(x[4])

# Matching constraints: |W_i - W_fi| <= 1 - Vc^3
def c_W1_diff_pos(x): return (1.0 - x[5]) - (x[0] - x[3])
def c_W1_diff_neg(x): return (1.0 - x[5]) - (x[3] - x[0])
def c_W3_diff_pos(x): return (1.0 - x[5]) - (x[1] - x[4])
def c_W3_diff_neg(x): return (1.0 - x[5]) - (x[4] - x[1])

# Collect all constraints
constraints_list = [
    {'type': 'ineq', 'fun': constraint_Nxmin},
    {'type': 'ineq', 'fun': c_W3_low},
    {'type': 'ineq', 'fun': c_W3_high},
    {'type': 'ineq', 'fun': c_Wf0_low},
    {'type': 'ineq', 'fun': c_Wf0_high},
    {'type': 'ineq', 'fun': c_Wf1_low},
    {'type': 'ineq', 'fun': c_Wf1_high},
    {'type': 'ineq', 'fun': c_Wf3_roof},
    {'type': 'ineq', 'fun': c_Wf3_floor},
    {'type': 'ineq', 'fun': c_W1_diff_pos},
    {'type': 'ineq', 'fun': c_W1_diff_neg},
    {'type': 'ineq', 'fun': c_W3_diff_pos},
    {'type': 'ineq', 'fun': c_W3_diff_neg},
]

# Bounds: W1,W3,Wf0,Wf1,Wf3 in [-1,1]; Vc in [0,1]
bounds = [(-1.0, 1.0)] * 5 + [(0.0, 1.0)]

# Initial guess
x0 = np.array([
    0.0,      # W1
    0.0,      # W3
    0.5**3,   # Wf0
    0.0,      # Wf1
    0.0,      # Wf3
    0.1       # Vc
])

# Run SLSQP
res = minimize(
    objective,
    x0,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints_list,
    options={'disp': True}
)

# Fallback to COBYLA
if not res.success:
    res = minimize(
        objective,
        x0,
        method='COBYLA',
        bounds=bounds,
        constraints=constraints_list,
        options={'disp': True, 'maxiter': 10000}
    )

# Report results
if res.success:
    W1_opt, W3_opt, Wf0_opt, Wf1_opt, Wf3_opt, Vc_opt = res.x
    Nx_opt = Nx_star(
        U1l, U2l, U3l, U4l, U5l,
        F1, F2, F3, F4,
        W1_opt, W3_opt, Wf0_opt, Wf1_opt, Wf3_opt,
        R
    )
    print("Minimal-Vc design:")
    print(f"  Vc   = {Vc_opt:.4f}")
    print(f"  W1,W3= {W1_opt:.4f}, {W3_opt:.4f}")
    print(f"  Wf0  = {Wf0_opt:.4f} (bounds: [{Vc_opt**3:.4f}, {1-(1-Vc_opt)**3:.4f}])")
    print(f"  Wf1  = {Wf1_opt:.4f} (|≤ {1-(1-Vc_opt)**3:.4f}])")
    print(f"  Wf3  = {Wf3_opt:.4f} (|≤ {1-(1-Vc_opt)**3:.4f}])")
    print(f"  Nx*  = {Nx_opt:.4f}")
else:
    print("Optimization failed:", res.message)

