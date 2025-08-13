# LP–DP Hybrid Laminate Optimisation
Python implementation of the master's thesis work carried out at the Faculty of Aerospace Engineering, TU Delft. The implementation serves as a proof of concept, supporting the parametrisation of a two-material inter-layer hybrid composite laminate.

## Master Thesis Details
Lamination Parameter–Dispersion Parameter (LP–DP) Formulation for Optimal Design of Hybrid Composite Laminates

## Contributors
Master's Thesis Student: Aranshu Modi  
Supervisors: Dr. Ir. D.M.J. Peeters and Dr. Ir. S.G.P. Castro

## Motivation
Conventional LP-based optimisation is efficient for single-material laminates but cannot represent material distribution in hybrid CFRP–GFRP laminates. This work augments LPs with Dispersion Parameters (DPs) to unify orientation and material distribution in a continuous, convex design space.  
The optimisation is split into two stages to keep the problem tractable and manufacturable:

1) Gradient-Based Optimisation (GBO): uses SLSQP to minimise carbon content \(V_c\) while meeting buckling and feasibility constraints in LP–DP space.  
2) Genetic Algorithm (GA): retrieves discrete, symmetric and balanced stacking sequences that reproduce the LP–DP targets, enforcing manufacturing rules (10% rule, UD run limits, material content).

## Repository
This repository consists of the following:
- `GBO_final.py`: SLSQP-based optimisation to find LP–DP targets by minimising \(V_c\) under buckling and feasibility constraints.
- `GA_final.py`: Genetic Algorithm for sequence retrieval from LP–DP targets with deterministic repairs.
- `Defs.py`: ABD, LP/FLP/DP evaluation, buckling load computation, utilities.
- `README.md`: this file
- `LICENSE`: CC BY-SA 4.0
