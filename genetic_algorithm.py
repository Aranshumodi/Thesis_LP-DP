import random
import numpy as np
from Defs import Wabs_norm, Wf_norm, Zmat
import argparse

# GA for laminate stacking, with exact Vc via repair
N_HALF      = 8
POP_SIZE    = 250
GENERATIONS = 2000
CXPB, MUTPB = 0.8, 0.2
ANGLES      = [0, 45, -45, 90]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--W1',  type=float, required=True)
    p.add_argument('--W3',  type=float, required=True)
    p.add_argument('--Wf0', type=float, required=True)
    p.add_argument('--Wf1', type=float, required=True)
    p.add_argument('--Wf3', type=float, required=True)
    p.add_argument('--Vc',  type=float, required=True)
    return vars(p.parse_args())

target = parse_args()
TARGET_VC  = target['Vc']
TARGET_CFRP_COUNT = int(round(TARGET_VC * N_HALF))

def create_individual():
    """Random atomic genes; Vc enforced by repair later."""
    return [(random.choice([0,1]), random.choice(ANGLES))
            for _ in range(N_HALF)]

def mirror_seq(half):
    return half + half[::-1]

# at top of file, tune these weights:
W1_WT, W3_WT    = 1.0, 1.0        # lower priority
Wf0_WT, Wf1_WT, Wf3_WT = 10.0, 1.0, 1.0  # higher priority

def fitness(genome):
    full = mirror_seq(genome)
    mats = np.array([m for m,a in full])
    angs = np.array([a for m,a in full])
    Z    = np.array(Zmat(angs))

    _, W1, W3    = Wabs_norm(angs, Z)
    Wf0, Wf1, Wf3= Wf_norm(angs, Z, mats)

    err_w1  = abs(W1  - target['W1'])**2
    err_w3  = abs(W3  - target['W3'])**2
    err_wf0 = abs(Wf0 - target['Wf0'])**2
    err_wf1 = abs(Wf1 - target['Wf1'])**2
    err_wf3 = abs(Wf3 - target['Wf3'])**2

    total_error = (W1_WT  * err_w1
                 + W3_WT  * err_w3
                 + Wf0_WT * err_wf0
                 + Wf1_WT * err_wf1
                 + Wf3_WT * err_wf3)

    return -total_error

def repair(genome):
    """Ensure exactly TARGET_CFRP_COUNT plies = CFRP (m=1)."""
    current = sum(m for m,a in genome)
    diff = current - TARGET_CFRP_COUNT
    if diff > 0:
        # too many CFRP: flip diff random CFRP→GFRP
        c_idx = [i for i,(m,a) in enumerate(genome) if m==1]
        for i in random.sample(c_idx, diff):
            genome[i] = (0, genome[i][1])
    elif diff < 0:
        # too few CFRP: flip -diff random GFRP→CFRP
        g_idx = [i for i,(m,a) in enumerate(genome) if m==0]
        for i in random.sample(g_idx, -diff):
            genome[i] = (1, genome[i][1])
    return genome

def repair_angles(genome):
    # count +45 vs -45 in the half‐genome
    plus = [i for i,(m,a) in enumerate(genome) if a == 45]
    minus= [i for i,(m,a) in enumerate(genome) if a == -45]
    diff = len(plus) - len(minus)
    if diff > 0:
        # too many +45: flip diff of them to -45
        for i in random.sample(plus, diff):
            genome[i] = (genome[i][0], -45)
    elif diff < 0:
        # too many -45: flip -diff of them to +45
        for i in random.sample(minus, -diff):
            genome[i] = (genome[i][0], 45)
    return genome


def tournament_selection(pop, k=3):
    return [max(random.sample(pop, k), key=lambda ind: ind['fitness'])
            for _ in pop]

def crossover(p1, p2):
    if random.random() > CXPB:
        return p1['genome'], p2['genome']
    pt = random.randint(1, N_HALF-1)
    return (p1['genome'][:pt] + p2['genome'][pt:],
            p2['genome'][:pt] + p1['genome'][pt:])

def mutate(genome):
    for i in range(N_HALF):
        if random.random() < MUTPB:
            # atomic resample of (material,angle)
            genome[i] = (random.choice([0,1]), random.choice(ANGLES))
    return genome

def main():
    desc = ", ".join(f"{k}={v}" for k,v in target.items())
    print(f"GA start (repair Vc={TARGET_VC}): {desc}")

    # --- init population with Vc- & angle-repair ---
    pop = []
    for _ in range(POP_SIZE):
        g = create_individual()
        g = repair(g)           # enforce exact Vc
        g = repair_angles(g)    # enforce ±45° balance
        pop.append({'genome': g})
    for ind in pop:
        ind['fitness'] = fitness(ind['genome'])

    # --- GA loop ---
    for gen in range(1, GENERATIONS+1):
        pool = tournament_selection(pop)
        offspring = []
        for i in range(0, POP_SIZE, 2):
            c1, c2 = crossover(pool[i], pool[i+1])

            # mutate + repair material & angles for child 1
            g1 = mutate(c1)
            g1 = repair(g1)
            g1 = repair_angles(g1)
            offspring.append({'genome': g1})

            # mutate + repair material & angles for child 2
            g2 = mutate(c2)
            g2 = repair(g2)
            g2 = repair_angles(g2)
            offspring.append({'genome': g2})

        # evaluate fitness
        for child in offspring:
            child['fitness'] = fitness(child['genome'])
        pop = offspring

        if gen % 10 == 0:
            best = max(pop, key=lambda x: x['fitness'])
            print(f"Gen {gen:3d}: Best fitness = {best['fitness']:.6f}")

    # --- final reporting ---
    best = max(pop, key=lambda x: x['fitness'])
    full = mirror_seq(best['genome'])
    mats = np.array([m for m,a in best['genome']])
    angs = np.array([a for m,a in best['genome']])
    Z    = np.array(Zmat(angs))
    _, W1, W3    = Wabs_norm(angs, Z)
    Wf0, Wf1, Wf3= Wf_norm(angs, Z, mats)
    actual_Vc = mats.mean()
    Wf0 = N_HALF*Wf0
    Wf1 = N_HALF*Wf1
    Wf3 = N_HALF*Wf3


    print("\nOptimal stacking sequence (full):")
    for i,(m,a) in enumerate(full,1):
        print(f"Ply {i:2d}: {'CFRP' if m else 'GFRP'}, {a:3d}°")

    print("\nResulting parameters:")
    print(f"  W1   = {W1:.4f}")
    print(f"  W3   = {W3:.4f}")
    print(f"  Wf0  = {Wf0:.4f}")
    print(f"  Wf1  = {Wf1:.4f}")
    print(f"  Wf3  = {Wf3:.4f}")
    print(f"  Vc   = {actual_Vc:.4f}")


if __name__ == '__main__':
    main()

#python genetic_algorithm.py --W1 1 --W3 1 --Wf0 1 --Wf1 1 --Wf3 1 --Vc 1
