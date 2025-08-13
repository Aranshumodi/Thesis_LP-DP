import numpy as np
from math import cos, radians, degrees, acos
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools

def Vc_invariants(C_Elas_Prop, G_Elas_prop):
    E1c = C_Elas_Prop[0]
    E1g = G_Elas_prop[0]
    M = E1c-E1g
    Ug1= G_Elas_prop[2]
    

    E2= C_Elas_Prop[1]

    Gc = C_Elas_Prop[2]
    Ug2 = (Gc-Ug1)

    v12 = C_Elas_Prop[3]
    v21c = v12*(E2/E1c)
    v21g = v12*(E2/E1g)
    v21avg = (v21c + v21g) / 2

    Q = 1 - (v12*v21c)

    F1 = (((3*M/Q) + (4*Ug2))/8)
    F2 = ((M/Q)/2)
    F3 = (((M/Q) - (4*Ug2))/8)
    F4 = (((M/Q) + (4*Ug2))/8)

    U1l = (((3*E1g/Q) + (3*E2/Q) + (2*v12*E2/Q) + (4*Ug1))/8)
    U2l = (((E1g/Q)-(E2/Q))/2)
    U3l = (((E1g/Q) + (E2/Q) - (2*v12*E2/Q) - (4*Ug1))/8)
    U4l = (((E1g/Q) + (E2/Q) + (6*v12*E2/Q) - (4*Ug1))/8)
    U5l = (((E1g/Q) + (E2/Q) - (2*v12*E2/Q) + (4*Ug1))/8)

    F = [F1,F2,F3,F4]
    Ul= [U1l,U2l,U3l,U4l,U5l]
    return Ul,F

def invariants(Elas_prop):
    E1  = Elas_prop[0]
    E2  = Elas_prop[1]
    G12 = Elas_prop[2]
    v12 = Elas_prop[3]

    v21=v12*(E2/E1)
    Q=1-(v12*v21)
    Q11=E1*(Q**(-1))
    Q22=E2*(Q**(-1))
    Q12=v12*E2*(Q**(-1))
    Q66=G12
    
    U1=(1/8)*((3*Q11) + (3*Q22) + (2*Q12) +(4*Q66))
    U2=(Q11-Q22)/2
    U3=(1/8)*((Q11)+(Q22)-(2*Q12)-(4*Q66))
    U4=(1/8)*((Q11)+(Q22)+(6*Q12)-(4*Q66))
    U5=(1/8)*((Q11)+(Q22)-(2*Q12)+(4*Q66))

    U=[0,U1,U2,U3,U4,U5]

    return U


def possible_SS(L):
    # Starting list of 8 zeros
    SS = np.zeros(len(L))
    all_possible_arrays = []
    volume_fractions = []

    # For each possible number of 1s (from 0 to 8)
    for num_ones in range(len(SS) + 1):
        # Find all combinations of positions where 1s could go
        for indices in itertools.combinations(range(len(SS)), num_ones):
            arr = [0] * len(SS)
            # Set the chosen positions to 1
            for i in indices:
                arr[i] = 1
            vf = sum(arr) / len(arr)
            all_possible_arrays.append(arr)
            volume_fractions.append(vf)
    
    return all_possible_arrays, volume_fractions

def Wabs_norm(L,Z):
    W0 = 0
    W1 = 0
    W3 = 0
    for k in range(len(L)):
        delta = ((Z[k]/Z[0])**3) - ((Z[k+1]/Z[0])**3) 
        W0 = W0 + delta
        W1 = W1 + delta*cos(radians(2*L[k]))
        W3 = W3 + delta*cos(radians(4*L[k]))
    return W0, W1, W3

def Wf_norm(L,Z,SS):
    Wf0 = 0
    Wf1 = 0
    Wf3 = 0
    for k in range(len(L)):
        if SS[k] == 1:
            delta = (((Z[k]/Z[0])**3) - ((Z[k+1]/Z[0])**3))*(1/len(L))
        elif SS[k] == 0:
            delta = 0
        Wf0 = Wf0 + delta
        Wf1 = Wf1 + delta*cos(radians(2*L[k]))
        Wf3 = Wf3 + delta*cos(radians(4*L[k]))
    return Wf0, Wf1, Wf3

def Vabs_norm(L,Z):
    W0 = 0
    W1 = 0
    W3 = 0
    for k in range(len(L)):
        delta = ((Z[k]/Z[0])) - ((Z[k+1]/Z[0])) 
        W0 = W0 + delta
        W1 = W1 + delta*cos(radians(2*L[k]))
        W3 = W3 + delta*cos(radians(4*L[k]))
    return W0, W1, W3

def Vf_norm(L,Z,SS):
    Wf0 = 0
    Wf1 = 0
    Wf3 = 0
    for k in range(len(L)):
        if SS[k] == 1:
            delta = (((Z[k]/Z[0])) - ((Z[k+1]/Z[0])))*(1/len(L))
        elif SS[k] == 0:
            delta = 0
        Wf0 = Wf0 + delta
        Wf1 = Wf1 + delta*cos(radians(2*L[k]))
        Wf3 = Wf3 + delta*cos(radians(4*L[k]))
    return Wf0, Wf1, Wf3

def Zmat(L):
    T = np.full(len(L),0.125)
    Z=[0]
    for i in range(1,len(T)+1):
        zi = (T[0] + Z[i-1])
        Z.append(zi)

    Z = Z[::-1]
    return Z

def Nx_star(U1p, U2p, U3p, U4p, U5p,
            F1, F2, F3, F4,
            W1, W3, Wf0, Wf1, Wf3,
            R):
    """
    Compute normalized buckling load (N_x)* given lamination parameters
    and aspect-ratio factor R = b/a.
    """
    R2 = R**2
    R4 = R**4

    # Terms for D11* and D22* contributions
    term_U1 = U1p * (1 + R4)
    term_U2 = U2p * W1 * (1 - R4)
    term_U3 = U3p * W3 * (1 + R4 - 6 * R2)
    term_F1 = F1 * Wf0 * (1 + R4)
    term_F2 = F2 * Wf1 * (1 - R4)

    # Combined R^2 terms: U4, U5, F3/F4 contributions
    term_U45 = (2 * U4p + 4 * U5p) * R2
    term_F34 = (2 * F3 + 4 * F4) * Wf0 * R2
    term_F3w3 = (-6 * R2 + 1 + R4) * F3 * Wf3 

    N_star = (1/R2)*(term_U1 + term_U2 + term_U3 + term_U45 + term_F1 + term_F2 + term_F34 + term_F3w3)
    
    return N_star


# material invariants
C_Elas_prop=[142e3,11.2e3,5e3,0.3] #E1, E2, G12, \nu12
C_plyt=0.125 *200/150

G_Elas_prop=[44494,11.2e3,3894,.3]
G_plyt=0.125 *200/150

Vc = np.arange(0,1.125,.125)

Ul, F = Vc_invariants(C_Elas_prop,G_Elas_prop)
U1l = Ul[0]
U2l = Ul[1]
U3l = Ul[2]
U4l = Ul[3]
U5l = Ul[4]

F1 = F[0]
F2 = F[1]
F3 = F[2]
F4 = F[3]

#-----------------------------------------------------------------------------------------------------------------------------------
R =1

C_Elas_prop=[142e3,11.2e3,5e3,0.3]
C_plyt=0.125 *200/150
U_c=invariants(C_Elas_prop)

G_Elas_prop=[44494,11.2e3,3894,.3]
G_plyt=0.125 *200/150
U_g=invariants(G_Elas_prop)

def Nxcr(U, R, W1, W3):
    R2, R4 = R**2, R**4
    return (U[1]*(1+R4) +
            U[2]*W1*(1-R4) +
            U[3]*W3*(1+R4-6*R2) +
            (2*U[4] + 4*U[5])*R2) / R2

# Parameter sweep
W1_vals = np.linspace(-1, 1, 200)
W3_vals = 2 * W1_vals**2 - 1

Nx_vals=np.array([Nx_star(U1l,U2l,U3l,U4l,U5l,F1,F2,F3,F4,W1,W3,1,W1,W3,R)
                      for W1, W3 in zip(W1_vals, W3_vals)]) 
idx_max = np.argmax(Nx_vals)
w1_max, w3_max, Nx_max = W1_vals[idx_max], W3_vals[idx_max], Nx_vals[idx_max]