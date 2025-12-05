import numpy as np
from math import sin, cos, sqrt, pi
def truss(A):
    """Computes mass and stress for the 10-bar truss problem (complex-compatible)."""

    if isinstance(A[0], complex):
        complexFlag = True
    else:
        complexFlag = False
        
    # ensure complex dtype if user passes floats
    A = np.asarray(A, dtype=complex)

    # --- truss setup -----
    P = 1e5
    Ls = 360.0
    Ld = np.sqrt(360**2 * 2)  

    start = [5, 3, 6, 4, 4, 2, 5, 6, 3, 4]
    finish = [3, 1, 4, 2, 3, 1, 4, 3, 2, 1]
    phi = np.array([0, 0, 0, 0, 90, 90, -45, 45, -45, 45]) * np.pi/180
    L = np.array([Ls, Ls, Ls, Ls, Ls, Ls, Ld, Ld, Ld, Ld])

    nbar = len(A)
    E = 1e7 * np.ones(nbar, dtype=complex)
    rho = 0.1 * np.ones(nbar, dtype=complex)

    Fx = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=complex)
    Fy = np.array([0.0, -P, 0.0, -P, 0.0, 0.0], dtype=complex)
    rigid = [False, False, False, False, True, True]

    n = len(Fx)
    DOF = 2

    # mass
    mass = np.sum(rho * A * L)

    # stiffness and stress matrices (complex)
    K = np.zeros((DOF*n, DOF*n), dtype=complex)
    S = np.zeros((nbar, DOF*n), dtype=complex)

    # assemble global stiffness
    for i in range(nbar):
        Ksub, Ssub = bar(E[i], A[i], L[i], phi[i])
        idx = node2idx([start[i], finish[i]], DOF)

        K[np.ix_(idx, idx)] += Ksub
        S[i, idx] = Ssub

    # load vector
    F = np.zeros((n*DOF, 1), dtype=complex)
    for i in range(n):
        idx = node2idx([i+1], DOF)
        F[idx[0]] = Fx[i]
        F[idx[1]] = Fy[i]

    # boundary conditions
    idx = [i+1 for i, val in enumerate(rigid) if val]
    remove = node2idx(idx, DOF)

    K = np.delete(K, remove, axis=0)
    K = np.delete(K, remove, axis=1)
    F = np.delete(F, remove, axis=0)
    S = np.delete(S, remove, axis=1)

    # solve
    d = np.linalg.solve(K, F)

    # stresses (each is possibly complex)
    stress = (S @ d).reshape(nbar)

    if not complexFlag:
        mass = np.real(mass)
        stress = np.real(stress)
    return mass, stress


def bar(E, A, L, phi):
    """Complex-compatible element stiffness and stress matrices."""

    # use complex-safe versions
    c = np.cos(phi)
    s = np.sin(phi)

    # stiffness
    k0 = np.array([[c**2, c*s], [c*s, s**2]], dtype=complex)
    k1 = np.hstack([k0, -k0])
    K = E*A/L * np.vstack([k1, -k1])

    # stress operator
    S = E/L * np.array([-c, -s, c, s], dtype=complex)

    return K, S


def node2idx(node, DOF):
    idx = np.array([], dtype=int)
    for n in node:
        start = DOF*(n-1)
        finish = DOF*n
        idx = np.concatenate((idx, np.arange(start, finish, dtype=int)))
    return idx
