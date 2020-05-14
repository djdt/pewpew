#cython: language_level=3

import numpy as np

from typing import List

def multiotsu(x: np.ndarray, levels: int, nbins: int = 256) -> List[float]:
    p, bin_edges = np.histogram(x.ravel(), bins=nbins, density=True)

    cdef float [:, :] H = np.empty((nbins, nbins), dtype=np.float32)
    multiotsu_build_H(p, H, nbins)

    t = np.zeros(levels, dtype=np.int32)
    if levels == 2:
        multiotsu_two_level(H, nbins, t)
    elif levels == 3:
        multiotsu_three_level(H, nbins, t)
    else:
        raise ValueError("Levels must be 2 or 3.")

    return (bin_edges[t] + bin_edges[t + 1]) / 2.0


cdef multiotsu_build_H(double[:] p, float[:, :] H, int nbins):
    cdef Py_ssize_t i, j

    cdef float [:] P = np.empty(nbins, dtype=np.float32)
    cdef float [:] S = np.empty(nbins, dtype=np.float32)

    P[0] = S[0] = p[0]

    for i in range(1, nbins):
        P[i] = p[i] + P[i - 1]
        S[i] = p[i] * i + S[i - 1]

        H[0, i] = S[i] * S[i]
        if P[i] != 0.0:
            H[0, i] /= P[i]

    for i in range(1, nbins):
        for j in range(i, nbins):
            Pij = P[j] - P[i - 1]
            Sij = S[j] - S[i - 1]
            H[i, j] = Sij * Sij
            if Pij != 0.0:
                H[i, j] /= Pij


cdef multiotsu_two_level(float[:, :] H, int nbins, int[:] t):
    cdef float smax = 0.0
    cdef float s

    cdef Py_ssize_t i, j

    for i in range(1, nbins - 2):
        for j in range(i + 1, nbins - 1):
            s = H[1, i] + H[i + 1, j] + H[j + 1, nbins - 1]
            if s > smax:
                smax = s
                t[0], t[1] = i, j


cdef multiotsu_three_level(float[:, :] H, int nbins, int[:] t):
    cdef float smax = 0.0
    cdef float s

    cdef Py_ssize_t i, j, k

    for i in range(1, nbins - 3):
        for j in range(i + 1, nbins - 2):
            for k in range(j + 1, nbins - 1):
                s = H[1, i] + H[i + 1, j] + H[j + 1, k] + H[k + 1, nbins - 1]
                if s > smax:
                    smax = s
                    t[0], t[1], t[2] = i, j, k
