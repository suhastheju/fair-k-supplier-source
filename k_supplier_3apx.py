import numpy as np
import random

from fair_k_supplier_3apx import choose_farthest_client
from fair_k_supplier_3apx import find_closest_facility
from fair_k_supplier_3apx import compute_k_supplier_cost

########################################################################################
def k_supplier_3apx(I, seed):
    prng = np.random.default_rng(seed)

    n_c    = I["n_c"]
    k      = I["k"]
    C_idx  = I["C_idx"]
    F_idx  = I["F_idx"]
    U      = I["U"]
    d      = U.shape[1]
    C      = U[C_idx,:]
    F      = U[F_idx,:]

    # Initialize and choose k farthest clients
    Cp = np.zeros((k, d))
    c_1 = C[prng.integers(0, n_c, size=1)[0]]
    Cp[0] = c_1
    # Choose farthest client for 'k-1' iterations
    for i in range(1, k):
        c_i = choose_farthest_client(C, Cp)
        Cp[i] = c_i
    # end for

    S = []
    S_idx = []
    for i in range(k):
        c_i = Cp[i]
        s, s_idx = find_closest_facility(c_i, F)
        S.append(s)
        S_idx.append(s_idx)
    # end for
    cost = compute_k_supplier_cost(C, np.vstack(S))

    return S, S_idx, cost
# end of k_supplier_three_apx()
