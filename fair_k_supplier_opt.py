import numpy as np
import itertools

from helper_routines import compute_k_supplier_cost, check_valid_solution
from generate_instance import get_fair_k_supplier_instance

def fair_k_supplier_opt(I):
    n_c      = I["n_c"]
    k        = I["k"]
    C_idx    = I["C_idx"]
    F_idx    = I["F_idx"]
    GG_idx   = I["GG_idx"]
    rvec     = I["rvec"]
    t        = I["t"]  # number of groups
    U        = I["U"] # data matrix
    d        = U.shape[1]  # dimension of clients and facilities
    C        = U[C_idx,:]

    combinations_list = []
    for i in range(t):
        G_idx_i = GG_idx[i]
        r_i     = rvec[i]
        G_idx_i_combinations = list(itertools.combinations(G_idx_i, r_i))
        combinations_list.append(G_idx_i_combinations)
    # end for

    # print("combinations_list = ", combinations_list)

    cost_opt  = np.inf
    S_idx_opt = None
    S_opt     = None
    for S_idx in itertools.product(*combinations_list):
        S_idx_flat = np.array(list(itertools.chain(*S_idx)))
        S = U[np.array(S_idx_flat),:]
        cost = compute_k_supplier_cost(C, S)
        if cost < cost_opt:
            cost_opt = cost
            S_idx_opt    = S_idx_flat
            S_opt        = S
        # end if
    # end for
    return S, S_idx_opt, cost_opt
# end of fair_k_supplier_opt()
