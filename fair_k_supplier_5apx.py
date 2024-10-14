import numpy as np
import random
import math
import sys

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from generate_instance import get_fair_k_supplier_instance
from fair_k_supplier_opt import fair_k_supplier_opt
from fair_k_supplier_3apx import fair_k_supplier_3apx
from helper_routines import find_closest_facility, check_valid_solution, debug_stats
from helper_routines import choose_farthest_client, get_sorted_radii, is_dist_ci_and_Gj_lmbda
from helper_routines import extend_graph_with_capacities, extend_graph_with_capacities_mk1
from helper_routines import compute_k_supplier_cost
from helper_routines import get_facility_type, min_distance, get_facility_type_with_S
from helper_routines import get_rvec_from_S_idx

from config import LOG_LEVEL

###############################################################################
def fair_k_supplier_5apx(I, seed):
    random.seed(seed)
    np.random.seed(seed)

    n_c      = I["n_c"]
    k        = I["k"]
    C_idx    = I["C_idx"]
    F_idx    = I["F_idx"]
    GG_idx   = I["GG_idx"]
    GG_sizes = I["GG_sizes"]
    rvec     = I["rvec"]
    t        = I["t"]  # number of groups
    U        = I["U"]
    d        = U.shape[1]  # dimension of clients and facilities

    # Extract clients, facilities and groups
    C        = U[C_idx,:]
    F        = U[F_idx,:]
    GG       = [U[GG_idx[i],:] for i in range(t)]

    # Initialize and choose k farthest clients
    Cp = np.zeros((k, d))
    Sp = np.zeros((k, d))
    Sp_idx = np.zeros(k, dtype=int)

    c_1        = C[random.randint(0, n_c - 1)]
    Cp[0]      = c_1
    s1, s1_idx = find_closest_facility(c_1, F)
    Sp[0]      = s1
    Sp_idx[0]  = F_idx[s1_idx]

    # Choose farthest client for 'k' iterations
    for i in range(1, k):
        c_i = choose_farthest_client(C, Cp)
        Cp[i] = c_i
        s_i, s_i_idx = find_closest_facility(c_i, F)
        Sp[i] = s_i
        Sp_idx[i] = F_idx[s_i_idx]
    # end for

    facility_types = get_facility_type(Sp_idx, GG_idx)
    rvec_Sp = np.zeros(t, dtype=int)
    for i in range(k):
        j = facility_types[i]
        rvec_Sp[j] += 1
    # end for

    # Check if requirements are met
    if np.all(rvec_Sp >= rvec):
        cost = compute_k_supplier_cost(C, Sp)
        return Sp, Sp_idx, cost
    # end if

    S = []
    S_idx = []
    radius = math.inf
    cost = math.inf
    # Iterate over l = 1 to k for each group of clients
    left, right = 1, k+1
    while left <= right:
        l  = (left + right) // 2
        Sl = []
        Sl_idx  = []
        Gammal  = get_sorted_radii(F, Cp[:l])
        lmbdal  = np.inf
        cost_Sl = np.inf

        rvec_Sl_lambda = rvec_Sp.copy()
        for i in range(l):
            j = facility_types[i]
            rvec_Sl_lambda[j] -= 1
        # end for
        rvec_diff = np.maximum( 0, np.subtract(rvec, rvec_Sl_lambda))

        valid_lambda = False
        # Process each λ in sorted radii
        for lmbda in Gammal:
            Sl_lmbda     = []
            Sl_lmbda_idx = []

            # Construct H_k_λ graph using broadcasting instead of loops
            Hl_lmbda = np.zeros((l, t), dtype=int)
            for i in range(l):
                c_i = Cp[i]
                for j in range(t):
                    G_j = GG[j]
                    if min_distance(c_i, G_j) <= lmbda:
                        Hl_lmbda[i, j] = 1
                    # end if
                # end for
            # end for

            # Extend the graph based on number of requirements
            Hl_lmbda_ext, ext_map = extend_graph_with_capacities_mk1(Hl_lmbda, rvec_diff)
            sparse_Hl_lmbda = csr_matrix(Hl_lmbda_ext)

            # Find maximum matching
            Ml_lmbda_ext = maximum_bipartite_matching(sparse_Hl_lmbda, perm_type='column')
            Ml_lmbda = [ext_map[col] if col != -1 else -1 for col in Ml_lmbda_ext]

            if -1 in Ml_lmbda:
                continue  # No valid matching, skip this λ
            else: 
                # Build solution S^l_λ
                rvec_Ml_lmbda = np.zeros(t, dtype=int)
                for i in range(l):
                    j = Ml_lmbda[i]
                    c_i = Cp[i]
                    G_j = GG[j]
                    s, s_idx = find_closest_facility(c_i, G_j)
                    Sl_lmbda.append(s)
                    Sl_lmbda_idx.append(GG_idx[j][s_idx])
                    rvec_Ml_lmbda[j] += 1  # reduce remaining requirements
                # end for

                if np.any(rvec_Ml_lmbda < rvec_diff):
                    continue  # not a fair-swap, skip this λ
                # end if
                valid_lambda = True # Found a valid matching

                # Add remaining facilities to meet requirements
                for j in range(l, k):
                    Sl_lmbda.append(Sp[j])  # Directly extend S^l_λ
                    Sl_lmbda_idx.append(Sp_idx[j])
                # end for

                # Compute cost and update if improved
                Sl_lmbda_vstack = np.vstack(Sl_lmbda)
                cost_Sl_lmbda = compute_k_supplier_cost(C, Sl_lmbda_vstack)
                if cost_Sl_lmbda < cost_Sl:
                    lmbdal = lmbda
                    cost_Sl = cost_Sl_lmbda
                    Sl = Sl_lmbda
                    Sl_idx = Sl_lmbda_idx
                # end if
            # end if
            break # Exit after finding the smallest λ
        # end while

        # Update the global solution if this facility subset is better
        if not valid_lambda:
            left = l + 1
        else:
            if cost_Sl < cost:
                radius = lmbdal
                cost = cost_Sl
                S = Sl
                S_idx = Sl_idx
            # end if
            right = l - 1
        # end if
        if left >= right:
            break
        # end if 
    # end for
    return S, S_idx, cost
#end fair_k_supplier_5apx()
