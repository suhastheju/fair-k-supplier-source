import sys
import numpy as np
from pprint import pprint
from collections import defaultdict
from scipy.special import comb

from fair_k_supplier_3apx import fair_k_supplier_3apx
from fair_k_supplier_5apx import fair_k_supplier_5apx
from fair_k_supplier_opt import fair_k_supplier_opt
from generate_instance import get_div_k_supplier_instance
from helper_routines import compute_k_supplier_cost, check_valid_solution

##########################################################################################
# helper routines
def get_bit_vector_map(F_idx, GG_idx, t):
    bit_vectors = np.zeros((len(F_idx), t), dtype=np.int8)

    for group_idx, group in enumerate(GG_idx):
        mask = np.isin(F_idx, group).astype(np.int8)  # Check if facilities belong to a group
        bit_vectors[:, group_idx] = mask  # Update the bit vector for each facility

    # Convert the bit vectors to tuples for grouping facilities with the same bit vector
    bit_vector_tuples = [tuple(row) for row in bit_vectors]

    # Group facilities by their bit vectors
    bit_vector_map = defaultdict(list)
    for facility, bit_vector in zip(F_idx, bit_vector_tuples):
        bit_vector_map[bit_vector].append(facility)
    # end for
    
    return bit_vector_map
# end get_bit_vector_map()

def get_all_assignments(l, k):
    result = []
    stack = [([], 0)]  # Each element is a tuple (current vector, current sum)
    while stack:
        current_vector, current_sum = stack.pop()

        if len(current_vector) == l:
            if current_sum == k:
                result.append(current_vector)
            # end if
            continue
        # end if
        
        for i in range(0, k + 1):
            if current_sum + i <= k:
                stack.append((current_vector + [i], current_sum + i))
            # end if
        # end for
    # end while

    # Convert the result list into a numpy array with dtype np.int8
    result_array = np.array(result, dtype=np.int8)

    return result_array
# end get_all_assignments()

##########################################################################################
def div_k_supplier_3apx(J, seed):
    prng = np.random.default_rng(seed)

    J_n = J["n"]
    J_d = J["d"]
    J_t = J["t"]    
    J_k = J["k"]
    J_rvec   = J["rvec"]
    J_C_idx  = J["C_idx"]
    J_F_idx  = J["F_idx"]
    J_n_c    = J["n_c"]
    J_n_f    = J["n_f"]
    J_GG_idx = J["GG_idx"]
    J_seed   = J["seed"]
    J_U      = J["U"]
  
    # Get the bit vector map
    bit_vector_map = get_bit_vector_map(J_F_idx, J_GG_idx, J_t)
    bit_vector_matrix = np.array(list(bit_vector_map.keys()), dtype=np.int8).transpose()
    h, w = bit_vector_matrix.shape
    EE_idx = [np.array(val) for val in bit_vector_map.values()]
    fvec     = np.array([len(val) for val in EE_idx], dtype=np.int32)

    all_assignments = get_all_assignments(w, J_k)
    feasibility_matrix = bit_vector_matrix.dot(all_assignments.transpose())
    feasible_row_idx = np.where((feasibility_matrix.transpose() >= J_rvec).all(axis=1))[0]
    feasible_assign_idx = np.where((all_assignments <= fvec).all(axis=1))[0]
    feasible_idx = np.intersect1d(feasible_row_idx, feasible_assign_idx)

    I_d        = J_d
    I_k        = J_k
    I_C_idx    = J_C_idx
    I_n_c      = J_n_c
    I_seed     = J_seed
    I_U        = J_U

    cost_opt = np.inf
    S_opt = None
    S_idx_opt = None
    for idx in feasible_idx:
        assignment = all_assignments[idx]
        I_GG_idx   = [EE_idx[i] for i in np.arange(w) if assignment[i] > 0]
        I_F_idx    = np.concatenate(I_GG_idx)

        I_n        = len(I_C_idx) + len(I_F_idx)
        I_t        = len(I_GG_idx)
        I_n_f      = len(I_F_idx)
        I_GG_sizes = [len(group) for group in I_GG_idx]
        I_rvec     = np.array([i for i in range(w) if assignment[i] > 0], dtype=np.int32)

        I = {"n"       : I_n, 
             "d"       : I_d, 
             "t"       : I_t, 
             "k"       : I_k,
             "C_idx"   : I_C_idx,
             "F_idx"   : I_F_idx,
             "n_c"     : I_n_c,
             "n_f"     : I_n_f,
             "GG_idx"  : I_GG_idx,
             "GG_sizes": I_GG_sizes,
             "rvec"    : I_rvec,
             "seed"    : I_seed,
             "U"       : I_U
            }

        S, S_idx, cost = fair_k_supplier_3apx(I, seed)

        if cost_opt > cost:
            cost_opt = cost
            S_opt = S
            S_idx_opt = S_idx
        # end if
    # end for
    return S_opt, S_idx_opt, cost_opt
# end of div_k_supplier_3apx()
