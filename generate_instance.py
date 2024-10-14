import sys
import random
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from pprint import pprint

from config import LOG_LEVEL
from config import UNDEFINED

###############################################################################
# generating a random instance
def get_fair_k_supplier_instance( gen_type,         # GEN_NP_RAND, GEN_SK_BLOBS
                                  cf_split_type,    # EQUAL_PARTITION, RANDOM_PARTITION, RAND_SAMPLE
                                  gg_split_type,    # EQUAL_PARTITION, RANDOM_PARTITION
                                  n,                # number of data points
                                  d,                # number of dimensions
                                  t,                # number of groups
                                  k,                # number of facilities
                                  n_c,              # number of clients
                                  n_f,              # number of facilities
                                  seed,             # random seed
                                  log               # log file
                                 ):
    # using a pseudo-random number generator to ensure reproducibility
    prng = np.random.default_rng(seed)

    # Generate n data points, each with d dimensions
    if gen_type == "GEN_NP_RAND":
        data_points = prng.random((n, d))
    elif gen_type == "GEN_SK_BLOBS":
        X, y = make_blobs(n_samples=n, 
                          centers=k, 
                          n_features=d, 
                          random_state=seed)
        scaler = MinMaxScaler()
        data_points = scaler.fit_transform(X)
    else:
        log.write("Error: Unknown generation version `%s`\n"%(gen_type))
        sys.exit(1)
    # end if
   
    # Separate data points into clients and facilities
    num_clients    = n_c
    num_facilities = n_f

    if "PARTITION" in cf_split_type:
        if num_clients + num_facilities != n:
            log.write("Error: Number of clients (%d) and facilities (%d) do not add up to n (%d)\n"%\
                       (num_clients, num_facilities, n))
            sys.exit(1)
        # end if
        idx = np.array(np.arange(n))
        prng.shuffle(idx)
        clients    = idx[:num_clients]
        facilities = idx[num_clients:]
    elif "SAMPLE" in cf_split_type:
        clients    = prng.choice(np.arange(n), num_clients, replace=False)
        facilities = prng.choice(np.arange(n), num_facilities, replace=False)
    else:
        log.write("Error: Unknown clients-facility type `%s`\n"%(cf_split_type))
        sys.exit(1)
    # end if
    
    # Randomly partition facilities into t groups (no intersections)
    if gg_split_type == "EQUAL_PARTITION":
        group_sizes = [num_facilities // t] * t
        for i in np.arange(num_facilities % t):
            group_sizes[i] += 1
        # end for 
    elif gg_split_type == "RANDOM_PARTITION":
        break_points = sorted(prng.choice(np.arange(1, num_facilities), t - 1, replace=False))
        group_sizes = [break_points[0]] + [break_points[i] - break_points[i - 1] \
                        for i in np.arange(1, t - 1)] + [num_facilities - break_points[-1]]
    else:
        log.write("Error: Unknown group split type `%s`\n"%(gg_split_type))
        sys.exit(1)
    # end if
    random.shuffle(facilities)
    groups = []
    start_idx = 0
    for size in group_sizes:
        groups.append(sorted(np.array(facilities[start_idx:start_idx + size])))
        start_idx += size
    # end for
     
    # Generate a random list of requirements
    while True:
        break_points = sorted(prng.choice(np.arange(1, k), t - 1))
        requirements = np.array([break_points[0]] + [break_points[i] - break_points[i - 1]
                        for i in np.arange(1, t - 1)] + [k - break_points[-1]])
        if ((np.all(np.array(requirements) <= np.array(group_sizes)) ==  True) \
                and (requirements > 0).all()):
            break;
        #end if
    #end while

    instance = { "n"       : n, 
                 "d"       : d, 
                 "t"       : t, 
                 "k"       : k,
                 "n_c"     : num_clients,
                 "n_f"     : num_facilities,
                 "C_idx"   : sorted(np.array(clients)),
                 "F_idx"   : sorted(np.array(facilities)),
                 "GG_idx"  : groups,
                 "GG_sizes": group_sizes,
                 "rvec"    : np.array(requirements),
                 "seed"    : seed,
                 "U"       : data_points
                }
    return instance
# end get_fair_k_supplier_instance()

###############################################################################
def get_div_k_supplier_instance( gen_type,         # GEN_NP_RAND, GEN_SK_BLOBS
                                 cf_split_type,    # RAND_PARTITION, RAND_SAMPLE
                                 gg_split_type,    # EQUAL_SIZE, RAND_SIZE
                                 rvec_type,        # EQUAL_SIZE, EQUAL_PARTITION
                                 n, 
                                 d, 
                                 t, 
                                 k, 
                                 n_c, 
                                 n_f, 
                                 seed, 
                                 log
                               ):
    # using a pseudo-random number generator to ensure reproducibility
    np.random.seed(seed)
    random.seed(seed)
    prng = np.random.default_rng(seed)

    # Generate n data points, each with d dimensions
    if gen_type == "GEN_NP_RAND":
        data_points = prng.random((n, d))
    elif gen_type == "GEN_SK_BLOBS":
        X, y = make_blobs(n_samples=n, 
                          centers=k, 
                          n_features=d, 
                          random_state=seed)
        scaler = MinMaxScaler()
        data_points = scaler.fit_transform(X)
    else:
        log.write("Error: Unknown generation version `%s`\n"%(gen_type))
        sys.exit(1)
    # end if
   
    num_clients    = n_c
    num_facilities = n_f

    # Separate data points into clients and facilities
    if cf_split_type == "RAND_PARTITION":
        if num_clients + num_facilities != n:
            log.write("Error: Number of clients (%d) and facilities (%d) do not add up to n (%d)\n"%\
                       (num_clients, num_facilities, n))
            sys.exit(1)
        # end if
        idx = np.array(np.arange(n))
        prng.shuffle(idx)
        clients    = idx[:num_clients]
        facilities = idx[num_clients:]
    elif cf_split_type == "RAND_SAMPLE" or cf_split_type == "EQUAL_SIZE":
        clients    = prng.choice(np.arange(n), num_clients, replace=False)
        facilities = prng.choice(np.arange(n), num_facilities, replace=False)
    else:
        log.write("Error: Unknown clients-facility type `%s`\n"%(cf_split_type))
        sys.exit(1)
    # end if
 
    groups = []
    random.shuffle(facilities)
    start_idx = 0
    init_size = num_facilities // t
    for i in np.arange(t):
        group_init = np.array(facilities[start_idx:start_idx + init_size])
        start_idx += init_size

        if gg_split_type == "EQUAL_SIZE":
            group_add = prng.choice(facilities, init_size, replace=False)
        elif gg_split_type == "RAND_SIZE":
            rand_size = prng.integers(low=1, high=num_facilities, size=1)[0]
            group_add = prng.choice(facilities, rand_size, replace=False)
        else:
            log.write("Error: Unknown group split type `%s`\n"%(gg_split_type))
            sys.exit(1)
        # end if
        groups.append(sorted(np.unique(np.concatenate((group_init, group_add)))))
    # end for
    group_sizes = [len(group) for group in groups]

    # Generate a random list of requirements
    if rvec_type == "EQUAL_PARTITION":
        while True:
            break_points = sorted(prng.choice(np.arange(1, k), t - 1))
            requirements = np.array([break_points[0]] + [break_points[i] - break_points[i - 1]
                            for i in np.arange(1, t - 1)] + [k - break_points[-1]])
            if ((np.all(np.array(requirements) <= np.array(group_sizes)) ==  True) \
                    and (requirements > 0).all()):
                break;
            #end if
        #end while
    elif rvec_type == "EQUAL_SIZE":
        requirements = np.array([k // t] * t)
        if k % t != 0:
            requirements += 1
    else:
        log.write("Error: Unknown requirement vector type `%s`\n"%(rvec_type))
        sys.exit(1)
    # end if

    instance = { "n"       : n, 
                 "d"       : d, 
                 "t"       : t, 
                 "k"       : k,
                 "n_c"     : num_clients,
                 "n_f"     : num_facilities,
                 "C_idx"   : sorted(np.array(clients)),
                 "F_idx"   : sorted(np.array(facilities)),
                 "GG_idx"  : groups,
                 "GG_sizes": group_sizes,
                 "rvec"    : np.array(requirements),
                 "seed"    : seed,
                 "U"       : data_points
                }
    return instance
# end get_div_k_supplier_instance()



###############################################################################
def test_get_fair_k_supplier_instance():
    print("Testing get_fair_k_supplier_instance()")

    n    = 10
    d    = 2
    t    = 3
    k    = 4
    seed = 12345
    log  = sys.stdout
    gg_split_type = "EQUAL_PARTITION"

    #################################
    gen_type = "GEN_NP_RAND"

    cf_split_type  = "RAND_PARTITION"
    n_c  = 6
    n_f  = 4
    I = get_fair_k_supplier_instance(gen_type, cf_split_type, gg_split_type, n, d, t, k, n_c, n_f, seed, log)
    np.set_printoptions(suppress=True)
    print("generator type: %s, clients-facilities: %s"%(gen_type, cf_split_type))
    print(I)

    cf_split_type  = "RAND_SAMPLE"
    n_c  = 8
    n_f  = 6
    I = get_fair_k_supplier_instance(gen_type, cf_split_type, gg_split_type, n, d, t, k, n_c, n_f, seed, log)
    np.set_printoptions(suppress=True)
    print("generator type: %s, clients-facilities: %s"%(gen_type, cf_split_type))
    print(I)

    ###############################
    gen_type = "GEN_SK_BLOBS"

    cf_split_type  = "RAND_PARTITION"
    n_c  = 6
    n_f  = 4
    I = get_fair_k_supplier_instance(gen_type, cf_split_type, gg_split_type, n, d, t, k, n_c, n_f, seed, log)
    print("generator type: %s, clients-facilities: %s"%(gen_type, cf_split_type))
    print(I)

    cf_split_type  = "RAND_SAMPLE"
    n_c  = 8
    n_f  = 6
    I = get_fair_k_supplier_instance(gen_type, cf_split_type, gg_split_type, n, d, t, k, n_c, n_f, seed, log)
    print("generator type: %s, clients-facilities: %s"%(gen_type, cf_split_type))
    print(I)
# end test_get_fair_k_supplier_instance()

###############################################################################
def test_get_div_k_supplier_instance():
    print("Testing get_div_k_supplier_instance()")

    n    = 10
    d    = 2
    t    = 3
    k    = 3
    seed = 12345
    log  = sys.stdout
    rvec_type = "EQUAL_SIZE"

    #################################
    gen_type = "GEN_NP_RAND"
    cf_split_type  = "RAND_PARTITION"
    gg_split_type  = "EQUAL_SIZE"
    n_c  = 6
    n_f  = 4
    I = get_div_k_supplier_instance(gen_type, cf_split_type, gg_split_type, rvec_type, n, d, t, k, n_c, n_f, seed, log)
    np.set_printoptions(suppress=True)
    print("generator type: %s, clients-facilities: %s"%(gen_type, cf_split_type))
    pprint(I)

    cf_split_type  = "RAND_SAMPLE"
    gg_split_type  = "RAND_SIZE"
    n_c  = 8
    n_f  = 6
    I = get_div_k_supplier_instance(gen_type, cf_split_type, gg_split_type, rvec_type, n, d, t, k, n_c, n_f, seed, log)
    np.set_printoptions(suppress=True)
    print("generator type: %s, clients-facilities: %s"%(gen_type, cf_split_type))
    pprint(I)

    ###############################
    gen_type = "GEN_SK_BLOBS"
    cf_split_type  = "RAND_PARTITION"
    gg_split_type  = "EQUAL_SIZE"

    n_c  = 6
    n_f  = 4
    I = get_div_k_supplier_instance(gen_type, cf_split_type, gg_split_type, rvec_type, n, d, t, k, n_c, n_f, seed, log)
    print("generator type: %s, clients-facilities: %s"%(gen_type, cf_split_type))
    pprint(I)

    cf_split_type  = "RAND_SAMPLE"
    gg_split_type  = "RAND_SIZE"
    n_c  = 8
    n_f  = 6
    I = get_div_k_supplier_instance(gen_type, cf_split_type, gg_split_type, rvec_type, n, d, t, k, n_c, n_f, seed, log)
    print("generator type: %s, clients-facilities: %s"%(gen_type, cf_split_type))
    pprint(I)
# end test_get_div_k_supplier_instance()


# testing the functions
if __name__ == "__main__":

    # test_get_fair_k_supplier_instance()
    test_get_div_k_supplier_instance()
# end if
