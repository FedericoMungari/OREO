import random

import matplotlib.pyplot as plt
import argparse
import sys
import numpy as np
from time import time
import gurobipy as gp

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

from test_scenario import test_scenario_init
from Orchestrator.lagrangian_multipliers import *
from Orchestrator.utils import *
from Orchestrator.LagrangianProblem import LP1, LP2
from Orchestrator.EnsuringFeasibility import compute_objectivefunction, EnsuringFeasibility

def main():
    printing_scenario_flag = False

    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--scenario", action = "store", required = False, type = str,
                        help = "Test scenario for the Orchestrator: ScenarioN or complete, with N being an integer number")
    parser.add_argument("-t", "--timelimit", action = "store", required = False, type = float,
                        help = "Timelimit for Gurobi models. Default value: None - no time limit")
    parser.add_argument("-ss", "--seed", action = "store", required = False, type = int,
                        help = "Seed for simulations")

    args = parser.parse_args()

    if (args.scenario is None) or not (check_substring_letter(args.scenario)):
        scenario = "ScenarioA"
    else:
        scenario = args.scenario

    if (args.seed is None) or (type(args.seed) != type(1)):
        seed_val = 8
    else:
        seed_val = args.seed

    if (type(args.timelimit) == type(1)) or (type(args.timelimit) == type(1.)):
        timelimit = args.timelimit
    else:
        timelimit = 10

    timelimit_0 = timelimit
    timelimit_lp2 = timelimit_0

    # ######################################################
    # printing phase has been removed by the original script
    # ######################################################

    if scenario == "ScenarioA":
        N_services = 2; max_num_servconf = 2; N_functions = 4; max_n_functions_per_s = 2; max_num_compl = 2; priority_lvls = 3
    elif scenario == "ScenarioB":
        N_services = 4; max_num_servconf = 3; N_functions = 6; max_n_functions_per_s = 2; max_num_compl = 3; priority_lvls = 3
    elif scenario == "ScenarioD":
        N_services = 6; max_num_servconf = 4; N_functions = 8; max_n_functions_per_s = 3; max_num_compl = 3; priority_lvls = 3
    elif scenario == "ScenarioE":
        N_services = 12; max_num_servconf = 5; N_functions = 10; max_n_functions_per_s = 4; max_num_compl = 3; priority_lvls = 3
    else:
        return -1

    np.random.seed(seed_val)  # Set the seed for numpy
    random.seed(seed_val)  # Set the seed for random
    gp.setParam('Seed', seed_val)  # Set the seed for gurobipy

    max_latency = 1
    BigM = max_latency

    functions, functions_compl, \
    services, services_L, services_Q, services_P, \
    services_conf, cs_to_s, service_freq, services_conf_graph, services_conf_graph_output, services_conf_graph_former, \
    resource, budget, \
    xApp_q, xApp_l, xApp_mem_req, \
    cs_list, lambda_f, theta, \
    semantic, lambda_semantic, semantic_cs,\
    quality_mapping_x, quality_mapping_q, f_multiplier, f_multiplier_c = \
        test_scenario_init(N_services = N_services,
                           max_n_functions_per_s = max_n_functions_per_s,
                           max_num_servconf = max_num_servconf,
                           priority_lvls = priority_lvls,
                           N_functions = N_functions,
                           max_num_compl = max_num_compl,
                           printing_flag = printing_scenario_flag,
                           seed_val = seed_val)

    f_to_cs = {}
    for f in functions:
        f_to_cs[f] = []
        for s in services:
            for cs in services_conf[s]:
                if f in list(cs.values())[0]:
                    f_to_cs[f].append(list(cs.keys())[0])


    '''
    - semantic: dictionary of the form {function: {semantic1:[cs1,cs2], semantic2:[cs1], ...}} where semantic1, 
                semantic2, ... are the available semantics of the function. For every pair (function,semantic), it
                returns the list of service configurations adopting the function in the semantic.
                e.g., semantic[f1][sem1] = [cs1, cs2] means that cs1 and cs2 foresees the exploitation of f1 in the 
                      semantic sem1
    - lambda_semantic: dictionary of the form {function: [semantic1, semantic2, ...]} where semantic1, semantic2, ...
                       are the available semantics of the function. For every pair (function,semantic), it returns the
                       load associated to the function when working in the semantic.
                       e.g., lambda_semantic[f1][sem1] = 0.5 means that the load of f1 in the semantic sem1 is 0.5
    - semantic_cs: dictionary of the form {f: {cs1:-1, cs2:'sem1', ...}} where semantic1, semantic2, ... are the
    '''

    services_L_list = []
    services_Q_list = []
    for s in services:
        services_L_list.append(services_L[s])
        services_Q_list.append(services_Q[s])

    J_MAX = len(services)
    J_MAX = 0
    for f in functions:
        max_sharing_of_f = 0
        for s in services:
            for cs in services_conf[s]:
                if f in list(cs.values())[0]:
                    max_sharing_of_f+=1
        J_MAX = max(J_MAX, max_sharing_of_f)

    normalization_factor = 0
    for s in services:
        normalization_factor += services_P[s]

    minimum_bound_gap = normalization_factor / 50

    # In[ ]:

    if printing_scenario_flag:
        for s in services:
            for cs in services_conf[s]:
                nx.draw(services_conf_graph[list(cs.keys())[0]], with_labels = True)

    if printing_scenario_flag:
        i=1
        for s in services:
            for cs in services_conf[s]:
                plt.subplot(3, 2, i)
                nx.draw(services_conf_graph[list(cs.keys())[0]],with_labels=True)
                i+=1
                if i==6:
                    break
            if i==6:
                break

    # Initialize time counters
    LP1_t_acc = 0  # lp1_t
    LP1_t_min = 1000
    LP1_t_max = 0

    LP2_t_acc = 0  # lp2_t
    LP2_t_min = 1000
    LP2_t_max = 0

    LP2_pruning_t_acc = 0  # lp2_pruning_end - lp2_pruning_start
    LP2_pruning_t_min = 1000
    LP2_pruning_t_max = 0

    LP_t_acc = 0
    LP_t_min = 1000
    LP_t_max = 0

    Subgradient_t_acc = 0
    Subgradient_t_min = 1000
    Subgradient_t_max = 0

    EnsuringFeasibility_t_acc = 0
    EnsuringFeasibility_t_min = 1000
    EnsuringFeasibility_t_max = 0

    Heuristic_t = 0  # t_start - t_end
    Heuristic_t_acc = 0
    Heuristic_t_min = 1000
    Heuristic_t_max = 0

    BoundComp_t = 0  # BoundComp_t_end - BoundComp_t_start
    BoundComp_t_acc = 0
    BoundComp_t_min = 1000
    BoundComp_t_max = 0

    # #########################################################################################################

    min_ser_p = 10
    for s in services:
        if services_P[s] < min_ser_p:
            min_ser_p = services_P[s]
    max_cs_cardinality = 0
    for s in services:
        max_cs_cardinality
        for cs in services_conf[s]:
            if len(list(cs.values())[0]) > max_cs_cardinality:
                max_cs_cardinality = len(list(cs.values())[0])

    beta_upperbound = min_ser_p / (max_cs_cardinality + BigM + 1) * 2
    gamma_upperbound = min_ser_p / (max_cs_cardinality + BigM + 1) * 2
    delta_upperbound = min_ser_p / (max_cs_cardinality + BigM + 1) * 2

    # **Lagrangian multipliers**
    beta = beta_init(services,services_conf,functions,ub=beta_upperbound)
    gamma = gamma_init(services, services_conf,ub=gamma_upperbound)
    delta = delta_init(services, services_conf,ub=delta_upperbound)

    n_binary_oreo_lp1 = []
    n_integer_oreo_lp1 = []
    n_continuous_oreo_lp1 = []
    n_constraints_oreo_lp1 = []

    n_binary_oreo_lp2 = []
    n_integer_oreo_lp2 = []
    n_continuous_oreo_lp2 = []
    n_constraints_oreo_lp2 = []

    # Model
    # LP1, LP2, z, v, n_aux, q = model_init(services, services_P, services_Q, services_conf, services_conf_graph_output, xApp_q, functions, functions_compl, beta, gamma=gamma, seed_val=seed_val,gp_printing=False)
    lp1 = LP1(services, services_P, services_Q, services_conf, services_conf_graph_output, functions, functions_compl, beta, gamma, delta, cs_list, BigM=BigM, seed_val=seed_val, gp_printing=False, timelimit=timelimit, MIPGap_v=0.01)

    # #########################################################################################################

    LB_list = []
    LB_notlagrangian_list = []
    LB_lagrangian_list = []

    ZB_list = []
    ZB_notlagrangian_list = []
    ZB_lagrangian_list = []

    LB_obj_list = []
    LB_norm_obj_list = []
    LB_q_list = []
    LB_tau_list = []
    LB_nserv_list = []

    ZB_obj_list = []
    ZB_norm_obj_list = []
    ZB_q_list = []
    ZB_tau_list = []
    ZB_nserv_list = []

    BLB = -10000
    BLB_lagrangian = +1 * BLB
    BLB_notlagrangian = +1 * BLB
    BLB_obj = -1 * BLB
    BLB_norm_obj = +1 * BLB

    BUB = -1 * BLB
    BZB_lagrangian = -1 * BLB
    BZB_notlagrangian = -1 * BLB
    BZB_obj = -1 * BLB
    BZB_norm_obj = -1 * BLB

    xApps_in_lp2_list = []
    xApps_in_ZB_list = []

    UB = 0
    ZB_obj = 0
    ZB_norm_obj = 0
    LB = 0
    LB_obj = 0
    LB_norm_obj = 0

    step_size = 1 / normalization_factor

    n_iterations = 20
    n_printing = int( n_iterations / 10 )

    # the step size will be halved every time the BLB does not improve for N_consecutive_times
    N_consecutive_times = 5

    minimum_stepsize = step_size / 4.1

    counter_stepsize = 0  # it countes the consecutive times in which the BLB does not improve

    counter_bound = 0
    N_consecutive_times_bound = 1
    # N_consecutive_times_bound = 2 #CHANGED

    # if the UB-LB is constant, stop the execution
    N_consecutive_times_gap = 5
    gap_history = []

    max_sub1 = -100
    max_sub2 = -100
    max_sub3 = -100
    min_sub1 = 100
    min_sub2 = 100
    min_sub3 = 100

    iteration_times_list = []

    counter_without_timelimit_increment_lp2 = 0
    counter_with_timelimit_increment_lp2 = 0

    counter_with_timelimit_increment_max = 3

    # z_LB = {(s,list(cs.keys())[0]):0 for s in services for cs in services_conf[s]} # we will init it later
    v_LB_0 = {(list(cs.keys())[0],f,c,j):0 for s in services for cs in services_conf[s] for f in functions for c in functions_compl[f] for j in range(1,J_MAX+1)}
    n_aux_LB_0 = {(f,c,j):0 for f in functions for c in functions_compl[f] for j in range(1,J_MAX+1)}
    rho_LB_0 = {(f,c,j,r):0 for f in functions for c in functions_compl[f] for j in range(1,J_MAX+1) for r in budget}
    tau_LB_0 = {list(cs.keys())[0]:max_latency for s in services for cs in services_conf[s]}
    q_LB_0 = {list(cs.keys())[0]:0 for s in services for cs in services_conf[s]}

    lp1_pruning_start = time()

    services_conf_pruned = {}
    for s in services:
        for cs in services_conf[s]:
            q_max_available = max_quality_comp(s, cs, functions_compl, services_Q[s], quality_mapping_x, quality_mapping_q,
                                   f_multiplier, f_multiplier_c)
            if q_max_available < services_Q[s]:
                # The service configuration cs for s is useless
                pass
            else:
                # check if s is not in services_conf_pruned keys.
                if s not in services_conf_pruned.keys():
                    services_conf_pruned[s] = [cs]
                else:
                    services_conf_pruned[s].append(cs)
    cs_list_pruned = [list(cs.keys())[0] for s in services_conf_pruned.keys() for cs in services_conf_pruned[s]]
    # print("PRUNED SERVICE CONFIGURATIONS: ", [_ for _ in cs_list if _ not in cs_list_pruned], "(%d)" % (len([_ for _ in cs_list if _ not in cs_list_pruned])))

    lp1_pruning_end = time()
    LP1_pruning_t = lp1_pruning_end - lp1_pruning_start

    t_start = time()

    for iteration in range(n_iterations):

        lp1_obj = 0
        lp2_obj = 0

        v_LB = v_LB_0.copy()
        n_aux_LB = n_aux_LB_0.copy()
        rho_LB = rho_LB_0.copy()
        tau_LB = tau_LB_0.copy()
        q_LB = q_LB_0.copy()

        Heuristic_t_iter_start = time()

        if iteration == 5:
            N_consecutive_times_gap = 3

        if iteration>0 and timelimit_0!=None and timelimit_0>0:
            timelimit = timelimit + timelimit_0
            lp2.m.setParam('TimeLimit', timelimit)

        # print(print_numb, "- Start LP1 optimization, iteration:",iteration); print_numb += 1; sys.stdout.flush()
        lp1.set_obj(services, services_P, services_Q, services_conf_pruned, functions, functions_compl)
        lp1_t_start = time()
        lp1.optimize()
        # print("LP1 OPTIMIZED - (iter %d)" % iteration)
        lp1_t_end = time()
        # print(print_numb, "- End LP1 optimization, iteration:",iteration); print_numb += 1; sys.stdout.flush()

        lp1_obj = lp1.m.ObjVal

        result = [k for k,v in lp1.z.items() if v.X >= 0.5] # It contains the (s,cs) pairs selected by LP1 solution.

        if result == []:
            '''Something wrong: the LP1 solution should always select at leas one service.'''
            print("\nEmpty result from LP1\n")
            return -1

        z_LB = {k:v.X for k,v in lp1.z.items()}
        # print("unfeasible services:", len({k:v for k,v in z_LB.items() if v>0.5}))

        lp2_pruning_start = time()

        services_prime = [r[0] for r in result]
        services_notprime = [s for s in services if s not in services_prime]
        cs_list_prime = [r[1] for r in result]
        # cs_list_notprime = [cs for cs in cs_list if cs not in cs_list_prime]
        cs_list_notprime = [cs for cs in cs_list_pruned if cs not in cs_list_prime]

        # among the service configurations "service_conf", we keep only the ones that have been selected
        # in LP1 solution
        services_conf_prime = {}
        for s,cs in result:
            services_conf_prime[s] = []
            for ccs in services_conf_pruned[s]:
                if list(ccs.keys())[0] == cs:
                    services_conf_prime[s].append(ccs)

        functions_prime_aux = []
        for s in services_prime:
            for cs in services_conf_prime[s]:
                functions_prime_aux = functions_prime_aux + list(cs.values())[0]
        functions_prime = list(dict.fromkeys(functions_prime_aux))
        J_MAX_prime = 0  # initialize counter to 0
        for item in functions_prime:
            count = functions_prime_aux.count(item)  # count occurrences in duplicated list
            if count > J_MAX_prime:
                J_MAX_prime = count

        services_conf_notprime = {}
        for s in services:
            services_conf_notprime[s] = []
            for cs in services_conf[s]:
                if list(cs.keys())[0] in cs_list_notprime:
                    services_conf_notprime[s].append(cs)

        services_conf_graph_output_prime = {k: v for k, v in services_conf_graph_output.items() if k in cs_list_prime}
        services_conf_graph_former_prime = {k: v for k, v in services_conf_graph_former.items() if k in cs_list_prime}
        # services_conf_graph_output_notprime = {k: v for k, v in services_conf_graph_output.items() if k not in cs_list_prime}
        # services_conf_graph_former_notprime = {k: v for k, v in services_conf_graph_former.items() if k not in cs_list_prime}

        semantic_prime = {}
        semantic_notprime = {}
        for f in functions:
            semantic_prime[f] = {k: shared_elements(v,cs_list_prime) for k, v in semantic[f].items()}
            semantic_notprime[f] = {k: shared_elements(v,cs_list_notprime) for k, v in semantic[f].items()}

        f_to_cs_prime = {}
        for f in functions_prime:
            f_to_cs_prime[f] = [x for x in f_to_cs[f] if x in cs_list_prime]

        # print(print_numb, "- Start LP2 definement, iteration:", iteration); print_numb += 1; sys.stdout.flush()
        lp2 = LP2(services_prime, services_P, services_Q, services_conf_prime,
                  services_conf_graph_output_prime, services_conf_graph_former_prime, J_MAX_prime,
                  quality_mapping_x, quality_mapping_q, f_multiplier, f_multiplier_c,
                  xApp_mem_req, functions, functions_compl,
                  beta, gamma, delta,
                  budget, theta, semantic_prime, lambda_semantic, cs_list_prime, services_L,
                  max_latency=max_latency, seed_val=seed_val,gp_printing=False, timelimit=timelimit_lp2, MIPGap_v=0.01) # MIPGap_v=0.5/(10*(iteration+1)))
        # print(print_numb, "- End LP2 definement, iteration:", iteration); print_numb += 1; sys.stdout.flush()

        lp2_pruning_end = time()

        lp2_t_start = time()
        while True:
            try:
                # print(print_numb, "- Start LP2 optimization, iteration:", iteration); print_numb += 1; sys.stdout.flush()
                lp2.set_obj(services_prime, services_conf_prime, functions_prime, functions_compl, J_MAX_prime, budget,services_L)
                lp2.optimize()
                # print("LP2 OPTIMIZATION TRIAL DONE - (iter %d)" % iteration)
                lp2.n_aux[list(lp2.n_aux.keys())[0]].X
                counter_without_timelimit_increment_lp2 += 1
                counter_with_timelimit_increment_lp2 = 0
                break
            except:
                print("Increasing timelimit lp2 from",timelimit_lp2,"to",timelimit_lp2 + 10*timelimit_0)
                if counter_with_timelimit_increment_lp2==0:
                    timelimit_lp2 = 10*timelimit_0
                else:
                    timelimit_lp2 = timelimit_lp2 + 10 * timelimit_0
                counter_without_timelimit_increment = 0
                counter_with_timelimit_increment_lp2+=1
                if counter_with_timelimit_increment_lp2 == counter_with_timelimit_increment_max:
                    break
                lp2.m.setParam('TimeLimit', timelimit_lp2)
        # print("LP2 OPTIMIZED - (iter %d)" % iteration)
        lp2_t_end = time()
        # print(print_numb, "- End LP2 optimization, iteration:", iteration); print_numb += 1; sys.stdout.flush()

        lp2_obj = lp2.m.ObjVal

        for k in lp2.v.keys():
            v_LB[k] = lp2.v[k].X

        for k in lp2.n_aux.keys():
            n_aux_LB[k] = lp2.n_aux[k].X

        for k in lp2.rho.keys():
            rho_LB[k] = lp2.rho[k].X


        if counter_with_timelimit_increment_lp2 == counter_with_timelimit_increment_max:
            print("LP2: No solution found.\nLaunched command:")
            print("-s "+str(scenario)+
                  " -ss "+str(seed_val)+
                  " -t "+str(timelimit_0))
            sys.stdout.flush()
            return -1

        lp1_t = lp1_t_end - lp1_t_start
        lp2_t = lp2_t_end - lp2_t_start
        lp_t = lp1_t + lp2_t

        lp2_pruning_t = lp2_pruning_end - lp2_pruning_start

        if counter_without_timelimit_increment_lp2 == 3 and timelimit != None and timelimit > 0:
            timelimit = max(timelimit - timelimit_0, timelimit_0)
            lp2.m.setParam('TimeLimit', timelimit)

        dict_tmp = {k: v for k, v in n_aux_LB.items() if v >= 0.5}
        xApps_in_lp2 = len(dict_tmp)
        xApps_in_lp2_list.append(xApps_in_lp2)

        BoundComp_t_start1 = time()

        LB_lagrangian = lp1_obj+lp2_obj

        LB_notlagrangian = compute_LB(services, services_P, services_conf, functions, functions_compl, J_MAX,
                                      z_LB, rho_LB, budget)
        LB = LB_lagrangian

        LB_obj, LB_norm_obj = compute_objectivefunction(services,
                                                        services_P,
                                                        services_conf,
                                                        functions,
                                                        functions_compl,
                                                        z_LB,
                                                        n_aux_LB,
                                                        rho_LB,
                                                        budget,
                                                        J_MAX_prime,
                                                        normalization_factor=normalization_factor)

        n_serv_counter = 0
        for key in z_LB:
            n_serv_counter += z_LB[key]

        LB_nserv_list.append(n_serv_counter)
        LB_list.append(LB)
        LB_notlagrangian_list.append(LB_notlagrangian)
        LB_lagrangian_list.append(LB_lagrangian)
        LB_obj_list.append(LB_obj)
        LB_norm_obj_list.append(LB_norm_obj)
        LB_q_list.append(q_LB)
        LB_tau_list.append(tau_LB)

        if (LB_notlagrangian > BLB_notlagrangian) and (iteration!=0):
            best_iteration_LB = iteration
            BLB = LB
            BLB_notlagrangian = LB_notlagrangian
            BLB_lagrangian = LB_lagrangian
            BLB_obj = LB_obj
            BLB_norm_obj = LB_norm_obj
            best_LP1_objfunct = lp1.m.ObjVal
            best_LP2_objfunct = lp2.m.ObjVal
            best_z_LB = z_LB.copy()
            best_v_LB = v_LB.copy()
            best_n_aux_LB = n_aux_LB.copy()
            best_rho_LB = rho_LB.copy()
            best_beta_LB = beta.copy()
            best_gamma_LB = gamma.copy()
            best_delta_LB = delta.copy()
            best_q_LB = q_LB.copy()
            best_tau_LB = tau_LB.copy()
        BoundComp_t_end1 = time()

        # ##################################################################################################################
        # UB computation
        EnsuringFeasibility_t_start = time()
        z_UB, v_UB, n_aux_UB, rho_UB, ZB_lagrangian, ZB_notlagrangian, ZB_obj, ZB_norm_obj, q_UB, tau_UB = \
            EnsuringFeasibility(services_prime, services_notprime,
                                  services_P, services_Q, services_L,
                                  services_conf_prime, services_conf_notprime,
                                  cs_list_prime, cs_to_s,
                                  services_conf_graph_output_prime, services_conf_graph_former_prime,
                                  functions, functions_compl,
                                  quality_mapping_x, quality_mapping_q, f_multiplier, f_multiplier_c,
                                  J_MAX_prime,
                                  budget,
                                  z_LB, v_LB, rho_LB, q_LB, tau_LB, n_aux_LB,
                                  semantic_prime, lambda_semantic, semantic_cs, theta,
                                  beta, gamma, delta, xApp_mem_req,
                                  max_latency=max_latency, BigM=BigM, normalization_factor=normalization_factor)
        EnsuringFeasibility_t_end = time()

        dict_tmp = {k: v for k, v in n_aux_UB.items() if v >= 0.5}
        xApps_in_UB = len(dict_tmp)
        xApps_in_ZB_list.append(xApps_in_UB)

        BoundComp_t_start2 = time()
        UB = ZB_lagrangian

        n_serv_counter = 0
        for key in z_UB:
            n_serv_counter += z_UB[key]
        ZB_nserv_list.append(n_serv_counter)
        ZB_list.append(UB)
        ZB_notlagrangian_list.append(ZB_notlagrangian)
        ZB_lagrangian_list.append(ZB_lagrangian)
        ZB_obj_list.append(ZB_obj)
        ZB_norm_obj_list.append(ZB_norm_obj)
        ZB_q_list.append(q_UB)
        ZB_tau_list.append(tau_UB)


        if (ZB_notlagrangian < BZB_notlagrangian) and (iteration!=0):
            best_iteration_UB = iteration
            BUB = UB
            BZB_notlagrangian = ZB_notlagrangian
            BZB_lagrangian = ZB_lagrangian
            BZB_obj = ZB_obj
            BZB_norm_obj = ZB_norm_obj
            best_z_UB = z_UB.copy()
            best_v_UB = v_UB.copy()
            best_n_aux_UB = n_aux_UB.copy()
            best_rho_UB = rho_UB.copy()
            best_beta_UB = beta.copy()
            best_gamma_UB = gamma.copy()
            best_delta_UB = delta.copy()
            best_q_UB = q_UB.copy()
            best_tau_UB = tau_UB.copy()

        BoundComp_t_end2 = time()

        if (ZB_notlagrangian // 0.0001 > BZB_notlagrangian // 0.0001):
                counter_stepsize += 1
        else:
            counter_stepsize = 0

        if counter_stepsize == N_consecutive_times:
            step_size = step_size / 2
            counter_stepsize = 0


        # ##################################################################################################################
        # Lagrangian multipliers update

        Subgradient_t_start = time()

        step_size_it_num = step_size * (ZB_notlagrangian - LB_notlagrangian)


        step_size_it_den_1 = (gp.quicksum(gp.quicksum(gp.quicksum(((z_LB[s,list(cs.keys())[0]] - gp.quicksum(gp.quicksum(
                v_LB[list(cs.keys())[0],f,c,j] for j in range(1,J_MAX_prime+1)) for c in functions_compl[f])).getValue())**2
            for f in list(cs.values())[0]) for cs in services_conf_prime[s]) for s in services_prime)).getValue()

        step_size_it_den_2 = (gp.quicksum(gp.quicksum((z_LB[s, list(cs.keys())[0]] * services_Q[s] - q_LB[list(cs.keys())[0]])**2 for cs in services_conf_prime[s]) for s in services_prime)).getValue()

        step_size_it_den_3 = (gp.quicksum(gp.quicksum(
                (tau_LB[list(cs.keys())[0]] - services_L[s] - BigM * (1 - z_LB[s,list(cs.keys())[0]]))**2
            for cs in services_conf_prime[s]) for s in services_prime)).getValue()

        step_size_it_den = step_size_it_den_1 + step_size_it_den_2 + step_size_it_den_3
        if step_size_it_den == 0. or step_size_it_den == 0:
            step_size_it_den = 1
        step_size_it = step_size_it_num / (step_size_it_den)

        for s in services_prime:
            for cs in services_conf_prime[s]:

                for f in list(cs.values())[0]:
                    subgradients1 = z_LB[s,list(cs.keys())[0]] - 1 * gp.quicksum(gp.quicksum(
                                   v_LB[list(cs.keys())[0],f,c,j] for j in range(1,J_MAX_prime+1)) for c in functions_compl[f])
                    # step_size_it = step_size_it_num/(step_size_it_den_1)
                    beta[s,list(cs.keys())[0],f] = beta[s,list(cs.keys())[0],f] + step_size_it * subgradients1
                    beta[s,list(cs.keys())[0],f] = beta[s,list(cs.keys())[0],f].getValue()
                    # if beta[s,list(cs.keys())[0],f]  < 0:
                    #     print("negative beta ", s, list(cs.keys())[0], f)
                    # if beta[s, list(cs.keys())[0], f] > beta_upperbound:
                    #     print("large beta", s, list(cs.keys())[0], f)
                    beta[s,list(cs.keys())[0],f] = max(beta[s,list(cs.keys())[0],f],0)
                    beta[s,list(cs.keys())[0],f] = min(beta[s,list(cs.keys())[0],f],beta_upperbound)
                    if subgradients1.getValue()/step_size_it_den > max_sub1:
                        max_sub1 = step_size_it * subgradients1.getValue()/step_size_it_den
                    elif subgradients1.getValue()/step_size_it_den < min_sub1:
                        min_sub1 = step_size_it * subgradients1.getValue()/step_size_it_den


                subgradients2 = z_LB[s, list(cs.keys())[0]] * services_Q[s] - lp2.q[list(cs.keys())[0]]
                # step_size_it = step_size_it_num / (step_size_it_den_2)
                gamma[s, list(cs.keys())[0]] = gamma[s, list(cs.keys())[0]] + step_size_it * subgradients2
                gamma[s, list(cs.keys())[0]] = gamma[s, list(cs.keys())[0]].getValue()
                # if gamma[s, list(cs.keys())[0]] < 0:
                #     print("negative gamma", s, list(cs.keys())[0])
                # if gamma[s, list(cs.keys())[0]] > gamma_upperbound:
                #     print("large gamma", s, list(cs.keys())[0])
                gamma[s, list(cs.keys())[0]] = max(gamma[s, list(cs.keys())[0]], 0)
                gamma[s, list(cs.keys())[0]] = min(gamma[s, list(cs.keys())[0]], gamma_upperbound)
                if subgradients2.getValue()/step_size_it_den > max_sub2:
                    max_sub2 = step_size_it * subgradients2.getValue()/step_size_it_den
                elif subgradients2.getValue()/step_size_it_den < min_sub2:
                    min_sub2 = step_size_it * subgradients2.getValue()/step_size_it_den


                subgradients3 = tau_LB[list(cs.keys())[0]] - services_L[s] - BigM * (1 - z_LB[s,list(cs.keys())[0]])
                try:
                    subgradients3 = subgradients3.getValue()
                except:
                    pass
                # step_size_it = step_size_it_num / (step_size_it_den_2)
                delta[s, list(cs.keys())[0]] = delta[s, list(cs.keys())[0]] + step_size_it * subgradients3
                try:
                    delta[s, list(cs.keys())[0]] = delta[s, list(cs.keys())[0]].getValue()
                except:
                    pass
                # if delta[s, list(cs.keys())[0]] < 0:
                #     print("negative delta", s, list(cs.keys())[0])
                # if delta[s, list(cs.keys())[0]] > delta_upperbound:
                #     print("large delta", s, list(cs.keys())[0])
                delta[s, list(cs.keys())[0]] = max(delta[s, list(cs.keys())[0]], 0)
                delta[s, list(cs.keys())[0]] = min(delta[s, list(cs.keys())[0]], delta_upperbound)
                if subgradients3/step_size_it_den > max_sub3:
                    max_sub3 = step_size_it * subgradients3/step_size_it_den
                elif subgradients3/step_size_it_den < min_sub3:
                    min_sub3 = step_size_it * subgradients3/step_size_it_den

        lp1.beta = beta.copy()
        lp1.gamma = gamma.copy()
        lp1.delta = delta.copy()

        lp2.beta = beta.copy()
        lp2.gamma = gamma.copy()
        lp2.delta = delta.copy()

        Subgradient_t_end = time()


        if (ZB_notlagrangian - LB_notlagrangian < minimum_bound_gap):
            counter_bound += 1
            step_size = step_size / 2
            if timelimit != None and timelimit > 0:
                lp2.m.setParam('TimeLimit', timelimit*2)
        else:
            counter_bound = 0


        t_iteration = time()
        iteration_times_list.append(t_iteration - t_start)

        Heuristic_t_iter_end = time()

        gap_history.append(ZB_notlagrangian - LB_notlagrangian)
        if len(gap_history) > N_consecutive_times_gap and all(abs(((gap_history[-1]) - gap) / (gap_history[-1])) < 0.02 for gap in gap_history[-N_consecutive_times_gap:]):
            break  # Gap has stabilized, stop iterating

        if (step_size < minimum_stepsize):
            break
        if (counter_bound == N_consecutive_times_bound) and iteration!=0:
            break

        Subgradient_t = Subgradient_t_end - Subgradient_t_start
        EnsuringFeasibility_t = EnsuringFeasibility_t_end - EnsuringFeasibility_t_start
        BoundComp_t = BoundComp_t_end2 - BoundComp_t_start2 + BoundComp_t_end1 - BoundComp_t_start1
        Heuristic_t = Heuristic_t_iter_end - Heuristic_t_iter_start

        # update the time accumulators, mins and maxes
        LP1_t_acc += lp1_t
        LP2_t_acc += lp2_t
        LP_t_acc += lp_t
        LP2_pruning_t_acc += lp2_pruning_t
        Subgradient_t_acc += Subgradient_t
        EnsuringFeasibility_t_acc += EnsuringFeasibility_t
        BoundComp_t_acc += BoundComp_t
        Heuristic_t_acc += Heuristic_t

        LP1_t_min = min(LP1_t_min, lp1_t)
        LP2_t_min = min(LP2_t_min, lp2_t)
        LP_t_min = min(LP_t_min, lp_t)
        LP2_pruning_t_min = min(LP2_pruning_t_min, lp2_pruning_t)
        Subgradient_t_min = min(Subgradient_t_min, Subgradient_t)
        EnsuringFeasibility_t_min = min(EnsuringFeasibility_t_min, EnsuringFeasibility_t)
        BoundComp_t_min = min(BoundComp_t_min, BoundComp_t)
        Heuristic_t_min = min(Heuristic_t_min, Heuristic_t)

        LP1_t_max = max(LP1_t_max, lp1_t)
        LP2_t_max = max(LP2_t_max, lp2_t)
        LP_t_max = max(LP_t_max, lp_t)
        LP2_pruning_t_max = max(LP2_pruning_t_max, lp2_pruning_t)
        Subgradient_t_max = max(Subgradient_t_max, Subgradient_t)
        EnsuringFeasibility_t_max = max(EnsuringFeasibility_t_max, EnsuringFeasibility_t)
        BoundComp_t_max = max(BoundComp_t_max, BoundComp_t)
        Heuristic_t_max = max(Heuristic_t_max, Heuristic_t)

        n_binary_oreo_lp1.append(sum(1 for v in lp1.m.getVars() if v.VType == gp.GRB.BINARY))
        n_integer_oreo_lp1.append(sum(1 for v in lp1.m.getVars() if v.VType == gp.GRB.INTEGER))
        n_continuous_oreo_lp1.append(sum(1 for v in lp1.m.getVars() if v.VType == gp.GRB.CONTINUOUS))
        n_constraints_oreo_lp1.append(len(lp1.m.getConstrs()))

        try:
            n_binary_lp2_tmp = sum(1 for v in lp2.m.getVars() if v.VType == gp.GRB.BINARY)
            n_integer_lp2_tmp = sum(1 for v in lp2.m.getVars() if v.VType == gp.GRB.INTEGER)
            n_continuous_lp2_tmp = sum(1 for v in lp2.m.getVars() if v.VType == gp.GRB.CONTINUOUS)
            n_constraints_lp2_tmp = len(lp2.m.getConstrs())
        except:
            n_binary_lp2_tmp = 0
            n_integer_lp2_tmp = 0
            n_continuous_lp2_tmp = 0
            n_constraints_lp2_tmp = 0

        n_binary_oreo_lp2.append(n_binary_lp2_tmp)
        n_integer_oreo_lp2.append(n_integer_lp2_tmp)
        n_continuous_oreo_lp2.append(n_continuous_lp2_tmp)
        n_constraints_oreo_lp2.append(n_constraints_lp2_tmp)

    t_end = time()

    LP1_t_avg = LP1_t_acc / (iteration+1)
    LP2_t_avg = LP2_t_acc / (iteration+1)
    LP2_pruning_t_avg = LP2_pruning_t_acc / (iteration+1)
    LP_t_avg = LP_t_acc / (iteration+1)
    Subgradient_t_avg = Subgradient_t_acc / (iteration+1)
    EnsuringFeasibility_t_avg = EnsuringFeasibility_t_acc / (iteration+1)
    BoundComp_t_avg = BoundComp_t_acc / (iteration+1)
    Heuristic_t_avg = Heuristic_t_acc / (iteration+1)


    # In[ ]:


    # In[ ]:

    LatencyList_feasible = []
    for k in best_tau_UB:
            LatencyList_feasible.append(best_tau_UB[k])

    QualityList_feasible = []
    for k in best_q_UB:
        QualityList_feasible.append(best_q_UB[k])

    LatencyList_unfeasible = []
    for k in best_tau_LB:
        try:
            LatencyList_unfeasible.append(best_tau_LB[k])
        except:
            LatencyList_unfeasible.append(max_latency)

    QualityList_unfeasible = []
    for k in best_q_LB:
        try:
            QualityList_unfeasible.append(best_q_LB[k])
        except:
            QualityList_unfeasible.append(0)

    TotCPU_feasible = 0
    for k in best_rho_UB:
        if k[-1] == 'cpu':
            TotCPU_feasible += best_rho_UB[k]

    TotCPU_unfeasible = 0
    for k in best_rho_LB:
        if k[-1] == 'cpu':
            TotCPU_unfeasible += best_rho_LB[k]

    TotRAM_feasible = 0
    for k in best_rho_UB:
        if k[-1] == 'mem':
            TotRAM_feasible += best_rho_UB[k]

    TotRAM_unfeasible = 0
    for k in best_rho_LB:
        if k[-1] == 'mem':
            TotRAM_unfeasible += best_rho_LB[k]

    TotDisk_feasible = 0
    for k in best_rho_UB:
        if k[-1] == 'disk':
            TotDisk_feasible += best_rho_UB[k]

    TotDisk_unfeasible = 0
    for k in best_rho_LB:
        if k[-1] == 'disk':
            TotDisk_unfeasible += best_rho_LB[k]


    print("\n*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")
    print("*-*-*-*-*-*TOTAL OREO TIME:,", t_end - t_start,"*-*-*-*-*-*-*")
    print("*-*-*-*-*-*OREO NORM. OBJ:,", BZB_norm_obj,"*-*-*-*-*-*-*")
    print("*-*-*-*-*-*OREO SERVICES:,", max(ZB_nserv_list),"*-*-*-*-*-*-*")
    print("*-*-*-*-*-*OREO CPU:,", TotCPU_feasible, "*-*-*-*-*-*-*-*")
    print("*-*-*-*-*-*OREO DISK:,", TotDisk_feasible, "*-*-*-*-*-*-*-*")
    print("*-*-*-*-*-*OREO MEM:,", TotRAM_feasible, "*-*-*-*-*-*-*-*")
    print("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")

    return best_z_UB, best_v_UB, best_rho_UB

    # #################################################################################################################

if __name__ == '__main__':

    main()
