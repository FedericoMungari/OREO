
from scipy.stats import truncnorm
import numpy as np
import gurobipy as gp
import random
from itertools import product

from Orchestrator.utils import *


def test_scenario_init(N_services = 3, max_num_servconf = 3, priority_lvls = 3, N_functions = 10, max_n_functions_per_s = 4, max_num_compl = 3, printing_flag=False, seed_val = None):
    '''
    N_services: number of services
    priority_lvls: number of service priority levels
    N_functions: number of functions
    max_num_compl: max number of complexity factors (max number of function working points)
    '''

    # ---> seed
    np.random.seed(seed_val)  # Set the seed for numpy
    random.seed(seed_val)  # Set the seed for random
    gp.setParam('Seed', seed_val)  # Set the seed for gurobipy

    # ---> functions
    functions_dict = {}
    for _ in range(N_functions):
        # functions_dict is a dictionary with:
        # -> function names as key
        # -> function complexities as value
        functions_dict['f_%s' % (_)] = [list(range(0, np.random.randint(0, max_num_compl) + 1))]
    functions, functions_compl = gp.multidict(functions_dict)

    if printing_flag:
        print("List of functions and the relative complexities")
        display(functions_compl)

    # ---> services
    services_dict = {}
    for s_num in range(N_services):
        # services_dict is a dictionary with:
        # -> service names as key
        # -> service target latency (services_L) as first value
        # -> service target quality (services_Q) as second value
        # -> service target priority (services_P) as third value
        # -> service config (services_conf) as fourth value
        # -> service freq (service_freq) as fifth value
        services_dict['s_%s' % (s_num)] = [np.random.choice([0.1, 0.2, 0.3, 0.4, 0.5]), # np.random.uniform(0.1, 0.5),  # target latency
                                           # np.random.choice([0.5, 0.6, 0.7, 0.8, 0.9]), # np.random.uniform(0.6, 0.9),  # target quality
                                           np.random.choice([0.7, 0.75, 0.8, 0.85, 0.9]), # np.random.uniform(0.6, 0.9),  # target quality
                                           np.random.choice([1+_ for _ in range(priority_lvls)]), # np.random.randint(1, priority_lvls + 1),  # target quality
                                           # [{'cs_%s_%s' % (s_num,conf_num): random.sample(functions,np.random.randint(1,N_functions+1))}
                                           [{'cs_%s_%s' % (s_num, conf_num): random.sample(functions,
                                                                                           np.random.randint(1, max_n_functions_per_s + 1))}for conf_num in range(max_num_servconf)],
                                           1 / random.choice([0.250, 0.500, 1.000])
                                           ]
    tmp = [services_dict[k][2] for k in services_dict.keys()]
    tmp.sort(reverse = True)

    services, services_L, services_Q, services_P, services_conf, service_freq = gp.multidict(services_dict)

    for _, s in enumerate(services_P):
        services_P[s] = tmp[_]

    sp_v = list(services_P.values())
    sp_v.sort(reverse=True)
    services_P = {k: sp_v[i] for i,k in enumerate(list(services_P.keys()))}

    if printing_flag:
        print("List of services and the relative target latecy")
        display(services_L)
        print("List of services and the relative target quality")
        display(services_Q)
        print("List of services and the relative priority")
        display(services_P)
        print("List of services and the relative configurations")
        display(services_conf)
        print("List of services and the relative periodicity (ms)")
        display(service_freq)

    cs_to_s = {}
    for s in services:
        for cs in services_conf[s]:
            cs_to_s[list(cs.keys())[0]] = s


    services_conf_graph = {}
    services_conf_graph_output = {}
    for s in services:
        for cs in services_conf[s]:
            services_conf_graph[list(cs.keys())[0]] = network_construction(list(cs.values())[0])
    for s in services:
        for cs in services_conf[s]:
            services_conf_graph_output[list(cs.keys())[0]] = list(services_conf_graph[list(cs.keys())[0]].nodes())[0]
    services_conf_graph_former = {}
    for s in services:
        for cs in services_conf[s]:
            services_conf_graph_former[list(cs.keys())[0]] = [f for f in list(cs.values())[0] if f != list(services_conf_graph[list(cs.keys())[0]].nodes())[0]]

    if printing_flag:

        for s in services:
            for cs in services_conf[s]:
                print(s,list(cs.keys())[0],services_conf_graph_output[list(cs.keys())[0]])


        print("Graphs:")
        display(services_conf_graph)
        print("Graphs' output nodes:")
        display(services_conf_graph_output)


    # ---> NearRT RIC resource budget
    resource, budget = gp.multidict({
        'cpu': 1600,
        'mem': 16,
        'disk': 1024})

    if printing_flag:
        print("Near-RT RIC resource budget")
        display(budget)


    # --> xApp output data quality
    xApp_q = {}
    for s in services:
        for cs in services_conf[s]:
            for f in functions:
                if f in list(cs.values())[0]:
                    # if f is involved by cs
                    q_c = random.sample([0.5, 0.6, 0.7, 0.8, 0.9], len(functions_compl[f]))
                    q_c.sort()
                    for c in functions_compl[f]:
                        xApp_q[list(cs.keys())[0], f, c] = q_c[c]
                else:
                    # if f is NOT involved by cs
                    for c in functions_compl[f]:
                        xApp_q[list(cs.keys())[0], f, c] = 0
    # for c in functions_compl['f_0']:
    #     xApp_q['cs_0_1', 'f_0', c ] = 0.1
    # xApp_q['cs_0_1', 'f_0', 1 ] = 0.1
    if printing_flag:
        display(xApp_q)

    # check if at least one cs per service can meet the quality constraint
    for s in services:
        service_done = False
        while not service_done:
            max_expected_q = 0
            for cs in services_conf[s]:
                expected_q_cs = 0
                for f_num,f in enumerate([services_conf_graph_output[list(cs.keys())[0]]] + services_conf_graph_former[list(cs.keys())[0]]):
                    if f_num == 0:
                        expected_q_cs += 0.9 * xApp_q[list(cs.keys())[0], f, functions_compl[f][-1]]
                    else:
                        expected_q_cs += 0.1 * xApp_q[list(cs.keys())[0], f, functions_compl[f][-1]]
                if expected_q_cs > max_expected_q:
                    max_expected_q = expected_q_cs
            if max_expected_q > services_Q[s]:
                service_done = True
            else:
                # decrease services_Q[s]
                services_Q[s] = services_Q[s] - 0.05


    # --> xApp processing latency
    xApp_l = {}
    for f in functions:
        for c in functions_compl[f]:
            xApp_l[f, c] = np.random.uniform(0.01, 0.05)

    # --> xApp mem requirements
    xApp_mem_req = {}
    # for s in services:
    #     for cs in services_conf[s]:
    for f in functions:
        for r in budget:
            if r == 'disk':
                mem_req_mu = budget[r]/60
                a = (budget[r]/70 - mem_req_mu) / mem_req_mu
                b = (budget[r]/40 - mem_req_mu) / mem_req_mu
            elif r == 'mem':
                mem_req_mu = budget[r]/40
                a = (budget[r]/55 - mem_req_mu) / mem_req_mu
                b = (budget[r]/35 - mem_req_mu) / mem_req_mu
            else:
                continue
            mem_reqs = list(truncnorm.rvs(a, b, loc=mem_req_mu, scale=mem_req_mu, size=len(functions_compl[f])))
            mem_reqs.sort()
            for c in functions_compl[f]:
                xApp_mem_req[f, c, r] = mem_reqs[c]


    # for c in functions_compl['f_0']:
    #     xApp_q['cs_0_1', 'f_0', c ] = 0.1
    # xApp_q['cs_0_1', 'f_0', 1 ] = 0.1
    if printing_flag:
        display(xApp_q)


    # --> Service quality
    #     --> WILL DEPEND ON DECISION VARIABLEs

    # --> Service latency
    #     --> WILL DEPEND ON DECISION VARIABLEs

    cs_list = []
    for s in services:
        for cs in services_conf[s]:
            cs_list.append(list(cs.keys())[0])

    '''
    lambda_f is a dictionary that receives as key a tuple containing:
    - a tuple of length equal to |C_s|, and i-th element equal to 0 if the i-th cs is not using f, 1 otherwise
    - function name
    '''
    # for every function, lambda_f[((0,...,0),f)] = 0
    lambda_f = {}


    # 'theta' expresses the amount of input data processed by the xApp in a CPU cycle.")
    # Preparing a dict with '(f,c,j)' as key, and the corresponding 'theta' as value")
    theta = {}
    for f in functions:
        theta_list = random.sample([0.5, 0.55, 0.60, 0.65, 0.7, 0.75], len(functions_compl[f]))
        theta_list.sort(reverse=True)

        for c in functions_compl[f]:
            theta[f, c] = theta_list[c]


    semantic = {}
    semantic_cs = {}
    lambda_semantic = {}

    for f in functions:
        semantic[f] = {}
        lambda_semantic[f] = {}

        sem_counter = 0

        semantic[f]['sem' + str(sem_counter)] = []
        lambda_semantic[f]['sem' + str(sem_counter)] = -1

        for s in services:
            for cs in services_conf[s]:

                if f in list(cs.values())[0]:

                    p = np.random.uniform(0, 1)

                    if p < 0.5:
                        # append to latest semantic
                        lambda_semantic[f]['sem' + str(sem_counter)] = max(lambda_semantic[f]['sem' + str(sem_counter)],
                                                                           service_freq[s])
                        semantic[f]['sem' + str(sem_counter)].append(list(cs.keys())[0])

                    else:
                        # create new semantic (if the old one is not empty)
                        if semantic[f]['sem' + str(sem_counter)] != []:
                            sem_counter += 1
                            semantic[f]['sem' + str(sem_counter)] = []
                            lambda_semantic[f]['sem' + str(sem_counter)] = -1
                        semantic[f]['sem' + str(sem_counter)].append(list(cs.keys())[0])
                        lambda_semantic[f]['sem' + str(sem_counter)] = max(lambda_semantic[f]['sem' + str(sem_counter)],
                                                                           service_freq[s])

    for f in functions:
        semantic_cs[f] = {}
        for s in services:
            for cs in services_conf[s]:
                for sem in semantic[f]:
                    if list(cs.keys())[0] in semantic[f][sem]:
                        semantic_cs[f][list(cs.keys())[0]] = sem
                        break
                    semantic_cs[f][list(cs.keys())[0]] = -1


    quality_mapping_x = {}
    quality_mapping_q = {}
    for s in services:
        for cs in services_conf[s]:
            quality_mapping_x[list(cs.keys())[0]] = []
            quality_mapping_q[list(cs.keys())[0]] = []

    f_multiplier = {}
    f_multiplier_c = {}
    for f_i, f in enumerate(functions):
        f_multiplier[f] = 5 ** (f_i + 1)
        f_multiplier_c[f] = 5 ** (f_i)
        # N_functions
        # max_num_compl

    for s in services:
        for cs in services_conf[s]:
            combs = list(product(*(zip([func] * len(functions_compl[func]), functions_compl[func]) for func in list(cs.values())[0])))

            x_tmp = []
            q_tmp = []

            for comb in combs:

                x_value = 0
                for ff, cc in comb:
                    x_value += f_multiplier[ff] + f_multiplier_c[ff] * (cc)
                # print(x_value)

                q_value = 0
                for ff,cc in comb:
                    if ff == services_conf_graph_output[list(cs.keys( ))[0]]:
                        q_value += 0.9 * xApp_q[list(cs.keys())[0], ff, cc]
                    else:
                        q_value += 0.1 * xApp_q[list(cs.keys())[0], ff, cc]

                x_tmp.append(x_value)
                q_tmp.append(q_value)

            # print("len(combs): ", len(combs))
            # print("len(x_tmp): ", len(x_tmp))
            # print("len(q_tmp): ", len(q_tmp))
            # print(" ")
            sorted_values = sorted(zip(x_tmp, q_tmp))
            x_tmp, q_tmp = zip(*sorted_values)
            x_tmp = list(x_tmp)
            q_tmp = list(q_tmp)
            x_tmp.insert(0, 0)
            x_tmp.append(x_tmp[-1] + 1)
            q_tmp.insert(0, q_tmp[0])
            q_tmp.append(q_tmp[-1])
            quality_mapping_x[list(cs.keys())[0]] = x_tmp[:]
            quality_mapping_q[list(cs.keys())[0]] = q_tmp[:]

    return functions, functions_compl, services, services_L, services_Q, services_P, services_conf, cs_to_s, \
           service_freq, services_conf_graph, services_conf_graph_output, services_conf_graph_former, resource, budget,\
           xApp_q, xApp_l, xApp_mem_req, cs_list, lambda_f, theta, semantic, lambda_semantic, semantic_cs, \
           quality_mapping_x, quality_mapping_q, f_multiplier, f_multiplier_c