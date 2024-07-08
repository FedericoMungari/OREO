import random
import copy

from Orchestrator.utils import *

# AUXILIARY FUNCTIONS #################################################################################################
def var_init(services, services_conf, functions, functions_compl, theta, budget, J_MAX, semantic, lambda_semantic, semantic_cs, cs_to_s, services_L, max_latency, v, z, rho, q, tau, n_aux):

    z_UB = {}
    v_UB = {}
    n_aux_UB = {}
    rho_UB = {}
    q_UB = {}
    tau_UB = {}
    n_aux_prime = {}
    lambda_aux_UB = {}
    lambda_aux_prime = {}

    try:
        for k in z:
            z_UB[k] = z[k].X
    except:
        for k in z:
            z_UB[k] = z[k]

    try:
        for k in q:
            q_UB[k] = q[k].X
    except:
        for k in q:
            q_UB[k] = q[k]

    try:
        for k in tau:
            tau_UB[k] = tau[k].X
    except:
        for k in tau:
            tau_UB[k] = tau[k]
    try:
        for k in v:
            v_UB[k] = v[k].X
    except:
        for k in v:
            v_UB[k] = v[k]

    try:
        for k in rho:
            rho_UB[k] = rho[k].X
    except:
        for k in rho:
            rho_UB[k] = rho[k]

    try:
        for k in n_aux:
            n_aux_UB[k] = n_aux[k].X
            n_aux_prime[k] = n_aux[k].X
    except:
        for k in n_aux:
            n_aux_UB[k] = n_aux[k]
            n_aux_prime[k] = n_aux[k]

    for f in functions:
        for c in functions_compl[f]:
            for j in range(1, J_MAX + 1):
                for sem in semantic[f]:
                    lambda_aux_UB[f, sem, c, j] = 0
                    lambda_aux_prime[f, sem, c, j] = 0

    # --> lambda_aux_prime definition based on z[s,cs] and v[cs,f,c,j] (LP1 and LP2 solutions)
    for f in semantic:
        for sem in semantic[f]:
            for c in functions_compl[f]:
                for j in range(1, J_MAX + 1):
                    for cs in semantic[f][sem]:
                        if (v[cs, f, c, j] == 1) and (z[cs_to_s[cs], cs] == 1):
                            lambda_aux_prime[f, sem, c, j] = 1
                            lambda_aux_UB[f, sem, c, j] = 1

    for s in services:
        for cs in services_conf[s]:
            if z[s, list(cs.keys())[0]] == 1:
                for c in functions_compl[f]:
                    for j in range(1, J_MAX + 1):
                        try:
                            if v_UB[list(cs.keys())[0], f, c, j] == 1:
                                rho_UB[f, c, j, 'cpu'] = max(rho_UB[f, c, j, 'cpu'], 1 / theta[f, c] * (
                                            len(list(cs.values())[0]) / services_L[s] + lambda_semantic[f][
                                        semantic_cs[f][list(cs.keys())[0]]]))
                        except:
                            pass

    return z_UB, v_UB, n_aux_UB, rho_UB, q_UB, tau_UB, n_aux_prime, lambda_aux_UB, lambda_aux_prime

# Check constraint
def check_constraints(services, services_Q, services_L, services_conf, services_conf_graph_output, services_conf_graph_former, functions, functions_compl, J_MAX, xApp_q, z_UB, v_UB, n_aux_UB, rho_UB, q_UB, tau_UB, semantic, semantic_cs, lambda_semantic, lambda_aux_UB, budget, theta, removed_services_flag):

    for s in services:
        z_s_c_check = 0
        for cs in services_conf[s]:
            z_s_c_check += z_UB[s, list(cs.keys())[0]]
        if z_s_c_check > 1:
            print("Not feasible choice over z[s,cs]")
            return False
    for s in services:
        for cs in services_conf[s]:
            for f in list(cs.values())[0]:
                n_xapp = 0
                for c in functions_compl[f]:
                    for j in range(1, J_MAX + 1):
                        n_xapp += v_UB[list(cs.keys())[0], f, c, j]
                if n_xapp != z_UB[s, list(cs.keys())[0]]:
                    print("Not feasible choice over v[cs,f,c,j] with f in cs")
                    print("s:", s)
                    print("cs:", list(cs.keys())[0])
                    print("cs functions:", list(cs.values())[0])
                    print("n_xapp:", n_xapp)
                    print("expected n_xApp if implemented:", list(cs.values())[0])
                    print("expected n_xApp ", z_UB[s, list(cs.keys())[0]] * len(list(cs.values())[0]))
                    print("z_UB:", z_UB)
                    print("v_UB:", v_UB)
                    print("f in cs:", list(cs.values())[0])
                    return False

    for s in services:
        for cs in services_conf[s]:
            for f in [ff for ff in functions if ff not in list(cs.values())[0]]:
                n_xapp = 0
                for c in functions_compl[f]:
                    for j in range(1, J_MAX + 1):
                        n_xapp += v_UB[list(cs.keys())[0], f, c, j]
                if n_xapp != 0:
                    print("Not feasible choice over v[cs,f,c,j] with f not in cs")
                    return False

    for f in functions:
        for c in functions_compl[f]:
            for j in range(1,J_MAX+1):
                for jj in range(j+1,J_MAX+1):
                    if n_aux_UB[f,c,jj] > n_aux_UB[f,c,j]:
                        print("Not feasible ordering:",(f,c,j), (f,c,jj))
                        # return False

    for s in services:
        for cs in services_conf[s]:
            qcs = 0
            f_out = services_conf_graph_output[list(cs.keys())[0]]
            for f in list(cs.values())[0]:
                for c in functions_compl[f]:
                    for j in range(1, J_MAX + 1):
                        if v_UB[list(cs.keys())[0], f, c, j]==1:
                            if f==f_out:
                                qcs += xApp_q[list(cs.keys())[0],f,c] * 0.9
                                f_out_chosen_c = c
                            else:
                                qcs += 0.1 * xApp_q[list(cs.keys())[0],f,c]

            if  qcs != q_UB[list(cs.keys())[0]]:
                if not removed_services_flag:
                    print("Warning while checking quality constraint")
                    print(qcs)
                    print(q_UB[list(cs.keys())[0]])
            if  qcs < services_Q[s] * z_UB[s, list(cs.keys())[0]]:
                print("Not feasible choice over q[cs]")
                print("s:", s)
                print("cs:", list(cs.keys())[0])
                print("f in cs:", list(cs.values())[0])
                print("f_out:", f_out)
                print("f_out chosen c:", f_out_chosen_c)
                print("f_out quality:", xApp_q[list(cs.keys())[0],f_out,f_out_chosen_c])
                print("z[s,cs]:", z_UB[s, list(cs.keys())[0]])
                print("Service quality target:", services_Q[s])
                print("Expected quality q:", qcs)
                return False


    for s in services:
        for cs in services_conf[s]:
            if z_UB[s, list(cs.keys())[0]]==1:
                tcs = 0
                for f in list(cs.values())[0]:
                    for c in functions_compl[f]:
                        for j in range(1, J_MAX + 1):
                            if v_UB[list(cs.keys())[0], f, c, j]==1:
                                lambda_xApp = 0
                                for sem in semantic[f]:
                                    lambda_xApp += lambda_aux_UB[f,sem,c,j] * lambda_semantic[f][sem]
                                tcs += 1 / ( rho_UB[f,c,j,'cpu'] * theta[f, c] -

                             lambda_xApp)
                if tcs > services_L[s]:
                    print("Not feasible choice over tau[cs]")
                    print("s:", s)
                    print("cs:", list(cs.keys())[0])
                    print("f in cs:", list(cs.values())[0])
                    for f in list(cs.values())[0]:
                        for c in functions_compl[f]:
                            for j in range(1, J_MAX + 1):
                                if v_UB[list(cs.keys())[0],f,c,j] == 1:
                                    print(f,c,j,rho_UB[f,c,j,'cpu'])
                    print("lambda_aux_UB:", lambda_aux_UB)
                    print("lambda_semantic:", lambda_semantic)
                    print("Service actual latency:", tcs)
                    print("Service latency target:", services_L[s])
                    return False


    for r in budget:
        consumption = 0
        for f in functions:
            for c in functions_compl[f]:
                for j in range(1, J_MAX+1):
                    consumption += n_aux_UB[f,c,j] * rho_UB[f,c,j,r]
        if consumption > budget[r]:
            print(r,": Not feasible choice over rho[ f, c, j,",r,"]")
            print("budget[r]:", budget[r])
            print("consumption:", consumption)
            for f in functions:
                for c in functions_compl[f]:
                    for j in range(1, J_MAX+1):
                        if rho_UB[f,c,j,r] != 0:
                            print("n_aux_UB[f,c,j]:",f,c,j,n_aux_UB[f,c,j])
                            print("rho_UB[f,c,j,r]:",f,c,j,r,rho_UB[f,c,j,r])
            return False

    return True

# Compute UB and objective functions
def compute_UB_lagrangianmult(services, services_notprime, services_P, services_L, services_Q, services_conf, services_conf_notprime, functions, functions_compl, J_MAX, z_UB, v_UB, rho_UB, q_UB, tau_UB, beta, gamma, delta, budget, BigM,max_latency):
    UB = 0
    for s in services:
        for cs in services_conf[s]:
            UB += z_UB[s, list(cs.keys())[0]] * (-1 * services_P[s])
            for f in list(cs.values())[0]:
                UB += (+1) * z_UB[s, list(cs.keys())[0]] * beta[s,list(cs.keys())[0],f]
            UB += (+1) * z_UB[s, list(cs.keys())[0]] * gamma[s, list(cs.keys())[0]] * services_Q[s]
            UB += (-1) * (1 - z_UB[s, list(cs.keys())[0]]) * delta[s, list(cs.keys())[0]] * BigM

    for f in functions:
        for c in functions_compl[f]:
            for j in range(1, J_MAX + 1):
                for r in budget:
                    try:
                        UB += (1/3) * rho_UB[f, c, j, r] / budget[r]
                    except:
                        pass
    for s in services:
        for cs in services_conf[s]:
            for f in list(cs.values())[0]:
                for c in functions_compl[f]:
                    for j in range(1, J_MAX + 1):
                        UB += (-1) * beta[s, list(cs.keys())[0], f] * v_UB[list(cs.keys())[0], f, c, j]
            UB += (-1) * gamma[s, list(cs.keys())[0]] * q_UB[list(cs.keys())[0]]
            UB += (+1) * delta[s, list(cs.keys())[0]] * (tau_UB[list(cs.keys())[0]] - services_L[s])

    for s in services_notprime:
        for cs in services_conf_notprime[s]:
            # UB += (+1) * delta[s, list(cs.keys())[0]] * (max_latency)
            UB += (+1) * delta[s, list(cs.keys())[0]] * (max_latency - services_L[s])
    return UB

def compute_UB(services, services_P, services_conf, functions, functions_compl, J_MAX, z_UB, rho_UB, budget):

    UB = 0
    for s in services:
        for cs in services_conf[s]:
            UB += z_UB[s, list(cs.keys())[0]] * (-1 * services_P[s])
    for f in functions:
        for c in functions_compl[f]:
            for j in range(1, J_MAX + 1):
                for r in budget:
                    try:
                        UB += (1/3) * rho_UB[f, c, j, r] / budget[r]
                    except:
                        pass
    return UB

def compute_objectivefunction(services, services_P, services_conf, functions, functions_compl, z, n_aux, rho, budget, J_MAX, normalization_factor=None):
    '''
    compute original objective function
    '''

    if normalization_factor == None:
        normalization_factor = 1

    obj = 0
    # obj_s = 0
    for s in services:
        for cs in services_conf[s]:
            obj += z[s, list(cs.keys())[0]] * services_P[s]

    for f,c,j,r in rho:
        obj += (-1/3) * rho[f, c, j,r] / budget[r]
    # try:
    #     print("Objective fnct resource component", obj.getValue() - obj_s.getValue())
    #     print("Final objective fnct",obj.getValue())
    # except:
    #     print("Objective fnct resource component",obj - obj_s)
    #     print("Final objective fnct",obj)
    #     print(" ")


    # consumption = {}
    # for r in budget:
    #     consumption[r] = 0
    #     for f in functions:
    #         for c in functions_compl[f]:
    #             for j in range(1, J_MAX + 1):
    #                 consumption[r] = consumption[r] + rho[f, c, j,r]

    try:
        obj = obj.getValue()
    except:
        pass

    norm_obj = obj / normalization_factor
    return obj, norm_obj

def share_or_not(rho1_l, rho2_l, consumptions, decay_rate = 1.0):

    # check if rho1_l and rho1_l are lists, and decay_rate is a integer or a float
    if type(rho1_l) != list or type(rho2_l) != list or type(consumptions) != list or not (type(decay_rate) == int or type(decay_rate) == float):
        raise TypeError("utilization, resource_allocation must be lists. decay_rate must be a number")
    if (len(rho1_l) != len(rho2_l)) or (len(rho1_l) != len(consumptions)) or (len(rho2_l) != len(consumptions)):
        raise ValueError("rho1_l, rho2_l and consumption must have same number of elements")
    if any([u <= 0 for u in consumptions]):
        raise ValueError("consumptions must be in (0,1]")
    # if any([r <= 0 or r > 1 for r in resource_allocation]):
    #     raise ValueError("consumption must be in (0,1]")

    # if any element of consumption is largen than one, better not to share
    if any([c > 1 for c in consumptions]):
        return 1

    # Calculate the saturation score using exponential decay
    weights = [(1 - math.exp(-decay_rate * _)) for _ in consumptions]


    score = sum([((r1 - r2)*w)/sum(weights) for r1,r2,w in zip(rho1_l,rho2_l,weights)])

    # Select the scenario with the higher overall score
    if score < 0:
        # scenario = "Scenario 1"
        return 0
    else:
        # scenario = "Scenario 2"
        return 1
    # return "Chosen scenario is: " + scenario + " with score: " + str(score)

def xApp_score(rho_list, consumptions, decay_rate = 1.0):

    # check if rho1_l and rho1_l are lists, and decay_rate is a integer or a float
    if type(rho_list) != list or type(consumptions) != list or not (type(decay_rate) == int or type(decay_rate) == float):
        raise TypeError("rho_list and consumptions must be lists.\ndecay_rate must be a number")
    if (len(rho_list) != len(consumptions)):
        raise ValueError("rho_list and consumption must have same number of elements")
    if any([u < 0 for u in consumptions]):
        raise ValueError("consumptions must be in (0,1].\nInput consumptions: ", consumptions)

    # Calculate the saturation score using exponential decay
    weights = [(1 - math.exp(-decay_rate * _)) for _ in consumptions]

    score = sum([(r1*w)/sum(weights) for r1,w in zip(rho_list,weights)])
    return score

def sharing_score_comp(f, c, j, s, cs, services, services_conf, z, v, lambda_aux_UB, semantic, lambda_semantic, semantic_cs, theta, budget, xApp_mem_req, services_L, current_res_utiliz):
    '''
    :param rho: xApp resource allocation
    :param f,c,j: xApp we are evalutating to be shared by s1
    :param cs: service configuration chosen for s1
    :param lambda_semantic:
    :param lambda_aux_prime:
    :param semantic_cs:
    :param theta:
    :param budget:
    :return: xApp sharing score
    '''

    '''
    Instead of trusting the resource allocation given by the LP2 solution, we compute the sharing score based on the
    service latency targets. Ofc we have to make a simplification i.e., assign a desired latency of the xApps equal to
    the target latency of the service, divided by the number of functions in the configurations.
    '''
    actual_load = 0
    for sem in semantic[f]:
        actual_load += lambda_aux_UB[f,sem,c,j] * lambda_semantic[f][sem]
    sharing_metric = []
    for ss in services:
        for css in services_conf[ss]:
            try:
                if v[list(css.keys())[0],f,c,j] == 1 and z[ss, list(css.keys())[0]] == 1:
                    sharing_metric.append((css, services_L[ss] / len(list(css.values())[0])))
                    min_sharing_metric = services_L[ss]
            except:
                if v[list(css.keys())[0],f,c,j].X == 1 and z[ss, list(css.keys())[0]].X == 1:
                    sharing_metric.append((css, services_L[ss] / len(list(css.values())[0])))
                    min_sharing_metric = services_L[ss]
    # given the sharing_metric list, take the minimum
    min_sharing_metric = [min(min_sharing_metric,_[1]) for _ in sharing_metric][-1]
    # css_with_min_sharing = [_[0] for _ in sharing_metric if _[1] == min_sharing_metric]
    rho_CPU_notsharing = 1/theta[f,c] * (1/min_sharing_metric + actual_load) # the CPU needed by the xApp when not shared
    rho_CPU_dedicated = 1/theta[f,c] * (len(list(cs.values())[0])/services_L[s] + lambda_semantic[f][semantic_cs[f][list(cs.keys())[0]]]) # the CPU needed by a new xApp, dedicated for the new service
    sharing_load = actual_load + lambda_semantic[f][semantic_cs[f][list(cs.keys())[0]]]
    rho_CPU_sharing = 1/theta[f,c] * (1/min(min_sharing_metric, services_L[s]/len(list(cs.values())[0])) + sharing_load)

    rho_sharing = {}
    rho_notsharing = {}
    rho_sharing['cpu'] = rho_CPU_sharing/budget['cpu']
    rho_notsharing['cpu'] = (rho_CPU_notsharing + rho_CPU_dedicated)/budget['cpu']

    for r in budget:
        if r != 'cpu':
            rho_sharing[r] = (xApp_mem_req[f,c,r]) / budget[r]
            rho_notsharing[r] = (2*xApp_mem_req[f,c,r]) / budget[r]


    sharing_score = share_or_not([rho_sharing[r] for r in budget],
                                           [rho_notsharing[r] for r in budget],
                                           [current_res_utiliz[r]/budget[r] for r in budget])

    if sharing_score<0:
        return sharing_score, rho_CPU_sharing
    else:
        return None, rho_CPU_notsharing

def check_number_of_functions(input_list):
    '''
    :param input_list: a list of tuples e.g., [(f1,c1,j1),(f1,c1,j2),(f2,c2,j2),(f3,c3,j3)]
    :return: the number of different functions provided by the xApps in input_list. In the example, output = 3
    '''
    output_list = []
    unique_first_elements = set()

    for tup in input_list:
        if tup[0] not in unique_first_elements:
            unique_first_elements.add(tup[0])
            output_list.append(tup)

    return len(output_list)

def act_quality_comp(current_config, cs, quality_mapping_x, quality_mapping_q, f_multiplier, f_multiplier_c):
    '''
    Given the current setting, compute the actual expected quality
    '''
    actual_quality = 0
    if len(current_config) > len(list(cs.values())[0]):
        print("Warning in quality computation: too many xApps")
        print("current_config :",current_config)
        print("cs :",cs)
    elif len(current_config) < len(list(cs.values())[0]):
        print("Warning in quality computation: not enough xApps")
        print("current_config :",current_config)
        print("cs :",cs)
    elif check_number_of_functions(current_config) > len(list(cs.values())[0]):
        print("Warning in quality computation: too many functions")
        print("current_config :",current_config)
        print("cs :",cs)
    elif check_number_of_functions(current_config) < len(list(cs.values())[0]):
        print("Warning in quality computation: not enough functions")
        print("current_config :",current_config)
        print("cs :",cs)
    else:
        # for f_num, (ff, cc, jj) in enumerate(current_config):
        #     if f_num == 0:
        #         actual_quality += xApp_q[list(cs.keys())[0], ff, cc] * 0.9
        #     else:
        #         actual_quality += xApp_q[list(cs.keys())[0], ff, cc] * 0.1

        x_tot = 0
        for f_num, (ff, cc, jj) in enumerate(current_config):
            x_tot += f_multiplier[ff] + f_multiplier_c[ff] * cc
        # find in the list quality_mapping_x[list(cs.keys())[0]] the element with the closest value to x_tot, and get its index
        idx = (np.abs(np.array(quality_mapping_x[list(cs.keys())[0]]) - x_tot)).argmin()
        actual_quality = quality_mapping_q[list(cs.keys())[0]][idx]

        # for f_num, (ff, cc, jj) in enumerate(current_config):
            # if f_num != 0:
                # actual_quality += xApp_q[list(cs.keys())[0], ff, cc]
        # # actual_quality = actual_quality / max(len(current_config)-1, 1)
        # actual_quality += xApp_q[list(cs.keys())[0], current_config[0][0], current_config[0][1]]
        return actual_quality

def act_latency_comp(current_config,cs,lambda_aux_UB,semantic,lambda_semantic,theta,max_latency):
    actual_latency = 0
    if len(current_config) < len(list(cs.values())[0]):
        print("Warning in latency computation: not enough xApps")
        print("current_config :",current_config)
        print("cs :",cs)
        actual_latency = max_latency
    elif len(current_config) > len(list(cs.values())[0]):
        print("Warning in latency computation: too many xApps")
        print("current_config :",current_config)
        print("cs :",cs)
        actual_latency = max_latency
    elif check_number_of_functions(current_config) < len(list(cs.values())[0]):
        print("Warning in latency computation: not enough functions")
        print("current_config :",current_config)
        print("cs :",cs)
        actual_latency = max_latency
    elif check_number_of_functions(current_config) > len(list(cs.values())[0]):
        print("Warning in latency computation: too many xApps")
        print("current_config :",current_config)
        print("cs :",cs)
        actual_latency = max_latency
    else:
        for (ff, cc, jj, cpu) in current_config:
            lambda_xApp = 0
            for sem in semantic[ff]:
                lambda_xApp += lambda_aux_UB[ff,sem,cc,jj] * lambda_semantic[ff][sem]
            actual_latency += 1 / max(( cpu * theta[ff, cc] - lambda_xApp), 1/max_latency)

    return actual_latency # max_latency if any warning

def act_latency_comp_from_v(css, v_UB, rho_UB, tau_UB, lambda_aux_UB,semantic,lambda_semantic,theta,max_latency,functions_compl,J_MAX):
    current_config = []
    for f in list(css.values())[0]:
        for c in functions_compl[f]:
            for j in range(1, J_MAX+1):
                if v_UB[list(css.keys())[0],f,c,j] == 1:
                    current_config.append((f,c,j,rho_UB[f,c,j,'cpu']))
    return act_latency_comp(current_config, css, lambda_aux_UB, semantic, lambda_semantic, theta, max_latency)

def new_instance(v_UB,f,c,services,services_conf,functions_compl,J_MAX):
    '''
    Iterate over the xApp instances and check if there is at least one service/serv.conf using such an xApp.
    If yes (counter>1), keep iterating through the xApp instances.
    If not (counter=0), return the first instance number (jj) that is not used by any service/serv.conf.
    '''
    for jj in range(1, J_MAX+1):
        counter = 0
        for ss in services:
            for css in services_conf[ss]:
                # try:
                if v_UB[list(css.keys())[0],f,c,jj] == 1:
                    counter += 1
                    break
        if counter == 0:
            return jj
    return None

def remove_service_s(s,cs,z_UB, v_UB, n_aux_UB, rho_UB, lambda_aux_UB, budget, functions, functions_compl, services, services_P, services_conf, semantic_cs, J_MAX):

    z_UB[s, list(cs.keys())[0]] = 0
    serv_list = {}
    for f in list(cs.values())[0]:
        for c in functions_compl[f]:
            for j in range(1, J_MAX+1):
                # we have fixed the xApp
                if v_UB[list(cs.keys())[0], f, c, j] == 1:
                    # we have fixed the xApp used by s in configuration cs
                    serv_list[f, c, j] = []
                    for ss in services:
                        for css in services_conf[ss]:
                            if v_UB[list(css.keys())[0], f, c, j] == 1:
                                serv_list[f, c, j].append(list(css.keys())[0])

                    v_UB[list(cs.keys())[0], f, c , j] = 0 # we disconnect f[c,j] from cs. In order to understand if we can deactivate it or not,
                       # we must look at serv_list[f, c, j].
                    if len(serv_list[f, c, j]) == 1:
                        n_aux_UB[f, c, j] = 0
                        for rr in budget:
                            rho_UB[f, c, j, rr] = 0
                    lambda_aux_UB[f, semantic_cs[f][list(cs.keys())[0]], c, j] = 0
                    for css in serv_list[f, c, j]:
                        if css != list(cs.keys())[0]:
                            # lambda_aux_prime[f1, semantic_cs[f1][css], c1, j1] = 1
                            lambda_aux_UB[f, semantic_cs[f][css], c, j] = 1

    return z_UB, v_UB, n_aux_UB, rho_UB, lambda_aux_UB

def ServiceLatencyAdjustment(services,s,services_P,services_L,services_conf,cs,cs_to_s,services_conf_graph_output,services_conf_graph_former,functions,functions_compl,quality_mapping_x, quality_mapping_q, f_multiplier, f_multiplier_c,J_MAX,budget,z_UB,v_UB,rho_UB,tau_UB,n_aux_UB,lambda_aux_UB,lambda_aux_prime,semantic,lambda_semantic,semantic_cs,theta,max_latency):

    current_config = []
    # it is important to scroll the f in list(cs.values())[0] starting from the output function, because of --> look at the definition of act_quality_comp
    for f in [services_conf_graph_output[list(cs.keys())[0]]] + services_conf_graph_former[list(cs.keys())[0]]:
        for c in functions_compl[f]:
            for j in range(1, J_MAX+1):
                if v_UB[list(cs.keys())[0], f, c, j] == 1:
                    current_config.append((f,c,j,rho_UB[f,c,j,'cpu']))

    original_config = copy.deepcopy(current_config)
    best_config = copy.deepcopy(current_config)

    actual_latency = act_latency_comp(current_config,cs,lambda_aux_UB,semantic,lambda_semantic,theta,max_latency)

    while actual_latency > services_L[s]:
        new_latency = actual_latency
        increment = max((actual_latency - services_L[s]) * 10 / actual_latency, 1) # cpu increment based on the
                                                                                     # target-actual latencies
                                                                                     # difference
                                                                                     # minimum cpu increment: 10
        increment = min(increment,10)
        improvement = (-1000, -1000)
        for trial in range(len(current_config)):
            new_config = copy.deepcopy(current_config)
            new_config[trial] = list(new_config[trial])
            if new_config[trial][3] + increment < budget['cpu']:
                new_config[trial][3] = min(new_config[trial][3] + increment, budget['cpu'])
                new_config[trial] = tuple(new_config[trial])
                current_improvement = actual_latency - act_latency_comp(new_config, cs, lambda_aux_UB, semantic, lambda_semantic, theta, max_latency)
                if current_improvement > improvement[-1]:
                    improvement = (new_config, current_improvement)
                    best_config = copy.deepcopy(new_config)
                    new_latency = actual_latency - current_improvement
            # else:
            #     print("current_improvement:",current_improvement)

        if improvement == (-1000, -1000):
            break

        current_config = copy.deepcopy(best_config)
        actual_latency = new_latency

    if actual_latency > services_L[s]:
        z_UB, v_UB, n_aux_UB, rho_UB, lambda_aux_UB = remove_service_s(s,cs,z_UB, v_UB, n_aux_UB, rho_UB, lambda_aux_UB, budget, functions, functions_compl, services, services_P, services_conf, semantic_cs, J_MAX)
    else:
        # at this point, cs configuration has NOT been changed. But the xApp cpu resource allocation did.
        lat_recompute_services_list = []
        for (f2,c2,j2,cpu2) in current_config:
            rho_UB[f2,c2,j2,'cpu'] = cpu2
            for ss in services:
                for css in services_conf[ss]:
                    if v_UB[list(css.keys())[0],f2,c2,j2] == 1:
                        lat_recompute_services_list.append(css)


        tau_UB[list(cs.keys())[0]] = actual_latency
        for css in lat_recompute_services_list:
            tau_UB[list(css.keys())[0]] = act_latency_comp_from_v(css, v_UB, rho_UB, tau_UB, lambda_aux_UB,semantic,lambda_semantic,theta,max_latency,functions_compl,J_MAX)

    return z_UB, v_UB, rho_UB, tau_UB, n_aux_UB, lambda_aux_UB, lambda_aux_UB

def xAppResourceFinetuning(services,services_L,services_conf,cs_to_s,services_conf_graph_output,services_conf_graph_former,functions,functions_compl,J_MAX,budget,z_UB,v_UB,rho_UB,tau_UB,n_aux_UB,lambda_aux_UB,semantic,lambda_semantic,semantic_cs,theta,max_latency):
    for f in functions:
        for c in functions_compl[f]:
            for j in range(1, J_MAX+1):
                serv_list = []
                for ss in services:
                    for css in services_conf[ss]:
                        try:
                            if v_UB[list(css.keys())[0], f, c, j] == 1:
                                serv_list.append((ss,css))
                        except:
                            pass
                if serv_list != []:
                    for cpu_delta in [10,5]:
                        while True:
                            check_increment = True
                            if rho_UB[f,c,j,'cpu'] - cpu_delta < 0:
                                break
                            rho_UB[f,c,j,'cpu'] = rho_UB[f,c,j,'cpu'] - cpu_delta
                            for ss,css in serv_list:
                                act_lat = act_latency_comp_from_v(css, v_UB, rho_UB, tau_UB, lambda_aux_UB, semantic, lambda_semantic,
                                                        theta, max_latency, functions_compl, J_MAX)
                                if act_lat > services_L[ss]:
                                    check_increment = False
                                    break
                            if not check_increment:
                                rho_UB[f, c, j, 'cpu'] = rho_UB[f, c, j, 'cpu'] + cpu_delta
                                break

    return z_UB, v_UB, rho_UB, tau_UB, n_aux_UB, lambda_aux_UB, lambda_aux_UB

def remove_service(z_UB, v_UB, n_aux_UB, rho_UB, q_UB, tau_UB, lambda_aux_UB, r, budget, consumption, functions, functions_compl, services, services_P, services_conf, semantic_cs, J_MAX, max_latency, cs_list):

    # I have to save (consumption - budget[r]) resources, with r, budget[r], consumption given by input.
    r_to_save = consumption - budget[r] # always > 0

    services_P_list = list(set([services_P[s] for s in services]))
    services_P_list.sort()

    sv_to_remove = {}
    s_to_remove_priority = {}

    notdone_flag = True # set it False when all the services to be removed are found

    # while notdone_flag:
    for s in services: # fix the service
        for cs in services_conf[s]: # fix the service configuration
            if z_UB[s, list(cs.keys())[0]] == 1:
                cs_list_sharing = []
                service_cost = 0
                sv_to_remove[s,list(cs.keys())[0]] = []
                for f in list(cs.values())[0]: # fix the function
                    for c in functions_compl[f]: # fix the complexity function
                        for j in range(1, J_MAX + 1): # fix the number of instance
                            if v_UB[list(cs.keys())[0],f,c,j] == 1: # if the xApp f[c,j] is involved
                                # service_cost = service_cost + rho_UB[f,c,j,r] # focus on res of type r
                                sv_to_remove[s,list(cs.keys())[0]].append((f,c,j))
                                service_cost += rho_UB[f,c,j,r]
                                for ss in services:  # fix the service
                                    for css in services_conf[ss]:
                                        # if css != cs:
                                        if v_UB[list(css.keys())[0],f,c,j] == 1:
                                            # cs_list_sharing.append(list(css.keys())[0])
                                            cs_list_sharing.append(ss)
                                            break
                service_cost = service_cost / sum([services_P[ss] for ss in list(set(cs_list_sharing))])
                s_to_remove_priority[s, list(cs.keys())[0]] = services_P[s], service_cost

    s_to_remove_priority = dict(sorted(s_to_remove_priority.items(), key = lambda x: (x[1][0], -x[1][1])))
    s_to_remove_priority = dict(sorted(s_to_remove_priority.items(), key = lambda x: (-x[1][1])))

    # now the services in sv_to_remove has to be removed.
    serv_list = {}
    for s,cs in s_to_remove_priority: # fix the service to remove
        z_UB[s, cs] = 0
        q_UB[cs] = 0
        tau_UB[cs] = max_latency
        for f,c,j in sv_to_remove[s,cs]: # fix the service configuration to remove and interate over the xApps
            serv_list[f, c, j] = [] # for each xApp we keep track of the services that share such an xApp
            for ss,css in sv_to_remove:
                    if v_UB[css, f, c, j] == 1:
                        serv_list[f, c, j].append(css)


            if v_UB[cs, f, c, j] == 1: # additional check
                v_UB[cs, f, c , j] = 0 # we disconnect f[c,j] from cs. In order to understand if we can deactivate it or not,
                                       # we must look at serv_list[f, c, j].
                if len(serv_list[f, c, j]) == 1:
                    n_aux_UB[f, c, j] = 0
                    r_to_save = r_to_save - rho_UB[f, c, j, r]
                    for rr in budget:
                        rho_UB[f, c, j, rr] = 0
                lambda_aux_UB[f, semantic_cs[f][cs], c, j] = 0
                for css in serv_list[f, c, j]:
                    if css != cs:
                        # lambda_aux_prime[f1, semantic_cs[f1][css], c1, j1] = 1
                        lambda_aux_UB[f, semantic_cs[f][css], c, j] = 1
            else:
                print("\n\n* * * \nWarning in 'remove_service': we are iterating over a not implemented xApp")
        if r_to_save <= 0:
            break

    return z_UB, v_UB, n_aux_UB, rho_UB, q_UB, tau_UB, lambda_aux_UB

# ###################################################################################################################

def compute_UB_lagrangianmult_noQ_noL(services, services_notprime, services_P, services_L, services_Q, services_conf, services_conf_notprime, functions, functions_compl, J_MAX, z_UB, v_UB, rho_UB, q_UB, tau_UB, beta, budget, BigM, max_latency):
    UB = 0
    for s in services:
        for cs in services_conf[s]:
            UB += z_UB[s, list(cs.keys())[0]] * (-1 * services_P[s])
            for f in list(cs.values())[0]:
                UB += (+1) * z_UB[s, list(cs.keys())[0]] * beta[s, list(cs.keys())[0], f]

    for f in functions:
        for c in functions_compl[f]:
            for j in range(1, J_MAX + 1):
                for r in budget:
                    UB += (1/3) * rho_UB[f, c, j, r] / budget[r]
    for s in services:
        for cs in services_conf[s]:
            for f in list(cs.values())[0]:
                for c in functions_compl[f]:
                    for j in range(1, J_MAX + 1):
                        UB += (-1) * beta[s, list(cs.keys())[0], f] * v_UB[list(cs.keys())[0], f, c, j]
    return UB

# ###################################################################################################################

def compute_UB_lagrangianmult_noQ_noL(services, services_notprime, services_P, services_L, services_Q, services_conf, services_conf_notprime, functions, functions_compl, J_MAX, z_UB, v_UB, rho_UB, q_UB, tau_UB, beta, budget, BigM, max_latency):
    UB = 0
    for s in services:
        for cs in services_conf[s]:
            UB += z_UB[s, list(cs.keys())[0]] * (-1 * services_P[s])
            for f in list(cs.values())[0]:
                UB += (+1) * z_UB[s, list(cs.keys())[0]] * beta[s, list(cs.keys())[0], f]
            UB += (+1) * z_UB[s, list(cs.keys())[0]] * gamma[s, list(cs.keys())[0]] * services_Q[s]

    for f in functions:
        for c in functions_compl[f]:
            for j in range(1, J_MAX + 1):
                for r in budget:
                    UB += (1/3) * rho_UB[f, c, j, r] / budget[r]
    for s in services:
        for cs in services_conf[s]:
            for f in list(cs.values())[0]:
                for c in functions_compl[f]:
                    for j in range(1, J_MAX + 1):
                        UB += (-1) * beta[s, list(cs.keys())[0], f] * v_UB[list(cs.keys())[0], f, c, j]
            UB += (-1) * gamma[s, list(cs.keys())[0]] * q_UB[list(cs.keys())[0]]
    return UB

# ###################################################################################################################

def xAppSelection(services, services_conf, cs_to_s, s, cs, functions_compl, missing_f, J_MAX, budget, z, v, rho, rho_UB, semantic, lambda_semantic, semantic_cs,lambda_aux_UB,lambda_aux_prime, theta, v_UB,n_aux_UB, n_aux_prime, xApp_mem_req,services_L, current_res_utiliz):
    """
    with respect to xAppSelection_new() we have modified the following:
    - less sharing probability.
    Consumiamo troppa CPU e pochissima memoria con la nostra heuristica. Per questo motivo, abbiamo deciso di modificare
    la policy con cui condividiamo una xApp.
    """
    for f in missing_f:
        f_done = False
        xapp_sharing = [] # list of xApps that share provide function f for any of the CSs in the semantic of cs
        # for sem in semantic[f]: # I iterate over all the semantics, but then I consider only the semantic in which cs is
        #     if list(cs.keys())[0] in semantic[f][sem]:
        #         for cs_two in semantic[f][sem]:  # semantic[f][sem] is the set of all the CSs that have the same semantic of cs, cs included
        for cs_two in semantic[f][semantic_cs[f][list(cs.keys())[0]]]: # semantic[f][sem] is the set of all the CSs that have the same semantic of cs, cs included
            try:
                if z[cs_to_s[cs_two],cs_two] == 1: # check if the service configuration has been actually selected
                    xApp_found = False
                    # given f, and a selected cs, check if there is an xApp implementing f and providing it to cs
                    for c in functions_compl[f]:
                        for j in range(1, J_MAX + 1):
                            if v[cs_two, f, c, j] == 1:
                                xapp_sharing.append((c,j,cs_two))
                                xApp_found = True
                                break
                        if xApp_found:
                            break # out of "for c in functions_compl[f]:" loop
            except: pass
        if len(xapp_sharing)>0:
            metric = [abs(services_L[cs_to_s[cs_two]]) for cc,jj,cs_two in xapp_sharing]
            min_load = min(metric)
            min_indices = [i for i, x in enumerate(metric) if x == min_load]
            if len(min_indices) > 1:
                min_index = random.choice(min_indices)
            else:
                min_index = min_indices[0]
            cc,jj,_ = xapp_sharing[min_index]
            v_UB[list(cs.keys())[0],f,cc,jj] = 1
            v[list(cs.keys())[0],f,cc,jj] = 1
            # v[list(cs.keys())[0],f,cc,jj] = 1
            n_aux_UB[f,cc,jj] = 1 # not needed
            lambda_aux_UB[f,semantic_cs[f][list(cs.keys())[0]],cc,jj] = 1
            # lambda_aux_prime[f,lambda_semantic[f][sem],cc,jj] = 1
            for r in budget:
                if r != 'cpu':
                    rho_UB[f, cc, jj, r] = xApp_mem_req[f,cc,r]
                # else:
                #     rho_UB[f, cc, jj, r] = 0 # keep rho_UB[f, cc, jj, 'cpu'] = 0
            f_done = True

        if not f_done:
            # CASE 2: some xApps providing f with a different semantic as foreseen by cs have been already instantiated
            # for other service configurations cs_two. For each of them, compute the sharing_score and share the one
            # with highest sharing_score.
            # xapp_sharing = [] # already defined
            for s_two in services:
                for cs_two in services_conf[s_two]:
                    if z[cs_to_s[list(cs_two.keys())[0]],list(cs_two.keys())[0]] == 1:
                        for c in functions_compl[f]:
                            for j in range(1, J_MAX + 1):
                                if v[list(cs_two.keys())[0], f, c, j] == 1:
                                    rho_UB[f, c, j, 'cpu'] = max(rho_UB[f, c, j, 'cpu'], 1/theta[f, c]*(len(list(cs.values())[0])/services_L[s] + lambda_semantic[f][semantic_cs[f][list(cs.keys())[0]]]))
                                    sharing_score, new_cpu = sharing_score_comp(f, c, j, s, cs, services, services_conf, z, v, lambda_aux_UB, semantic, lambda_semantic, semantic_cs, theta, budget, xApp_mem_req, services_L, current_res_utiliz)
                                    if sharing_score != None:
                                        xapp_sharing.append((c,j,sharing_score,new_cpu))
            if len(xapp_sharing) > 0:
                # cc,jj,sh_score,new_cpu = random.choices(xapp_sharing, weights=[t[-2] for t in xapp_sharing])[0]
                cc,jj,sh_score,new_cpu = max(xapp_sharing, key=lambda x: -x[2])
                v_UB[list(cs.keys())[0], f, cc, jj] = 1
                v[list(cs.keys())[0], f, cc, jj] = 1
                n_aux_UB[f, cc, jj] = 1 # not needed
                # n_aux_prime[f, cc, jj] = 1
                lambda_aux_UB[f,semantic_cs[f][list(cs.keys())[0]],cc,jj] = 1
                # lambda_aux_prime[f,semantic_cs[f][list(cs.keys())[0]],cc,jj] = 1
                for r in budget:
                    if r == 'cpu':
                        current_res_utiliz[r] += new_cpu - rho_UB[f, cc, jj, r]
                        rho_UB[f, cc, jj, r] = new_cpu
                    else: # in theory, useless
                        rho_UB[f, cc, jj, r] = xApp_mem_req[f,cc,r] # in theory, useless
                f_done = True

        if not f_done:
            # CASE 3: instantiate a new xApp providing f with the lowest complexity factor
            for c in functions_compl[f]:
                for j in range(1, J_MAX + 1):
                    if n_aux_UB[f,c,j]==0:  # if the xApp [f,c,j] is not associated with any service
                        v_UB[list(cs.keys())[0], f, c, j] = 1
                        v[list(cs.keys())[0], f, c, j] = 1
                        n_aux_UB[f, c, j] = 1
                        lambda_aux_UB[f,semantic_cs[f][list(cs.keys())[0]],c,j] = 1
                        for r in budget:
                            if r == 'cpu':
                                rho_UB[f, c, j, r] = 1/theta[f, c]*(len(list(cs.values())[0])/services_L[s] + lambda_semantic[f][semantic_cs[f][list(cs.keys())[0]]])
                                current_res_utiliz[r] += rho_UB[f, c, j, r]
                            else:
                                rho_UB[f, c, j, r] = xApp_mem_req[f,c,r]
                                current_res_utiliz[r] += rho_UB[f, c, j, r]
                        f_done = True
                        break
                if f_done:
                    break

        if not f_done:
            print("xAppSelection Warning: impossible to instantiate an xApp")

    return z,v,rho_UB,lambda_aux_UB,lambda_aux_prime,v_UB,n_aux_UB,n_aux_prime


def ServiceQualityAdjustment(services,s,services_L,services_P,services_Q,services_conf,cs,cs_to_s,services_conf_graph_output,services_conf_graph_former,functions,functions_compl,quality_mapping_x, quality_mapping_q, f_multiplier, f_multiplier_c,xApp_mem_req,J_MAX,budget,z_UB,v_UB,rho_UB,q_UB,n_aux_UB,lambda_aux_UB,semantic,lambda_semantic,semantic_cs,theta, current_res_utiliz):
    '''
    when selecting a new complexity, instead of selecting the deriving xApp just for the under-object service,
    we should select it for all the services that use the same xApp, or the deriving xApp can be taken already existing.
    To sum up:
    1) For a selected (s,cs) pair for which the quality constraint is not respected, identify the xApp whose
       improvement leads to the greatest quality improvement. Update the corresponding complexity looking for xApps
       that can be shared. --> NON SOLO PRENDO IN CONSIDERAZIONE INCREMENTS DI UN SOLO LIVELLO DI COMPLESSITA', MA
       ANCHE DI PIù LIVELLI. PERCHè? PERCHE' SE UNA FUNZIONE E' STATA SCELTA CON COMPLESSITA' 1 MA ESISTE GIà UNA XAPP
       CON COMPLESSITà 3, ALLORA SI PUò CONDIVIDERE LA XAPP CON COMPLESSITà 3 ED ELIMINARE QUELLA CON COMPLESSITA 1
    2) For a selected (s,cs) pair for which the quality constraint is not respected, identify the xApp whose
       improvement leads to the greatest quality improvement. IF THE NEW COMPLEXITY IS NOT ALREADY AVAILABLE, update
       the corresponding complexity, and update the v[cs,f,c,j] for every cs using/sharing the xApp
    '''

    current_config = []
    # it is important to scroll the f in list(cs.values())[0] starting from the output function, because of --> look at the definition of act_quality_comp
    for f in [services_conf_graph_output[list(cs.keys())[0]]] + services_conf_graph_former[list(cs.keys())[0]]:
        for c in functions_compl[f]:
            for j in range(1, J_MAX+1):
                if v_UB[list(cs.keys())[0], f, c, j] == 1:
                    current_config.append([f,c,j])

    original_config = copy.deepcopy(current_config)

    actual_quality = act_quality_comp(current_config, cs, quality_mapping_x, quality_mapping_q, f_multiplier, f_multiplier_c)

    try:
        if actual_quality > services_Q[s]:
            q_UB[list(cs.keys())[0]] = actual_quality
            return z_UB, v_UB, rho_UB, q_UB, n_aux_UB, lambda_aux_UB
    except:
        return -1
    # else:
    #     print("Need improvement!")

    actual_quality = actual_quality - services_Q[s]

    new_cost_dict = {}

    current_improvement = 0

    while actual_quality < 0:
        improvement = (-1000, -1000)
        for f_num,cur_conf in enumerate(current_config):
            for c in functions_compl[cur_conf[0]][1:]:
                if c > cur_conf[1]:
                    new_config = copy.deepcopy(current_config)
                    new_config[f_num][1] = c
                    new_quality = act_quality_comp(new_config, cs, quality_mapping_x, quality_mapping_q, f_multiplier, f_multiplier_c) - services_Q[s]
                    # new cost is the additional memory and disk required by the new configuration
                    new_cost = {}
                    for r in budget:
                        new_cost[r] = 0
                    disk_new = xApp_mem_req[cur_conf[0], c, 'disk'] - xApp_mem_req[cur_conf[0], cur_conf[1], 'disk']
                    mem_new = xApp_mem_req[cur_conf[0], c, 'mem'] - xApp_mem_req[cur_conf[0], cur_conf[1], 'mem']
                    cpu_new = (1 / theta[new_config[f_num][0], new_config[f_num][1]]) * (
                                len(list(cs.values( ))[0]) / services_L[s] + lambda_semantic[f][semantic_cs[f][
                            list(cs.keys( ))[0]]])  # the CPU needed by a new xApp, dedicated for the new service
                    for j in range(1,J_MAX+1):
                        if n_aux_UB[new_config[f_num][0], new_config[f_num][1], j] == 1:
                            # the xApp is already instantiated
                            new_cost['mem'] = 0
                            new_cost['disk'] = 0
                            if lambda_aux_UB[new_config[f_num][0],semantic_cs[new_config[f_num][0]][list(cs.keys())[0]],new_config[f_num][1],j] == 1:
                                new_cost['cpu'] = 0
                            else:
                                current_load = sum(lambda_aux_UB[new_config[f_num][0], sssem, new_config[f_num][1], j] * lambda_semantic[new_config[f_num][0]][sssem] for sssem in semantic[new_config[f_num][0]])
                                current_l = 1 / (rho_UB[new_config[f_num][0], new_config[f_num][1], j, 'cpu'] * theta[new_config[f_num][0], new_config[f_num][1]] - current_load)
                                new_CPU_saring = (1 / theta[new_config[f_num][0], new_config[f_num][1]]) * \
                                                 (current_load + lambda_semantic[f][semantic_cs[f][list(cs.keys())[0]]] + (1/current_l))
                                new_cost['cpu'] = new_CPU_saring - rho_UB[new_config[f_num][0], new_config[f_num][1], j, 'cpu']
                        else:
                            new_cost['mem'] = disk_new
                            new_cost['disk'] = mem_new
                            new_cost['cpu'] = cpu_new

                        score = xApp_score([new_cost[r] for r in budget],
                                           [current_res_utiliz[r] for r in budget])
                        if new_quality > 0:
                            if score != 0:
                                current_improvement = new_quality / score
                            else:
                                current_improvement = 900
                                score = 999999
                        else:
                            current_improvement = new_quality * score

                        try:
                            if current_improvement > improvement[-1]: # I take the max current_improvement
                                actual_quality = new_quality
                                improvement = (new_config, current_improvement)
                                new_config[f_num][2] = j
                                for r in budget:
                                    new_cost_dict[new_config[f_num][0], new_config[f_num][1], j, r] = new_cost[r]
                        except:
                            print("Attention here")

        if improvement == (-1000, -1000):
            z_UB, v_UB, n_aux_UB, rho_UB, lambda_aux_UB = remove_service_s(s, cs, z_UB, v_UB, n_aux_UB, rho_UB,
                                                                           lambda_aux_UB, budget, functions,
                                                                           functions_compl, services, services_P,
                                                                           services_conf, semantic_cs, J_MAX)
            return z_UB, v_UB, rho_UB, q_UB, n_aux_UB, lambda_aux_UB

        current_config = copy.deepcopy(improvement[0])

    # if (s, cs, original_config, current_config, new_config) == ('s_10', {'cs_10_0': ['f_6']}, [['f_6', 0, 1]], [['f_6', 1, 1]], [['f_6', 0, 1]]):
    #     print("debugging")

    serv_list = {}
    # at this point, cs configuration has been changed
    #print("ORIGINAL VS CURRENT CONFIGURATION FOR SERVICE",s,"WITH CONFIGURATION",list(cs.keys())[0],":1\n\t\t",original_config,current_config)
    for (f1, c1, j1), (f2, c2, j2) in zip(original_config, current_config):
        if (f1, c1, j1) == (f2, c2, j2):
            pass
        else:
            serv_list[f1, c1, j1] = []
            for ss in services:
                for css in services_conf[ss]:
                    if v_UB[list(css.keys())[0], f1, c1, j1] == 1:
                        serv_list[f1, c1, j1].append(list(css.keys())[0])
            if n_aux_UB[f2,c2,j2] == 1:
                for css in serv_list[f1, c1, j1]:
                    if semantic_cs[f1][css] == semantic_cs[f1][list(cs.keys())[0]]:
                        v_UB[css, f1, c1, j1] = 0
                        v_UB[css, f2, c2, j2] = 1
                        lambda_aux_UB[f2, semantic_cs[f2][css], c2, j2] = 1
                        # for sss in services:
                        #     for csss in services_conf[sss]:
                        #         if csss != css:
                        #             if v_UB[list(csss.keys())[0], f1, c1, j1] == 1:
                        #                 lambda_aux_UB[f1, semantic_cs[f1][csss], c1, j1] = 1
                lambda_aux_UB[f1, semantic_cs[f1][list(cs.keys())[0]], c1, j1] = 0
                for r in budget:
                    if r != 'cpu':
                        rho_UB[f2, c2, j2, r] = xApp_mem_req[f2, c2, r] #useless
                    else:
                        try:
                            rho_UB[f2, c2, j2, r] += new_cost_dict[f2,c2,j2,r]
                            current_res_utiliz[r] += new_cost_dict[f2,c2,j2,r]
                        except:
                            raise Exception("ServiceQualityAdjustment: error in the new_cost_dict for service", s, "with configuration", list(cs.keys())[0])
                            pass


                if len(serv_list[f1, c1, j1]) == \
                        len([css for css in serv_list[f1, c1, j1] if semantic_cs[f1][css] == semantic_cs[f1][list(cs.keys())[0]]]):
                    n_aux_UB[f1,c1,j1] = 0
                    for r in budget:
                        # current_res_utiliz[r] -= rho_UB[f1, c1, j1, r]
                        rho_UB[f1,c1,j1,r] = 0
                        if any([current_res_utiliz[r] < 0 for r in budget]):
                            raise Exception("ServiceQualityAdjustment: negative resource utilization for service", s,
                                            "with configuration", list(cs.keys( ))[0]
                                            )
            else:
                n_aux_UB[f2, c2, j2] = 1
                n_aux_UB[f1, c1, j1] = 0
                for css in serv_list[f1, c1, j1]:
                    v_UB[css, f1, c1, j1] = 0
                    v_UB[css, f2, c2, j2] = 1
                    lambda_aux_UB[f1, semantic_cs[f1][css], c1, j1] = 0
                    lambda_aux_UB[f2, semantic_cs[f2][css], c2, j2] = 1
                for r in budget:
                    if r == 'cpu':
                        rho_UB[f2, c2, j2, r] = new_cost_dict[f2,c2,j2,r]
                    else:
                        rho_UB[f2, c2, j2, r] = xApp_mem_req[f2, c2, r]
                    current_res_utiliz[r] += new_cost_dict[f2, c2, j2, r]
                    # current_res_utiliz[r] -= rho_UB[f1, c1, j1, r]
                    # check if any element of current_res_utiliz is negative
                    if any([current_res_utiliz[r] < 0 for r in budget]):
                        raise Exception("ServiceQualityAdjustment: negative resource utilization for service", s, "with configuration", list(cs.keys())[0])
                    rho_UB[f1, c1, j1, r] = 0


    q_UB[list(cs.keys())[0]] = actual_quality + services_Q[s]

    return z_UB, v_UB, rho_UB, q_UB, n_aux_UB, lambda_aux_UB

def xAppQualityFinetuning(services, services_P, services_Q, services_conf, cs, cs_to_s, services_conf_graph_output, services_conf_graph_former, functions, functions_compl, quality_mapping_x, quality_mapping_q, f_multiplier, f_multiplier_c, xApp_mem_req, J_MAX, budget, z_UB, v_UB, rho_UB, q_UB, n_aux_UB, lambda_aux_UB, semantic, lambda_semantic, semantic_cs, theta):

    for cs,f,c,j in v_UB:
        if v_UB[cs, f, c, j] == 1:
            if n_aux_UB[f, c, j] == 0:
                print("Heuristic - step xAppQualityFinetuning (1) : Errore 1")
            for r in ["disk", "mem"]:
                if rho_UB[f, c, j, r] != xApp_mem_req[f, c, r]:
                    print("Heuristic - step xAppQualityFinetuning (1) : Errore 2\tExpected rho:", [xApp_mem_req[f, c, rr] for rr in ['mem','disk']], "\tActual rho:", [rho_UB[f, c, j, rr] for rr in ['mem','disk']])


    serv_list = {}
    serv_conf = {}
    for s in services:
        for cs in services_conf[s]:
            if z_UB[s, list(cs.keys())[0]] == 1:
                serv_conf[s] = []
    for f,c,j in n_aux_UB:
        if n_aux_UB[f,c,j] == 1:
            serv_list[f,c,j] = []
            for s in services:
                for cs in services_conf[s]:
                    if z_UB[s,list(cs.keys())[0]] == 1:
                        if v_UB[list(cs.keys())[0],f,c,j] == 1:
                            serv_list[f,c,j].append((s,list(cs.keys())[0]))
                            serv_conf[s].append([f,c,j])

    combos = [(f,c,j,theta[f,c]) for f,c,j in serv_list.keys()]
    combos = sorted(combos, key=lambda x: x[3], reverse=False)

    for f,c,j,_ in combos:
        # (f,c,j) are the instantiated xApps
        for c2 in functions_compl[f]:
            if c2<c:
                possible_decrement_flag = True
                new_serv_conf = copy.deepcopy(serv_conf)
                for s,cs in serv_list[f,c,j]:
                    new_serv_conf[s] = modify_serv_conf_c(new_serv_conf[s], f, c2)
                    try:
                        if act_quality_comp(new_serv_conf[s], get_dict_with_key(services_conf[s], cs), quality_mapping_x, quality_mapping_q, f_multiplier, f_multiplier_c) < services_Q[s]:
                            possible_decrement_flag = False
                            break
                    except:
                        print(s, cs, services_Q[s],new_serv_conf[s], get_dict_with_key(services_conf[s], cs))
                if possible_decrement_flag:
                    j2 = new_instance(v_UB, f, c2, services, services_conf, functions_compl, J_MAX)
                    for s,cs in serv_list[f,c,j]:
                        new_serv_conf[s] = modify_serv_conf_j(new_serv_conf[s], f, j2)
                        v_UB[cs,f,c,j] = 0
                        v_UB[cs,f,c2,j2] = 1
                    n_aux_UB[f,c,j] = 0
                    n_aux_UB[f,c2,j2] = 1
                    for r in budget:
                        if r != 'cpu':
                            rho_UB[f, c2, j2, r] = xApp_mem_req[f,c2,r]
                        else:
                            rho_UB[f, c2, j2, r] = rho_UB[f,c,j,r]
                        rho_UB[f,c,j,r] = 0
                    lambda_aux_UB[f,semantic_cs[f][cs],c2,j2] = 1
                    lambda_aux_UB[f,semantic_cs[f][cs],c,j] = 0
                    serv_conf = copy.deepcopy(new_serv_conf)
                    break

    for cs,f,c,j in v_UB:
        if v_UB[cs, f, c, j] == 1:
            if n_aux_UB[f, c, j] == 0:
                print("Heuristic - step xAppQualityFinetuning (2) : Errore 1")
            for r in ["disk", "mem"]:
                if rho_UB[f, c, j, r] != xApp_mem_req[f, c, r]:
                    print("Heuristic - step xAppQualityFinetuning (2) : Errore 2")
                    break

    return z_UB, v_UB, rho_UB, q_UB, n_aux_UB, lambda_aux_UB

def EnsuringFeasibility(services, services_notprime, services_P, services_Q, services_L,
                        services_conf,services_conf_notprime, cs_list, cs_to_s,
                        services_conf_graph_output, services_conf_graph_former,functions,
                        functions_compl,
                        quality_mapping_x, quality_mapping_q, f_multiplier, f_multiplier_c,
                        J_MAX, budget,
                        z, v, rho, q, tau, n_aux,
                        semantic, lambda_semantic,semantic_cs, theta,
                        beta, gamma, delta,
                        xApp_mem_req,
                        max_latency = 1, BigM = 1,normalization_factor = None):

    if normalization_factor == None:
        normalization_factor = 1

    z = {k: 1 if v > 0.5 else 0 for k, v in z.items()}
    v = {k: 1 if v > 0.5 else 0 for k, v in v.items()}
    n_aux = {k: 1 if v > 0.5 else 0 for k, v in n_aux.items()}

    removed_services_flag = False

    z_UB, v_UB, n_aux_UB, rho_UB, q_UB, tau_UB, n_aux_prime, lambda_aux_UB, lambda_aux_prime = \
        var_init(services, services_conf, functions, functions_compl, theta, budget, J_MAX, semantic, lambda_semantic, semantic_cs, cs_to_s, services_L, max_latency, v, z, rho, q, tau, n_aux)

    lambda_aux_UB = check_lambda_aux(lambda_aux_UB, functions, functions_compl, semantic, J_MAX, cs_to_s, v_UB, z_UB)

    current_res_utiliz = {}
    for r in budget:
        current_res_utiliz[r] = 0
    for ff,cc,jj,rr in rho_UB:
        if n_aux_prime[ff,cc,jj] == 1:
            if rr == 'cpu':
                pass
            else:
                rho_UB[ff,cc,jj,rr] = xApp_mem_req[ff,cc,rr]
    for ff,cc,jj,rr in rho_UB:
        current_res_utiliz[rr] += rho_UB[ff,cc,jj,rr]

    for s in services:
        for cs in services_conf[s]:
            service_done = True
            if (z[s, list(cs.keys())[0]] == 1.):
                z_UB[s, list(cs.keys())[0]] = 1
                # Check if f eorvery function f involved by cs, an xApp f[c,j] has been associated to cs
                missing_f = []
                for f in list(cs.values())[0]:
                    xApp_found = False
                    for c in functions_compl[f]:
                        for j in range(1, J_MAX + 1):
                        # for j in [1]:
                            if v[list(cs.keys())[0], f, c, j] == 1:
                                xApp_found = True
                                v_UB[list(cs.keys())[0], f, c, j] = 1
                                n_aux_UB[f, c, j] = 1
                                lambda_aux_UB[f, semantic_cs[f][list(cs.keys())[0]], c, j] = 1
                                for r in budget:
                                    if r != 'cpu':
                                        rho_UB[f, c, j, r] = xApp_mem_req[f,c,r]
                                        # current_res_utiliz[r] += xApp_mem_req[f,c,r]
                                    else:
                                        rho_UB[f, c, j, r] = rho[f, c, j, r]
                                        # current_res_utiliz[r] += rho[f, c, j, r]
                                        # rho_UB[f, c, j, r] = rho[f, c, j, r] / 10
                                        # rho_UB[f, c, j, r] = 0
                                break
                        if xApp_found:
                            break
                    if not xApp_found:
                        missing_f.append(f)

                if len(missing_f):
                    z, v, rho_UB, lambda_aux_UB, lambda_aux_prime, v_UB, n_aux_UB, n_aux_prime = xAppSelection(services, services_conf, cs_to_s, s, cs, functions_compl, missing_f, J_MAX, budget, z, v, rho, rho_UB, semantic, lambda_semantic, semantic_cs, lambda_aux_UB, lambda_aux_prime, theta, v_UB, n_aux_UB, n_aux_prime, xApp_mem_req,services_L, current_res_utiliz)
    lambda_aux_UB = check_lambda_aux(lambda_aux_UB, functions, functions_compl, semantic, J_MAX, cs_to_s, v_UB, z_UB)

    # update the current_res_utiliz
    current_res_utiliz = {}
    for r in budget:
        current_res_utiliz[r] = 0
    for ff,cc,jj,rr in rho_UB:
        current_res_utiliz[r] += rho_UB[ff,cc,jj,rr]

    for s in services:
        for cs in services_conf[s]:
            # At this point, the selected service configuration has been properly configured : enough xApps have
            # been instantiated to provide the requested service.
            # We start looking at the Quality
            if z_UB[s,list(cs.keys())[0]] == 1:
                z_UB, v_UB, rho_UB, q_UB, n_aux_UB, lambda_aux_UB = ServiceQualityAdjustment(services, s, services_L, services_P, services_Q, services_conf, cs, cs_to_s, services_conf_graph_output, services_conf_graph_former, functions, functions_compl, quality_mapping_x, quality_mapping_q, f_multiplier, f_multiplier_c, xApp_mem_req, J_MAX, budget, z_UB, v_UB, rho_UB, q_UB, n_aux_UB, lambda_aux_UB, semantic, lambda_semantic, semantic_cs, theta, current_res_utiliz)

    lambda_aux_UB = check_lambda_aux(lambda_aux_UB, functions, functions_compl, semantic, J_MAX, cs_to_s, v_UB, z_UB)

    for cs, f, c, j in v_UB:
        if v_UB[cs, f, c, j] == 1:
            if n_aux_UB[f, c, j] == 0:
                print("Heuristic - step 3 : Errore 1")
            for r in ["disk", "mem"]:
                if rho_UB[f, c, j, r] != xApp_mem_req[f, c, r]:
                    print("Heuristic - step 3 : Errore 2")

    for s in services:
        for cs in services_conf[s]:
            # At this point, the selected service configuration has been properly configured : enough xApps have
            # been instantiated to provide the requested service.
            # We start looking at the Quality
            if z_UB[s,list(cs.keys())[0]] == 1:
                rho_UB_old = rho_UB.copy()
                tau_UB_old = tau_UB.copy()
                z_UB, v_UB, rho_UB, tau_UB, n_aux_UB, lambda_aux_UB, lambda_aux_prime = ServiceLatencyAdjustment(services, s, services_P, services_L, services_conf, cs, cs_to_s, services_conf_graph_output, services_conf_graph_former, functions, functions_compl, quality_mapping_x, quality_mapping_q, f_multiplier, f_multiplier_c, J_MAX, budget, z_UB, v_UB, rho_UB, tau_UB, n_aux_UB, lambda_aux_UB, lambda_aux_prime, semantic, lambda_semantic, semantic_cs, theta, max_latency)


    z_UB, v_UB, rho_UB, tau_UB, n_aux_UB, lambda_aux_UB, lambda_aux_UB = xAppResourceFinetuning(services, services_L, services_conf, cs_to_s, services_conf_graph_output, services_conf_graph_former, functions, functions_compl, J_MAX, budget, z_UB, v_UB, rho_UB, tau_UB, n_aux_UB, lambda_aux_UB, semantic, lambda_semantic, semantic_cs, theta, max_latency)

    lambda_aux_UB = check_lambda_aux(lambda_aux_UB, functions, functions_compl, semantic, J_MAX, cs_to_s, v_UB, z_UB)

    # reset of n_aux_UB since in 'remove_service()' step I am facing problems
    for cs, f, c, j in v_UB:
        n_aux_UB[f, c, j] = 0
    for cs, f, c, j in v_UB:
        if v_UB[cs, f, c, j] == 1:
            n_aux_UB[f, c, j] = 1
            for r in budget:
                if r != 'cpu':
                    rho_UB[f, c, j, r] = xApp_mem_req[f, c, r]

    for r in budget:
        consumption = 0
        for f,c,j,rrrr in rho_UB:
            if rrrr == r:
                consumption += rho_UB[f, c, j, r] * n_aux_UB[f, c, j]
        if consumption > budget[r]:
            removed_services_flag = True
            z_UB, v_UB, n_aux_UB, rho_UB, q_UB, tau_UB, lambda_aux_UB = remove_service(z_UB, v_UB, n_aux_UB, rho_UB, q_UB, tau_UB, lambda_aux_UB, r, budget, consumption, functions, functions_compl, services, services_P, services_conf, semantic_cs, J_MAX, max_latency, cs_list)
        nservices_counter = 0
        for s in services:
            for cs in services_conf[s]:
                if z_UB[s, list(cs.keys())[0]] == 1:
                    nservices_counter += 1

    for f in functions:
        for c in functions_compl[f]:
            for j in range(1, J_MAX + 1):
                n_aux_UB[f, c, j] = 0
                for s in services:
                    for cs in services_conf[s]:
                        try:
                            if v_UB[list(cs.keys())[0], f, c, j] == 1:
                                n_aux_UB[f, c, j] = 1
                                for r in budget:
                                    if r != 'cpu':
                                        rho_UB[f, c, j, r] = xApp_mem_req[f, c, r]
                        except:
                            pass

    UB = compute_UB_lagrangianmult(services, services_notprime, services_P, services_L, services_Q, services_conf,
                                   services_conf_notprime, functions, functions_compl, J_MAX, z_UB, v_UB, rho_UB, q_UB,
                                   tau_UB, beta, gamma, delta, budget, BigM, max_latency)
    UB_notlagrangian = compute_UB(services, services_P, services_conf, functions, functions_compl, J_MAX, z_UB, rho_UB,
                                  budget)
    UB_obj, UB_norm_obj = compute_objectivefunction(services,
                                                    services_P,
                                                    services_conf,
                                                    functions,
                                                    functions_compl,
                                                    z_UB,
                                                    n_aux_UB,
                                                    rho_UB,
                                                    budget,
                                                    J_MAX,
                                                    normalization_factor = normalization_factor)

    for f in functions:
        for c in functions_compl[f]:
            for j in range(1, J_MAX + 1):
                counter = 0
                for cs in cs_list:
                    counter += v_UB[cs,f,c,j]
                if counter < 0.5:
                    for r in budget:
                        rho_UB[f,c,j,r] = 0

    return z_UB, v_UB, n_aux_UB, rho_UB, UB, UB_notlagrangian, UB_obj, UB_norm_obj, q_UB, tau_UB
