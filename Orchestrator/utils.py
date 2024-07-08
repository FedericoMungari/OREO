
import math
import networkx as nx
import numpy as np
from itertools import product


# #########################################################################################################

def network_construction(g_nodes):
    G = nx.Graph()
    G = nx.random_tree(len(g_nodes))
    G_dir = nx.DiGraph()
    old_layer = [0]
    curr_layer = [0]
    next_layer = []
    for n in G.nodes():
        G_dir.add_node(n)
    while (1):
        for u in G.nodes():
            for v in G.nodes():
                if ((u, v) in G.edges() or (v, u) in G.edges()) and (u in curr_layer):
                    if not v in old_layer:
                        G_dir.add_edge(v, u)
                        next_layer.append(v)
                        old_layer.append(v)
                if ((u, v) in G.edges() or (v, u) in G.edges()) and (v in curr_layer):
                    if not u in old_layer:
                        G_dir.add_edge(u, v)
                        next_layer.append(u)
                        old_layer.append(u)

        if next_layer == []:
            break
        curr_layer[:] = next_layer
        next_layer = []

    mapping = {}
    for gn in G_dir.nodes():
        mapping[gn] = g_nodes[gn]
    nx.relabel_nodes(G_dir, mapping, copy=False)

    return (G_dir)

# #########################################################################################################

def compute_LB(services, services_P, services_conf, functions, functions_compl, J_MAX, z, rho, budget):
    LB = 0

    if hasattr(z[list(z.keys())[0]], 'X'):
        for s in services:
            for cs in services_conf[s]:
                LB += z[s, list(cs.keys())[0]].X * (-1 * services_P[s])
    else:
        for s in services:
            for cs in services_conf[s]:
                LB += z[s, list(cs.keys())[0]] * (-1 * services_P[s])

    if hasattr(rho[list(rho.keys())[0]], 'X'):
        for f in functions:
            for c in functions_compl[f]:
                for j in range(1, J_MAX + 1):
                    for r in budget:
                        LB += (1/3) * rho[f, c, j, r].X / budget[r]
    return LB

# #########################################################################################################

def shared_elements(A, B):
    set_A = set(A)
    set_B = set(B)
    shared_set = set_A.intersection(set_B)
    shared_list = list(shared_set)
    return shared_list

# #########################################################################################################

def modify_serv_conf_c(serv_conf, funct, new_c):
    for el in serv_conf:
        if el[0] == funct:
            el[1] = new_c
    # serv_conf = [tuple(el) for el in serv_conf]
    return serv_conf

# #####################################################################################################################

def modify_serv_conf_j(serv_conf, funct, new_j):
    for el in serv_conf:
        if el[0] == funct:
            el[2] = new_j
    # serv_conf = [tuple(el) for el in serv_conf]
    return serv_conf

# #########################################################################################################

def get_dict_with_key(lst, K):
    for d in lst:
        if K in d:
            return d
    return None

# #########################################################################################################

def check_lambda_aux(lambda_aux_UB, functions, functions_compl, semantic, J_MAX, cs_to_s, v_UB, z_UB):
    lambda_aux_new = {}

    # lambda_aux_new init
    for f in functions:
        for c in functions_compl[f]:
            for j in range(1, J_MAX + 1):
                for sem in semantic[f]:
                    lambda_aux_new[f, sem, c, j] = 0

    for f in semantic:
        for sem in semantic[f]:
            for c in functions_compl[f]:
                for j in range(1, J_MAX + 1):
                    for cs in semantic[f][sem]:
                        if (v_UB[cs, f, c, j] == 1) and (z_UB[cs_to_s[cs], cs] == 1):
                            lambda_aux_new[f, sem, c, j] = 1
    return lambda_aux_new

# #####################################################################################################################

def actual_quality_computation(current_config, cs, quality_mapping_x, quality_mapping_q, f_multiplier, f_multiplier_c):
    # print("Actual quality computation for cs :",cs)
    '''
    Given the current setting, compute the actual expected quality
    '''
    actual_quality = 0
    x_tot = 0
    if len(current_config[0]) == 3:
        for f_num, (ff, cc, jj) in enumerate(current_config):
            x_tot += f_multiplier[ff] + f_multiplier_c[ff] * cc
    else:
        for f_num, (ff, cc) in enumerate(current_config):
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

# #####################################################################################################################

def max_quality_comp(s, cs, functions_compl, q_target, quality_mapping_x, quality_mapping_q, f_multiplier, f_multiplier_c):
    config = []
    for ff in list(cs.values())[0]:
        cc = max(functions_compl[ff])
        config.append((ff, cc))

    return actual_quality_computation(config, cs, quality_mapping_x, quality_mapping_q, f_multiplier, f_multiplier_c)

# #####################################################################################################################

def check_substring_letter(input_str):
    if len(input_str) != 9:
        return False
    if input_str[:8] != "Scenario":
        return False
    if not input_str[8].isupper():
        return False
    return True

# #####################################################################################################################

def convert_key_xApp_mem_req(key):
    # Remove the parentheses and split by comma
    key = key.strip("()")
    parts = key.split(", ")
    # Convert to the appropriate types (string, int, string)
    return (parts[0], int(parts[1]), parts[2])

def convert_key_theta(key):
    # Remove the parentheses and split by comma
    key = key.strip("()")
    parts = key.split(", ")
    # Convert to the appropriate types (string, int, string)
    return (parts[0], int(parts[1]))

# Function to convert string key to list of tuples
def convert_key_service_output_quality(key):
    # Split the string by commas followed by parentheses and spaces, then filter out empty strings
    parts = [part for part in key.split("),(") if part]
    # Remove leading and trailing parentheses and split by comma, then convert to tuple
    tuples = [tuple(item.strip("()").split(",")) for item in parts]
    # Convert integer part of the tuple appropriately
    tuples = [(t[0], int(t[1])) for t in tuples]
    return tuples

# #####################################################################################################################

def process_custom_scenario(services, services_L, services_Q, services_P, service_freq, functions, functions_compl, services_conf, resource, budget, xApp_mem_req, theta, semantic, services_conf_graph, service_output_quality):
    N_services = len(services)
    max_num_servconf = max([len(services_conf[s]) for s in services])
    N_functions = len(functions)
    max_n_functions_per_s = max([len(cs[list(cs.keys())[0]]) for s in services for cs in services_conf[s]])
    max_num_compl = max([len(functions_compl[f]) for f in functions])
    priority_lvls = len(set([services_P[s] for s in services]))

    cs_to_s = {}
    for s in services:
        for cs in services_conf[s]:
            cs_to_s[list(cs.keys())[0]] = s

    cs_list = []
    for s in services:
        for cs in services_conf[s]:
            cs_list.append(list(cs.keys())[0])

    for cs in services_conf_graph:
        if type(services_conf_graph[cs]) == type([1.]):
            G = nx.DiGraph()
            G.add_node(services_conf_graph[cs][0])
            services_conf_graph[cs] = G
        else:
            services_conf_graph[cs] = nx.DiGraph(services_conf_graph[cs])

    services_conf_graph_output = {}
    for s in services:
        for cs in services_conf[s]:
            services_conf_graph_output[list(cs.keys())[0]] = list(services_conf_graph[list(cs.keys())[0]].nodes())[0]

    services_conf_graph_former = {}
    for s in services:
        for cs in services_conf[s]:
            services_conf_graph_former[list(cs.keys())[0]] = [f for f in list(cs.values())[0] if f != list(services_conf_graph[list(cs.keys())[0]].nodes())[0]]

    semantic_cs = {}
    for f in functions:
        semantic_cs[f] = {}
        for cs in cs_list:
            semantic_cs[f][cs] = -1
            for sem in semantic[f]:
                if cs in semantic[f][sem]:
                    semantic_cs[f][cs] = sem

    lambda_semantic = {}
    for f in functions:
        lambda_semantic[f] = {}
        for sem in semantic[f]:
            lambda_semantic[f][sem] = max([service_freq[cs_to_s[cs]] for cs in semantic[f][sem]])

    xApp_mem_req = {convert_key_xApp_mem_req(k): v for k, v in xApp_mem_req.items( )}
    theta = {convert_key_theta(k): v for k, v in theta.items( )}

    f_multiplier = {}
    f_multiplier_c = {}
    for f_i, f in enumerate(functions):
        f_multiplier[f] = 5 ** (f_i + 1)
        f_multiplier_c[f] = 5 ** (f_i)

    # Create a new nested dictionary with converted keys
    service_output_quality_conv = {outer_key: {tuple(convert_key_service_output_quality(inner_key)): value for inner_key, value in inner_dict.items( )} for
        outer_key, inner_dict in service_output_quality.items( )}

    quality_mapping_x = {}
    quality_mapping_q = {}
    for s in services:
        for cs in services_conf[s]:
            quality_mapping_x[list(cs.keys())[0]] = []
            quality_mapping_q[list(cs.keys())[0]] = []

    for s in services:
        for cs in services_conf[s]:
            combs = list(product(
                *(zip([func] * len(functions_compl[func]), functions_compl[func]) for func in list(cs.values( ))[0])
            )
            )
            # print(combs)
            x_tmp = []
            q_tmp = []

            for comb in combs:

                x_value = 0
                for ff, cc in comb:
                    x_value += f_multiplier[ff] + f_multiplier_c[ff] * (cc)
                # print(x_value)

                # q_value = 0
                # for ff, cc in comb:
                #     if ff == services_conf_graph_output[list(cs.keys( ))[0]]:
                #         q_value += 0.9 * xApp_q[list(cs.keys( ))[0], ff, cc]
                #     else:
                #         q_value += 0.1 * xApp_q[list(cs.keys( ))[0], ff, cc]

                q_value = service_output_quality_conv[list(cs.keys())[0]][comb]

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

    return services, services_L, services_Q, services_P, service_freq, functions, functions_compl, services_conf, resource, \
           budget, xApp_mem_req, theta, semantic, services_conf_graph, service_output_quality,\
           N_services, max_num_servconf, N_functions, max_n_functions_per_s, max_num_compl, priority_lvls, cs_to_s,\
           cs_list, services_conf_graph_output, services_conf_graph_former, semantic_cs, lambda_semantic, \
           f_multiplier, f_multiplier_c, quality_mapping_x, quality_mapping_q
