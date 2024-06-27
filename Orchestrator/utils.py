
import math
import networkx as nx
import numpy as np

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