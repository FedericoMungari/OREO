
from numpy.random import randn

def beta_init(services,services_conf,functions,ub=None):
    if ub==None:
        ub = 0.1
    beta = {}
    for s in services:
        for cs in services_conf[s]:
            for f in functions:
                while True:
                    _ = ub/10 * randn() + ub/10
                    if _ > 0 and _ < ub:
                        beta[s,list(cs.keys())[0],f] = _
                        beta[s,list(cs.keys())[0],f] = _ / 10
                        # beta[s,list(cs.keys())[0],f] = 0
                        # beta[s,list(cs.keys())[0],f] = ub/10
                        break
    return beta

def gamma_init(services,services_conf,ub=None):
    if ub==None:
        ub = 0.1
    gamma = {}
    for s in services:
        for cs in services_conf[s]:
            while True:
                _ = ub/10 * randn() + ub/10
                if _ > 0 and _ < ub:
                    gamma[s,list(cs.keys())[0]] = _
                    gamma[s,list(cs.keys())[0]] = _ / 10
                    # gamma[s,list(cs.keys())[0]] = 0
                    # gamma[s,list(cs.keys())[0]] = ub / 10
                    break
    return gamma


def delta_init(services,services_conf,ub=None):
    if ub==None:
        ub = 0.1
    delta = {}
    for s in services:
        for cs in services_conf[s]:
            while True:
                _ = ub/10 * randn() + ub/10
                if _ > 0 and _ < ub:
                    delta[s,list(cs.keys())[0]] = _
                    delta[s,list(cs.keys())[0]] = _ / 10
                    # delta[s,list(cs.keys())[0]] = 0
                    # delta[s,list(cs.keys())[0]] = ub/10
                    break
    return delta