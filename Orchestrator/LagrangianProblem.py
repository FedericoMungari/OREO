'''

Here are some tips to speed up the resolution of a problem in Gurobi:

1. Improve your model formulation: A well-formulated model can significantly reduce the time taken to solve a problem.
Ensure that the constraints and objectives are correctly formulated, and unnecessary variables are removed.
2. Set appropriate solver parameters: Gurobi has several solver parameters that can be set to optimize the solver
performance. For example, the parameter 'Presolve' can be set to 'Aggressive' to enable more preprocessing, which can
improve the solver's efficiency.
3. Use warm-starting: Warm-starting allows you to provide an initial feasible solution to the solver, which can
significantly reduce the time taken to find a solution.
4. Reduce problem size: You can reduce the size of the problem by removing unnecessary constraints, variables, or
objective functions. This can help improve the solver's efficiency.
5. Use parallel processing: Gurobi can use multiple processors to solve a problem faster. You can set the parameter
'Threads' to the number of processors available on your machine to utilize parallel processing.
6. Use Gurobi's presolve feature: Gurobi's presolve feature can analyze the model and make simplifications, such as
removing redundant constraints or variables. This can significantly reduce the size of the problem and speed up the
solver.
7. Consider using a different algorithm: Gurobi offers several algorithms for solving optimization problems. If you are
not satisfied with the performance of the default algorithm, you can try using a different one.
8. Consider upgrading hardware: Optimization problems can be computationally intensive, and upgrading hardware, such as
getting a faster CPU or more RAM, can help solve problems faster.


Here are some additional Gurobi solver parameters that can be set to optimize solver performance:

i.  TimeLimit: This parameter specifies the maximum amount of time (in seconds) that Gurobi should spend trying to solve
    the problem. Setting a reasonable time limit can help prevent the solver from getting stuck on difficult problems.
ii. MIPFocus: This parameter controls the trade-off between finding good feasible solutions quickly and searching for
    better solutions. Setting MIPFocus to 1 or 2 can help the solver find good solutions quickly, while setting it to 3
    can help the solver spend more time searching for better solutions.
iii.Cuts: This parameter controls the number of cutting planes that Gurobi generates during the optimization process.
    Increasing the number of cuts can help the solver find better solutions more quickly, but it can also increase the time
    required to solve the problem.
iv. Heuristics: This parameter controls the use of heuristics (i.e., problem-specific techniques for finding good solutions
    quickly) during the optimization process. Setting Heuristics to 0 can disable heuristics and force the solver to use
    only exact methods, which can be slower but can also find better solutions.
v.  NumericFocus: This parameter controls the trade-off between numerical accuracy and performance. Setting NumericFocus to
    0 can prioritize accuracy over performance, while setting it to 1 can prioritize performance over accuracy.
vi. FeasibilityTol: This parameter specifies the tolerance for determining whether a solution is feasible. Setting
    FeasibilityTol to a smaller value can improve the accuracy of the solver, but it can also increase the time required
    to solve the problem.
vii.OptimalityTol: This parameter specifies the tolerance for determining whether a solution is optimal. Setting
    OptimalityTol to a smaller value can improve the accuracy of the solver, but it can also increase the time required
    to solve the problem.

'''


import gurobipy as gp


class LP1:
    def __init__( self, services, services_P, services_Q, services_conf, services_conf_graph_output, functions,
                  functions_compl, beta, gamma, delta, cs_list, BigM = 1, seed_val = None, gp_printing = False, timelimit = None, MIPGap_v = None):
        self.m = gp.Model()
        self.m.Params.LogToConsole = gp_printing
        self.m.Params.OutputFlag = gp_printing
        # self.m.Params.Presolve = 0
        self.m.Params.MIPFocus = 1
        # self.m.Params.Cuts = 0
        # self.m.Params.Heuristics = 0.1
        # self.m.Params.NonConvex = 2
        if timelimit != None and timelimit > 0:
            self.m.setParam('TimeLimit', timelimit)
        if seed_val != None:
            self.m.setParam(gp.GRB.Param.Seed, seed_val)
        if MIPGap_v != None:
            self.m.Params.MIPGap = 0.01
            pass

        # #########################################################################################################
        # PARAMETERS
        self.beta = beta.copy()
        self.gamma = gamma.copy()
        self.delta = delta.copy()

        self.BigM = BigM
        self.cs_list = cs_list

        # #########################################################################################################
        # DECISION VARIABLES

        # --> Service configuration selection
        self.z = {}
        for s in services:
            for cs in services_conf[s]:
                self.z[s, list(cs.keys())[0]] = self.m.addVar(vtype = gp.GRB.BINARY,
                                                               name = 'z_%s_%s' % (s, list(cs.keys())[0])
                                                               )

        # #########################################################################################################
        # CONSTRAINTS

        # --> Constraints: at most one service configuration selection per service
        for s in services:
            self.m.addConstr(gp.quicksum(self.z[s, list(cs.keys())[0]] for cs in services_conf[s]) <= 1,
                             'config_selection_%s' % (s)
                             )

        # #########################################################################################################
        # OBJECTIVE FUNCTION

        self.m.update()
        self.set_obj(services, services_P, services_Q, services_conf, functions, functions_compl)

    def set_obj(self, services, services_P, services_Q, services_conf, functions, functions_compl):
        self.m.setObjective(
            -1 * gp.quicksum(gp.quicksum(
                self.z[s, list(cs.keys())[0]] *
                (services_P[s] - self.gamma[s, list(cs.keys())[0]] * services_Q[s] -
                 self.BigM * self.delta[s, list(cs.keys())[0]] -
                 gp.quicksum(self.beta[s, list(cs.keys())[0], f] for f in functions_compl))
            for cs in services_conf[s]) for s in services),
            gp.GRB.MINIMIZE
        )

    def optimize(self):
        self.m.update()
        self.m.optimize()


class LP2:
    def __init__(self, services, services_P, services_Q, services_conf, services_conf_graph_output,
                 services_conf_graph_former, J_MAX,
                 quality_mapping_x, quality_mapping_q, f_multiplier, f_multiplier_c,
                 xApp_mem_req, functions, functions_compl, beta, gamma, delta, budget, theta,
                 semantic, lambda_semantic, cs_list, services_L,
                 # v_prime, lambda_xApp, n_aux_prime, cs_list_all_f, services_all_f,
                 max_latency = 1.0, seed_val = None, gp_printing = False, timelimit = None, MIPGap_v = None):

        self.m = gp.Model()
        self.m.Params.LogToConsole = gp_printing # True
        self.m.Params.OutputFlag = gp_printing # True
        # self.m.Params.LPWarmStart = 1
        # self.m.Params.Presolve = 0
        self.m.Params.MIPFocus = 1
        # self.m.Params.Cuts = 0
        # self.m.Params.Heuristics = 0.1
        self.m.Params.NonConvex = 2
        if timelimit != None and timelimit > 0:
            self.m.setParam('TimeLimit', timelimit)
        if seed_val != None:
            self.m.setParam(gp.GRB.Param.Seed, seed_val)
        if MIPGap_v != None:
            self.m.Params.MIPGap = MIPGap_v
            pass

        # #########################################################################################################
        # PARAMETERS

        self.budget_mem = {r:v for r,v in budget.items()} #if r != 'cpu'}

        self.beta = beta
        self.gamma = gamma
        self.delta = delta

        self.theta = theta
        self.cs_list = cs_list

        # #########################################################################################################
        # DECISION VARIABLES

        # --> xApp and services mapping
        self.v = {}
        for s in services:
            for cs in services_conf[s]:
                for f in functions:
                    for c in functions_compl[f]:
                        for j in range(1, J_MAX + 1):
                            self.v[list(cs.keys())[0], f, c, j] = self.m.addVar(vtype = gp.GRB.BINARY,
                                                                                 name = 'v_%s_%s_%s_%s' % (
                                                                                     list(cs.keys())[0], f, c, j))

        # --> xApp resource allocation
        self.rho = {}
        for f in functions:
            for c in functions_compl[f]:
                for j in range(1, J_MAX + 1):
                    for r in self.budget_mem:
                        self.rho[f, c, j, r] = self.m.addVar(vtype = gp.GRB.CONTINUOUS,
                                                     lb = 0.0,
                                                     # lb = -float('inf'),
                                                     ub = self.budget_mem[r],
                                                     name = 'rho_%s_%s_%s_%s' % (f, c, j, r))


        # #########################################################################################################
        # AUXILIARY DECISION VARIABLES

        # --> n_aux auxiliary variable tells if an xApp (f,c,j) is utilized by a service s
        # --> Auxiliary variable: eta == n_aux
        self.n_times_v = {}
        for f in functions:
            for c in functions_compl[f]:
                for j in range(1, J_MAX + 1):
                    self.n_times_v[f, c, j] = self.m.addVar(vtype = gp.GRB.INTEGER,
                                                            name = 'n_times_v_%s_%s_%s' % (f, c, j)
                                                            )
        self.n_aux = {}
        for f in functions:
            for c in functions_compl[f]:
                for j in range(1, J_MAX + 1):
                    self.n_aux[f, c, j] = self.m.addVar(vtype = gp.GRB.BINARY, name = 'n_aux_%s_%s_%s' % (f, c, j))

        for f in functions:
            for c in functions_compl[f]:
                for j in range(1, J_MAX + 1):
                    self.m.addConstr(self.n_times_v[f, c, j] == gp.quicksum(
                        gp.quicksum(self.v[list(cs.keys( ))[0], f, c, j] for cs in services_conf[s]) for s in services
                    ), name = 'n_times_v_def_%s_%s_%s' % (f, c, j)
                                     )
        for f in functions:
            for c in functions_compl[f]:
                for j in range(1, J_MAX + 1):
                    self.m.addGenConstrIndicator(self.n_aux[f, c, j], True, self.n_times_v[f, c, j],
                                                 gp.GRB.GREATER_EQUAL, 0.5
                                                 )
                    self.m.addGenConstrIndicator(self.n_aux[f, c, j], False, self.n_times_v[f, c, j], gp.GRB.LESS_EQUAL,
                                                 0.5
                                                 )
        # --> Auxiliary variables: f_per_cs[cs,f]
        # f_per_cs tells me if, for every cs and function f, if there is or there is not an xApp providing f for cs
        # namely if it exists c,j s.t. v_[cs,f,c,j] == 1
        # f_per_cs is then a binary variable
        self.f_per_cs = {}
        for s in services:
            for cs in services_conf[s]:
                # for f in functions:
                for f in list(cs.values())[0]:
                    self.f_per_cs[list(cs.keys())[0], f] = self.m.addVar(vtype = gp.GRB.BINARY,
                                                                          name = 'f_per_cs_%s_%s' % (list(cs.keys())[0], f))

        for s in services:
            for cs in services_conf[s]:
                # for f in functions:
                for f in list(cs.values())[0]:
                    self.m.addGenConstrIndicator(self.f_per_cs[list(cs.keys())[0], f],
                                                 True,
                                                 gp.quicksum(gp.quicksum(self.v[list(cs.keys())[0], f, c, j]
                                                                         for c in functions_compl[f])
                                                             for j in range(1, J_MAX + 1)),
                                                 gp.GRB.GREATER_EQUAL, 0.5)
                    self.m.addGenConstrIndicator(self.f_per_cs[list(cs.keys())[0], f],
                                                 False,
                                                 gp.quicksum(gp.quicksum(self.v[list(cs.keys())[0], f, c, j]
                                                                         for c in functions_compl[f])
                                                             for j in range(1, J_MAX + 1)),
                                                 gp.GRB.LESS_EQUAL, 0.5)

        '''
        Avendo messo il constraint sum(v[cs,f,c,j]) = 1 for every f in cs, non serve più introdurre una variabile che mi 
        dica se per ogni cs tutte le funzioni sono state implementate. all_f_per_cs[cs] sarà sempre 1
        '''
        # --> Auxiliary variables: all_f_per_cs[cs]
        # all_f_per_cs tells me if, for every cs, ALL the involved functions are implemented (and associated to cs)
        # namely if FOR EVERY f in cs it exists (c,j) s.t. v_[cs,f,c,j] == 1
        # f_per_cs is then a binary variable
        self.all_f_per_cs = {}
        for s in services:
            for cs in services_conf[s]:
                self.all_f_per_cs[list(cs.keys())[0]] = self.m.addVar(vtype = gp.GRB.BINARY,
                                                                       name = 'all_f_per_cs_%s' % (list(cs.keys())[0]))
        for s in services:
            for cs in services_conf[s]:
                self.m.addGenConstrIndicator(self.all_f_per_cs[list(cs.keys())[0]], True,
                                             gp.quicksum(self.f_per_cs[list(cs.keys())[0], f] for f in list(cs.values())[0]),
                                             gp.GRB.EQUAL, len(list(cs.values())[0])
                                             )
                self.m.addGenConstrIndicator(self.all_f_per_cs[list(cs.keys())[0]], False,
                                             len(list(cs.values())[0]) - gp.quicksum(
                                                 self.f_per_cs[list(cs.keys())[0], f] for f in list(cs.values())[0]
                                             ), gp.GRB.GREATER_EQUAL, 0.5
                                             )

        # --> Auxiliary variables: lambda_aux[f,sem]
        self.lambda_aux = {}
        for f in semantic:
            for sem in semantic[f]:
                for c in functions_compl[f]:
                    for j in range(1, J_MAX + 1):
                        self.lambda_aux[f,sem,c,j] = self.m.addVar(vtype = gp.GRB.BINARY,
                                                                   name = 'lambda_aux_%s_%s_%s_%s' % (f,sem,c,j))

        # --> Auxiliary variables: lambda_aux[f,sem]
        for f in semantic:
            for sem in semantic[f]:
                for c in functions_compl[f]:
                    for j in range(1, J_MAX + 1):
                        self.m.addGenConstrIndicator(self.lambda_aux[f,sem,c,j],
                                                     1.0,
                                                     gp.quicksum(self.v[cs,f,c,j] for cs in semantic[f][sem]),
                                                     gp.GRB.GREATER_EQUAL,
                                                     0.5)
                        self.m.addGenConstrIndicator(self.lambda_aux[f,sem,c,j],
                                                     0.0,
                                                     gp.quicksum(self.v[cs,f,c,j] for cs in semantic[f][sem]),
                                                     gp.GRB.LESS_EQUAL,
                                                     0.5
                                                     )

        # --> Auxiliary variables: q[cs]
        # The variable q[cs] tells the nominal quality with which the service s in configuration cs has delivered
        self.q = {}
        for s in services:
            for cs in services_conf[s]:
                self.q[list(cs.keys())[0]] = self.m.addVar(vtype = gp.GRB.CONTINUOUS,
                                                           name = 'q_%s' % (list(cs.keys())[0]))


        # --> Auxiliary variables: q_computed[cs]
        # --> Auxiliary variables: q_computed_multiplier[cs]
        # The variables q_computed[cs] and q_computed_multiplier[cs] are used to compute the quality of the service s
        # under configuration cs. In particular, since we modeled the service quality as a piecewise linear function,
        # the variable q_computed_multiplier[cs] is used to compute the x value of the piecewise linear function, while
        # the variable q_computed[cs] is used to compute the y value of the piecewise linear function, namely the
        # quality of the service s under configuration cs.
        self.q_computed = {}
        for s in services:
            for cs in services_conf[s]:
                self.q_computed[list(cs.keys())[0]] = self.m.addVar(vtype = gp.GRB.CONTINUOUS,
                                                            name = 'q_computed_%s' % (list(cs.keys())[0]))
        self.q_computed_multiplier = {}
        for s in services:
            for cs in services_conf[s]:
                self.q_computed_multiplier[list(cs.keys( ))[0]] = self.m.addVar(vtype = gp.GRB.CONTINUOUS,
                                                                                name = 'q_computed_multiplier_%s' % (list(cs.keys( ))[0]))
                self.m.addConstr(self.q_computed_multiplier[list(cs.keys( ))[0]] == gp.quicksum(gp.quicksum(gp.quicksum(
                    self.v[list(cs.keys( ))[0], f, c, j] * (f_multiplier[f] + f_multiplier_c[f] * (c))
                    for j in range(1, J_MAX + 1)) for c in functions_compl[f]) for f in list(cs.values( ))[0]),
                                 name = 'q_computed_multiplier_%s' % (list(cs.keys( ))[0]))

        for s in services:
            for cs in services_conf[s]:
                self.m.addGenConstrPWL(
                    # gp.quicksum(gp.quicksum(gp.quicksum(
                    #     self.v[list(cs.keys())[0], f, c, j] * self.q_computed_multiplier[f,c]
                    #     for j in range(1, J_MAX + 1)) for c in functions_compl[f]) for f in functions), # x
                    xvar=self.q_computed_multiplier[list(cs.keys())[0]], # x
                    yvar=self.q_computed[list(cs.keys( ))[0]], # y
                    xpts=list(quality_mapping_x[list(cs.keys())[0]]), # x pts
                    ypts=list(quality_mapping_q[list(cs.keys())[0]]), # y pts
                    name="quality_table_%s" % (list(cs.keys())[0]))


        for s in services:
            for cs in services_conf[s]:
                '''Abbiamo tolto self.all_f_per_cs perchè sempre valore unitario. Guarda sopra per info'''
                self.m.addConstr((self.all_f_per_cs[list(cs.keys())[0]] == 1) >> (self.q[list(cs.keys())[0]] == self.q_computed[list(cs.keys())[0]]),
                                name = 'q_def_%s' % (list(cs.keys())[0]))
                self.m.addConstr((self.all_f_per_cs[list(cs.keys())[0]] == 0) >> (self.q[list(cs.keys())[0]] == 0.0),
                                 name = 'q_ub_%s' % (list(cs.keys())[0]))
                # self.m.addConstr(self.q[list(cs.keys())[0]] == self.q_computed[list(cs.keys())[0]],
                #                  name = 'q_def_%s' % (list(cs.keys())[0]))

        self.tau_aux_1 = {}
        for f, c, j in self.n_aux:
            self.tau_aux_1[f, c, j, 'cpu'] = self.m.addVar(vtype = gp.GRB.CONTINUOUS,
                                                      # lb = 0.0,
                                                      lb = -float('inf'),
                                                      ub = float('inf'),
                                                      # ub = max_latency,
                                                      name = 'tau_aux_1_%s_%s_%s_%s' % (f, c, j, 'cpu')
                                                      )
        for f, c, j in self.n_aux:
            self.m.addConstr(self.tau_aux_1[f, c, j, 'cpu'] *
                             (self.rho[f, c, j, 'cpu'] * self.theta[f, c] - gp.quicksum(
                                         self.lambda_aux[f,sem,c,j] * lambda_semantic[f][sem] for sem in semantic[f])) == 1,
                             name = 'tau_aux_1_def_selectedxApp_%s_%s_%s' % (f, c, j))

        # # --> Auxiliary variables: tau[cs]
        self.tau = {}
        for cs in cs_list:
            self.tau[cs] = self.m.addVar(vtype = gp.GRB.CONTINUOUS,
                                                         lb = 0.0,
                                                         # lb = -1*float('inf'),
                                                         # ub = max_latency,
                                                         ub = 1000,
                                                         # ub = float('inf'),
                                                         name = 'tau_%s' % (cs)
                                                         )
        # --> Definition of auxiliary variable: tau[cs]
        for cs in cs_list:
            self.m.addConstr(self.tau[cs] == gp.quicksum(self.tau_aux_1[f, c, j, 'cpu']*self.v[cs,f,c,j] for f, c, j in self.n_aux),
                             name = 'tau_def_%s' % (cs))
            '''Abbiamo tolto self.all_f_per_cs perchè sempre valore unitario. Guarda sopra per info'''

        # #########################################################################################################
        # rho[CPU] SUBDOMAINs
        # rho[cpu] can either takes 0 value - when the xApp is not instantiated (as we will see, n_aux[f,c,j] = 0
        # or rho[cpu] can takes values larger than  - when the xApp is not instantiated (as we will see, n_aux[f,c,j] = 0
        max_theta = -1
        min_theta = 999
        for f, c, j in self.n_aux:
                if theta[f, c] > max_theta:
                    max_theta = theta[f, c]
                elif min_theta > theta[f, c]:
                    min_theta = theta[f, c]

        max_lambda = -1
        min_lambda = 999
        for f in lambda_semantic:
            lambda_function = 0
            for sem in lambda_semantic[f]:
                lambda_function += lambda_semantic[f][sem]
            if lambda_function > max_lambda:
                max_lambda = lambda_function
            elif min_lambda > lambda_function and lambda_function>0:
                min_lambda = lambda_function

        rho_lb = ((1/max_latency) + min_lambda) / max_theta

        for f,c,j in self.n_aux:
            # self.m.addConstr(rho_lb * self.n_aux[f, c, j] <= self.rho[f, c, j, 'cpu'], 'rho_LB_subdomain_constr_%s_%s_%s' % (f, c, j))
            self.m.addConstr(0.01 <= self.rho[f, c, j, 'cpu'], 'rho_LB_subdomain_constr_%s_%s_%s' % (f, c, j))
            self.m.addConstr(self.rho[f, c, j, 'cpu'] <= self.budget_mem['cpu'] * self.n_aux[f, c, j] + 0.01, 'rho_UB_subdomain_constr_%s_%s_%s' % (f, c, j))




        # #########################################################################################################
        # CONSTRAINTS
        if True: # PRUNING: it is necessary to apply pruning with LP1, LP2 and LP3
            # --> Constraints: exactly one xApp per f \in cs
            for s in services:
                for cs in services_conf[s]:
                    # for f in functions:
                    for f in list(cs.values())[0]:
                        self.m.addConstr(gp.quicksum(gp.quicksum(
                                self.v[list(cs.keys())[0], f, c, j]
                                for c in functions_compl[f]) for j in range(1, J_MAX + 1)) == 1,
                            'xApp_per_f_%s_%s_%s' %(s, list(cs.keys())[0], f))
        # else:
        #     # --> Constraints: at most one xApp per f \in cs
        #     for s in services:
        #         for cs in services_conf[s]:
        #             for f in functions:
        #                 self.m.addConstr(gp.quicksum(gp.quicksum(
        #                         self.v[list(cs.keys())[0], f, c, j]
        #                         for c in functions_compl[f]) for j in range(1, J_MAX + 1)) <= 1,
        #                     'xApp_per_f_%s_%s_%s' %(s, list(cs.keys())[0], f))

        # --> Constraints: do not instantiate xApps providing function f to cs if f is not foreseen by cs
        for s in services:
            for cs in services_conf[s]:
                for f in [ff for ff in functions if ff not in list(cs.values())[0]]:
                    self.m.addConstr(gp.quicksum(gp.quicksum(
                        self.v[list(cs.keys())[0], f, c, j]
                        for c in functions_compl[f]) for j in range(1, J_MAX + 1)) == 0,
                         'funct_selection_%s_%s_%s' % (s, list(cs.keys())[0], f))

        # --> Constraints: xApp instance ordering
        # for f in functions:
        #     for c in functions_compl[f]:
        #         for j in range(1, J_MAX + 1):
        #             for jj in range(j + 1, J_MAX + 1):
        #                 self.m.addConstr(self.n_aux[f, c, jj] <= self.n_aux[f, c, j],
        #                                  'xApp_ordering_%s_%s_%s_%s' % (f, c, j, jj)
        #                                  )


        # --> Constraints: MEMORY/DISK : allocate mem requirements for implemented xApps
        #                                do not reserve resources for not implemented xApps
        for f,c,j in self.n_aux:
            for r in self.budget_mem:
                if r != 'cpu':
                    self.m.addConstr(self.rho[f, c, j, r] == self.n_aux[f, c, j] * xApp_mem_req[f,c,r],
                                     'resource_domain_%s_%s_%s_%s' % (f, c, j, r)
                                     )

        # --> resource budget
        for r in self.budget_mem:
            self.m.addConstr(gp.quicksum(self.rho[f, c, j, r] for f,c,j in self.n_aux) <= self.budget_mem[r],
                'budget_%s' % (r))

        # #########################################################################################################
        # OBJECTIVE FUNCTION

        self.m.update()
        self.set_obj(services, services_conf, functions, functions_compl, J_MAX, self.budget_mem, services_L)

    def set_obj(self, services, services_conf, functions, functions_compl, J_MAX, budget, services_L):

        self.m.setObjective(
            (-1) * gp.quicksum(gp.quicksum(
                self.gamma[s, list(cs.keys())[0]] * self.q[list(cs.keys())[0]]
                + gp.quicksum(gp.quicksum(gp.quicksum(
                    self.beta[s, list(cs.keys())[0], f] * self.v[list(cs.keys())[0], f, c, j]
                    for j in range(1, J_MAX + 1)) for c in functions_compl[f])for f in list(cs.values())[0])
                for cs in services_conf[s]) for s in services) +
            (-1) * gp.quicksum(gp.quicksum((-1/3) * self.rho[f,c,j,r] / self.budget_mem[r]
                 for r in self.budget_mem) for (f,c,j) in self.n_aux) + \
            (-1) * gp.quicksum(gp.quicksum(
                -1 * self.tau[list(cs.keys( ))[0]] * self.delta[s, list(cs.keys( ))[0]] for cs in services_conf[s]) for s in services),
            gp.GRB.MINIMIZE)

    def optimize(self):
        self.m.update()
        self.m.optimize()