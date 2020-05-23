from __future__ import division
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import seaborn as sns
import math
import numpy as np
import midaco
from PyGMO.util import hypervolume
from PyGMO import population
from PyGMO import individual
from PyGMO import algorithm
from PyGMO.problem import base
from math import sqrt
from pylab import *
from PyGMO import *
import ast
import warnings
import csv


class my_mo_problem(base):
    """
    A multi-objective problem.
    Optimum Design of Disk Brake Using NSGA-2 Algorithm

    USAGE: my_mo_problem()
    """

    def __init__(self, dim=4):
        # 1 integer variable, 2 objective, 5 total constraint, 5 inequality constraint
        super(my_mo_problem,self).__init__(dim, 1, 2, 5, 5)
        b_inf = [55,75,1000,2]
        b_sup = [80,110,3000,20]
        self.set_bounds(b_inf, b_sup)
    
    
    def _objfun_impl(self, x):
        
        f0 = 0.000049*(pow(x[1],2) - pow(x[0],2))*(x[3] - 1)
        f1 = ((9.82*pow(10,6))*(pow(x[1],2) - pow(x[0],2))) / ((x[2]*x[3])*(pow(x[1],3) - pow(x[0],3)))
        
        return (f0, f1, )
    
    def _compute_constraints_impl(self, x):
        c = list()
        g0 = -1*((x[1] - x[0]) - 20) 
        g1 = -1*(30 - 2.5*(x[3] + 1))
        g2 = -1*(0.4 - (x[2] / (3.14*(pow(x[1],2) - pow(x[0],2)))))
        g3 = -1*(1 - (0.00222*x[2]*((pow(x[1],3) - pow(x[0],3)))) / ((pow(x[1],2) - pow(x[0],2))**2))
        g4 = -1*(((0.0266*x[2]*x[3]*((pow(x[1],3) - pow(x[0],3)))) / ((pow(x[1],2) - pow(x[0],2)))) - 900)
        c = [g0,g1,g2,g3,g4]
       
        return c
    
    def human_readable_extra(self):
        return "\n\tMulti-Objective problem"



def eval_fo(x):
    f0 = 0.000049*(pow(x[1],2) - pow(x[0],2))*(x[3] - 1)
    f1 = ((9.82*pow(10,6))*(pow(x[1],2) - pow(x[0],2))) / ((x[2]*x[3])*(pow(x[1],3) - pow(x[0],3)))
    return [f0, f1]

def eval_con(x):
    g0 = -1*((x[1] - x[0]) - 20) 
    g1 = -1*(30 - 2.5*(x[3] + 1))
    g2 = -1*(0.4 - (x[2] / (3.14*(pow(x[1],2) - pow(x[0],2)))))
    g3 = -1*(1 - (0.00222*x[2]*((pow(x[1],3) - pow(x[0],3)))) / ((pow(x[1],2) - pow(x[0],2))**2))
    g4 = -1*(((0.0266*x[2]*x[3]*((pow(x[1],3) - pow(x[0],3)))) / ((pow(x[1],2) - pow(x[0],2)))) - 900)
    return [g0,g1,g2,g3,g4]

def Epsilon_Constraint(x):
    f = [0.0]*1
    f[0] = ((9.82*pow(10,6))*(pow(x[1],2) - pow(x[0],2))) / ((x[2]*x[3])*(pow(x[1],3) - pow(x[0],3)))

    f1 = 0.000049*(pow(x[1],2) - pow(x[0],2))*(x[3] - 1)
    g = [0.0]*6
    g[0] = epsilon - f1
    g[1] = ((x[1] - x[0]) - 20) 
    g[2] = (30 - 2.5*(x[3] + 1))
    g[3]= (0.4 - (x[2] / (3.14*(pow(x[1],2) - pow(x[0],2)))))
    g[4] = (1 - (0.00222*x[2]*((pow(x[1],3) - pow(x[0],3)))) / ((pow(x[1],2) - pow(x[0],2))**2))
    g[5] = (((0.0266*x[2]*x[3]*((pow(x[1],3) - pow(x[0],3)))) / ((pow(x[1],2) - pow(x[0],2)))) - 900)

    
    return f,g

########################################################################
### Step 1: Problem definition for MIDACO     ##########################
########################################################################

key = 'MIDACO_LIMITED_VERSION___[CREATIVE_COMMONS_BY-NC-ND_LICENSE]'

problem_midaco = {} # Initialize dictionary containing problem specifications
option  = {} # Initialize dictionary containing MIDACO options

problem_midaco['@'] = Epsilon_Constraint # Handle for problem function name

########################################################################
### Step 1: Problem definition     #####################################
########################################################################

# STEP 1.A: Problem dimensions
##############################
problem_midaco['o']  = 1  # Number of objectives 
problem_midaco['n']  = 4  # Number of variables (in total) 
problem_midaco['ni'] = 1  # Number of integer variables (0 <= ni <= n) 
problem_midaco['m']  = 6  # Number of constraints (in total) 
problem_midaco['me'] = 0  # Number of equality constraints (0 <= me <= m) 

# STEP 1.B: Lower and upper bounds 'xl' & 'xu'  
##############################################  
problem_midaco['xl'] = [ 55, 75, 1000, 2 ]
problem_midaco['xu'] = [ 80, 110, 3000, 20 ]

########################################################################
### Step 2: Choose stopping criteria and printing options    ###########
########################################################################
   
# STEP 2.A: Stopping criteria 
#############################
option['maxeval'] = 10000 # Maximum number of function evaluation (e.g. 1000000) <------------------------ important!!!
option['maxtime'] = 60*60*24  # Maximum time limit in Seconds (e.g. 1 Day = 60*60*24) 

# STEP 2.B: Printing options  
############################ 
option['printeval'] = 2000 # Print-Frequency for current best solution (e.g. 1000) 
option['save2file'] = 0    # Save SCREEN and SOLUTION to TXT-files [0=NO/1=YES]

########################################################################
### Step 3: Choose MIDACO parameters (FOR ADVANCED USERS)    ###########
########################################################################

option['param1']  = 0.0001  # ACCURACY  
option['param2']  = 0.0  # SEED  
option['param3']  = 0.0  # FSTOP  
option['param4']  = 0.0  # ALGOSTOP  
option['param5']  = 0.0  # EVALSTOP  
option['param6']  = 0.0  # FOCUS  
option['param7']  = 0.0  # ANTS  
option['param8']  = 0.0  # KERNEL  
option['param9']  = 0.0  # ORACLE  
option['param10'] = 0.0  # PARETOMAX
option['param11'] = 0.0  # ON  
option['param12'] = 0.0  # CHARACTER
option['parallel'] = 0 # Serial: 0 or 1, Parallel: 2,3,4,5,6,7,8...

# STEP 1 : THE ALGORITHM NSGA-2 + global mixed integer ant colony optimization (MIDACO)

if __name__ == '__main__':
    n_experiments = 1
    hv_nsga2_MID_list = []
    hv_nsga2_p_list = []
    for exp in range(n_experiments):
        print '\t Running NSGA-2 + global mixed integer ant colony optimization (MIDACO) \n'
        solutions_X = []
        solutions_F = []
        time_MIDACO = 0
        prob_multi = my_mo_problem()  # load the problem (constrained multiobjective)
        prob = problem.death_penalty(prob_multi,problem.death_penalty.method.SIMPLE) # define the constraint handling approach
        n_individuals = 40   # define the number of the individuals 
        n_generations = 80  # define the number of generations
        alg = algorithm.nsga_II(n_generations)
        pop = population(prob, n_individuals)
        start1 = timer()
        pop = alg.evolve(pop)
        end1 = timer()
        print 'NSGA-2 has finished with '+str(n_individuals)+' individuals and '+str(n_generations)+' generations in '+str(end1-start1)+' seconds'
        print 'Initialize MIDACO and solve the Epsilon Constraint model'
        for j in xrange(n_individuals):
            individual = pop[j].cur_x  # take the decision vector
            problem_midaco['x'] = [individual[0],individual[1],individual[2],individual[3]] # setting the starting point
            global epsilon
            epsilon = pop[j].cur_f[0] # epsilon is the value of the first objective function (of the j-th individual) in the last generation of NSGA-2 
            start2 = timer()         
            solution = midaco.run( problem_midaco, option, key )
            end2 = timer()
            solutions_X.append(solution['X'])
            end2 = timer()
            time_MIDACO += (end2-start2)
        print 'MIDACO has found the solution of '+str(n_individuals)+' MINLPc problems in '+str(time_MIDACO)+' seconds'
        print 'Total execution time Epsilon Constraint Method (NSGA-2 + MIDACO) '+str((end1-start1)+(time_MIDACO))+' seconds \n'
        solutions_F = np.asarray([eval_fo(list_sol) for list_sol in solutions_X if 0 not in list_sol]) # list of objective function values 
        
        ## STEP 2 : comparison with another indipendent NSGA-2

        print '\t Running NSGA-2 '
        prob_multi_p = my_mo_problem()
        prob_p = problem.death_penalty(prob_multi_p,problem.death_penalty.method.SIMPLE) # define the constraint handling approach
        n_individuals_p = 40
        n_generations_p = 2000
        alg_p = algorithm.nsga_II(n_generations_p)
        pop_p = population(prob_p, n_individuals_p)
        start3 = timer()
        pop_p = alg_p.evolve(pop_p)
        end3 = timer()
        print 'NSGA-2 has finished with '+str(n_individuals_p)+' individuals and '+str(n_generations_p)+' generations in '+str(end3-start3)+' seconds \n'

        ## STEP 3 : computing the hypervolume to evaluate the pareto fronts
        
        ref_point = (2,9)
        tuple_solutions_F = [(k) for k in solutions_F]
        hv_nsga2_MID = hypervolume(tuple_solutions_F,verify=False)
        HYP1 = hv_nsga2_MID.compute(r=ref_point)
        print '\t Computing the hypervolume (high is better) \n'
        print 'hypervolume value for epsilon constraint method (NSGA-2 + MIDACO) '+str(HYP1)
        hv_nsga2_MID_list.append(hv_nsga2_MID.compute(r=ref_point))
        
        hv_nsga2_p = hypervolume(pop_p,verify=False)
        HYP2 = hv_nsga2_p.compute(r=ref_point)
        print 'hypervolume value for NSGA-2 '+str(HYP2)+'\n'
        hv_nsga2_p_list.append(hv_nsga2_p.compute(r=ref_point))

        ## STEP 4 : Plot the Pareto front

        sns.set_style("whitegrid")
        plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
        plt.rc('text', usetex=True)
        F_p = np.array([ind.cur_f for ind in pop]).T
        F_p2 = np.array([ind.cur_f for ind in pop_p]).T
        fig = plt.gcf()
        fig.set_size_inches(12, 8)

        points_nsga_mi = plt.scatter(solutions_F[:,0], solutions_F[:,1],color = 'r',alpha=0.5) # plot the result of NSGA-2 + MIDACO
        points_nsga = plt.scatter(F_p2[0], F_p2[1],color = 'g',alpha = 0.5) # plot the other NSGA-2 for comparison
        plt.title('Pareto Fronts for the Optimum Design of Disk Brake')
        plt.legend((points_nsga_mi,points_nsga),('NSGA-2 ('+str(n_generations)+' generations) + Global Mixed Integer Ant Colony Optimization ('+str(option['maxeval'])+' max function evaluations) HYP = '+str(HYP1)+'','NSGA-2 ('+str(n_generations_p)+' generations) HYP = '+str(HYP2)+''),loc = 'upper right')
        plt.xlabel("$f^{(1)}$ Brake Mass (kg)")
        plt.ylabel("$f^{(2)}$ Stopping Time (s)")
        plt.show()

        

    ## FINAL STEP : saving the hypervolume values to file and computing the average of the results
##    diz = {}
##    diz['nsga+midaco'] = ('maxeval = '+str(option['maxeval']),'n_individuals = '+str(n_individuals),'n_generations = '+str(n_generations),hv_nsga2_MID_list)
##
##    diz['nsga'] = ('n_individuals = '+str(n_individuals_p),'n_generations = '+str(n_generations_p),hv_nsga2_p_list)
##    f = open('hypervolume_values.txt','a')
##    f.write(str(diz)+'\n')
##    f.close()
##    print 'hypervolume mean value (NSGA-2 + MIDACO) with '+str(n_experiments)+' measures is '+str(sum(hv_nsga2_MID_list)/len(hv_nsga2_MID_list))
##    print 'hypervolume mean value NSGA-2 with '+str(n_experiments)+' measures is '+str(sum(hv_nsga2_p_list)/len(hv_nsga2_p_list))
####    











