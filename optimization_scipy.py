import numpy as np
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
from objective_function import *
from constraints import *

def run_optimizer(constraint, grad_constraint, hess_v_constraint):

    # Equality constraints are specified by using the same lower and upper bounds
    constraint_lower_bound = 0.0;
    constraint_upper_bound = 0.0;

    x0 = np.array([0.0, 0.1]); #initial guess 
    eq_constraint = NonlinearConstraint(constraint, constraint_lower_bound, constraint_upper_bound, 
                                        jac=grad_constraint, hess=hess_v_constraint);
    res = minimize(obj_func, x0, method='trust-constr', jac=grad_obj_func, hess=hess_obj_func, 
                    constraints=eq_constraint, options={'gtol': 1e-8,'xtol':1e-12, 'maxiter': 250, 'verbose': 3});
    print("Optimal solution = ",res.x);
    return res.x;

print("===================================================================");
print("                 Optimization using SCIPY                          ");
print("===================================================================");
print('\n\n');
print("===================================================================");
print("Running optimization for f(x,y) = (1-x)^2 + 100*(y-x^2)^2");
print("With constraint c1(x,y) = 1-x-y = 0");
print("===================================================================");
run_optimizer(constraint_1, grad_constraint_1, hess_v_constraint_1);
print("===================================================================");
print('\n\n');
print("===================================================================");
print("Running optimization for f(x,y) = (1-x)^2 + 100*(y-x^2)^2");
print("With constraint c2(x,y) = 1-x^2-y^2 = 0");
print("===================================================================");
run_optimizer(constraint_2, grad_constraint_2, hess_v_constraint_2);
print("===================================================================");





