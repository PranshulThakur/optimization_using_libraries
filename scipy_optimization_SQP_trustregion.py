import numpy as np
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
from objective_function import *
from constraints import *


# Equality constraints are specified by using the same lower and upper bounds
constraint_lower_bound = 0.0;
constraint_upper_bound = 0.0;

x0 = np.array([0.0, 0.1]); #initial guess 

print("===================================================================");
print("                 Optimization using SQP                            ");
print("===================================================================");
print("Running optimization for f(x,y) = (1-x)^2 + 100*(y-x^2)^2");
print("With constraint c1(x,y) = 1-x-y = 0");
eq_constraint = NonlinearConstraint(constraint_1, constraint_lower_bound, constraint_upper_bound, 
                                    jac=grad_constraint_1, hess=hess_v_constraint_1);
res = minimize(obj_func, x0, method='trust-constr', jac=grad_obj_func, hess=hess_obj_func, 
                constraints=eq_constraint, options={'gtol': 1e-8,'xtol':1e-12, 'maxiter': 250, 'verbose': 1});
print("Optimal solution = ",res.x);

print("===================================================================");


print("Running optimization for f(x,y) = (1-x)^2 + 100*(y-x^2)^2");
print("With constraint c1(x,y) = 1-x^2-y^2 = 0");
eq_constraint = NonlinearConstraint(constraint_2, constraint_lower_bound, constraint_upper_bound, 
                                    jac=grad_constraint_2, hess=hess_v_constraint_2);
res = minimize(obj_func, x0, method='trust-constr', jac=grad_obj_func, hess=hess_obj_func, 
                constraints=eq_constraint, options={'gtol': 1e-8,'xtol':1e-12, 'maxiter': 250, 'verbose': 1});
print("Optimal solution = ",res.x);






