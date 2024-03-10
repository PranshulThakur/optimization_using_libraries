import numpy as np
from scipy.optimize import minimize
from objective_function import *
from constraints import *


# Equality constraints are specified by using the same lower and upper bounds
constraint_lower_bound = 0.0;
constraint_upper_bound = 0.0;
x0 = np.array([0.0, 0.1]); #initial guess 

print("===================================================================");
print("                 Optimization using SLSQP                          ");
print("===================================================================");
print("Running optimization for f(x,y) = (1-x)^2 + 100*(y-x^2)^2");
print("With constraint c1(x,y) = 1-x-y =0");
eq_constraint = {'type':'eq', 'fun': constraint_1, 'jac' : grad_constraint_1};
res = minimize(obj_func, x0, method='SLSQP', jac=grad_obj_func, 
                constraints=eq_constraint, options={'ftol': 1e-9, 'maxiter': 250,'disp':True});
print("Optimal solution = ",res.x);

print("===================================================================");


print("Running optimization for f(x,y) = (1-x)^2 + 100*(y-x^2)^2");
print("With constraint c2(x,y) = 1-x^2-y^2 =0");
eq_constraint = {'type':'eq', 'fun': constraint_2, 'jac': grad_constraint_2};
res = minimize(obj_func, x0, method='SLSQP', jac=grad_obj_func, 
                constraints=eq_constraint, options={'ftol': 1e-9, 'maxiter': 250,'disp':True});
print("Optimal solution = ",res.x);
print("===================================================================");



