import numpy as np
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
from objective_function import *
from constraints import *


# Equality constraints are specified by using the same lower and upper bounds
constraint_lower_bound = 0.0;
constraint_upper_bound = 0.0;

#nonlinear_constraint = NonlinearConstraint(constraint_1, constraint_lower_bound, constraint_upper_bound, 
#                       jac=grad_constraint_1, hess=hess_v_constraint_1);
nonlinear_constraint = ( NonlinearConstraint(constraint_2, constraint_lower_bound, constraint_upper_bound, 
                         jac=grad_constraint_2, hess=hess_v_constraint_2) );

x0 = np.array([0.0, 0.1]); #initial guess 
res = minimize(obj_func, x0, method='trust-constr', jac=grad_obj_func, hess=hess_obj_func, 
                constraints=nonlinear_constraint, options={'gtol': 1e-8,'xtol':1e-12, 'maxiter': 250, 'verbose': 3});
print("Optimal solution = ",res.x);





