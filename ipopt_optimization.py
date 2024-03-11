import numpy as np
from objective_function import *
from constraints import *
import ipyopt

# Wrapper functions according to ipopt's syntax.
def grad_obj_func_ipopt(x,out):
    out[()] = np.copy(grad_obj_func(x));
    return out;

def constraint_1_ipopt(x,out):
    out[()]= constraint_1(x);
    return out;

def grad_constraint_1_ipopt(x,out):
    out[()] = np.copy(grad_constraint_1(x));
    return out;

def constraint_2_ipopt(x,out):
    out[()] = constraint_2(x);
    return out;

def grad_constraint_2_ipopt(x,out):
    out[()] = np.copy(grad_constraint_2(x));
    return out;

def lagrangian_hess_constraint_1(x, lagrange_mult, obj_factor, out): # IPOPT requires Hessian of the lagrangian
    out1 = np.copy(hess_obj_func(x));
    out2 = np.copy(hess_v_constraint_1(x,lagrange_mult));
    out[()] = (np.add(out1, out2)).flatten();
    return out;

def lagrangian_hess_constraint_2(x, lagrange_mult, obj_factor, out): 
    out1 = np.copy(hess_obj_func(x));
    out2 = np.copy(hess_v_constraint_2(x,lagrange_mult));
    out[()] = (np.add(out1, out2)).flatten();
    return out;

# Specify non-zero values at (0,0) and (0,1).
sparsity_indices_gradient = (np.array([0,0]),np.array([0,1]));

# Specify non-zero values at (0,0), (0,1), (1,0) and (1,1).
sparsity_indices_hessian = (np.array([0,0,1,1]),np.array([0,1,0,1]));

big_number = np.inf;
x_L = np.array([-big_number,-big_number]);
x_U = np.array([big_number,big_number]);
constraint_L = np.array([0.0]); # Equality constrains are specified by the same lower and upper bounds.
constraint_U = np.array([0.0]); 
n_design_variables = 2;
n_constraints = 1;
zl = np.ones(n_design_variables);
zu = np.ones(n_design_variables);
nlp = ipyopt.Problem(
    n_design_variables,
    x_L,
    x_U,
    n_constraints,
    constraint_L,
    constraint_U,
    sparsity_indices_gradient,
    sparsity_indices_hessian,
    obj_func,
    grad_obj_func_ipopt,
    constraint_1_ipopt,
    grad_constraint_1_ipopt,
    lagrangian_hess_constraint_1);
x0 = np.array([0.0, 0.1]); #initial guess
nlp.set(tol=1.0e-9);
x, obj, status = nlp.solve(x0);
print("Optimal solution = ",x);


