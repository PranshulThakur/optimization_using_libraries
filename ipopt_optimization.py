import numpy as np
from objective_function import *
from constraints import *
import ipyopt

# Wrapper functions according to ipopt's syntax.
def grad_obj_func_ipopt(x,out):
    out[()] = grad_obj_func(x);
    return out;

def constraint_1_ipopt(x,out):
    out[()]= constraint_1(x);
    return out;

def grad_constraint_1_ipopt(x,out):
    out[()] = grad_constraint_1(x);
    return out;

def constraint_2_ipopt(x,out):
    out[()] = constraint_2(x);
    return out;

def grad_constraint_2_ipopt(x,out):
    out[()] = grad_constraint_2(x);
    return out;

def lagrangian_hess_constraint_1(x, lagrange_mult, obj_factor, out): # IPOPT requires Hessian of the lagrangian
    out1 = hess_obj_func(x);
    out2 = hess_v_constraint_1(x,lagrange_mult);
    out[()] = (np.add(out1, out2)).flatten();
    return out;

def lagrangian_hess_constraint_2(x, lagrange_mult, obj_factor, out): 
    out1 = hess_obj_func(x);
    out2 = hess_v_constraint_2(x,lagrange_mult);
    out[()] = (np.add(out1, out2)).flatten();
    return out;

def run_optimizer(constraint, grad_constraint, lagrangian_hess_constraint):
    # Specify non-zero values at (0,0) and (0,1).
    sparsity_indices_gradient = (np.array([0,0]),np.array([0,1]));

    # Specify sparsity pattern of the Hessian. Currently assuming all values are non-zero.
    # For a 2x2 Hessian, non-zero values are at (i,j) = (0,0), (0,1), (1,0) and (1,1)
    # and are specified as (np.array([i]), np.array([j])).
    sparsity_indices_hessian = (np.array([0,0,1,1]),np.array([0,1,0,1]));

    big_number = np.inf;
    x_L = np.array([-big_number,-big_number]);
    x_U = np.array([big_number,big_number]);
    constraint_L = np.array([0.0]); # Equality constrains are specified by the same lower and upper bounds.
    constraint_U = np.array([0.0]); 
    n_design_variables = 2;
    n_constraints = 1;
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
        constraint,
        grad_constraint,
        lagrangian_hess_constraint);
    x0 = np.array([0.0, 0.1]); #initial guess
    nlp.set(tol=1.0e-9);
    x, obj, status = nlp.solve(x0);
    print("Optimal solution = ",x);
    return x;


print("===================================================================");
print("                 Optimization using IPOPT                          ");
print("===================================================================");
print('\n\n');
print("===================================================================");
print("Running optimization for f(x,y) = (1-x)^2 + 100*(y-x^2)^2");
print("With constraint c1(x,y) = 1-x-y = 0");
print("===================================================================");
run_optimizer(constraint_1_ipopt, grad_constraint_1_ipopt, lagrangian_hess_constraint_1);
print("===================================================================");
print('\n\n');
print("===================================================================");
print("Running optimization for f(x,y) = (1-x)^2 + 100*(y-x^2)^2");
print("With constraint c2(x,y) = 1-x^2-y^2 = 0");
print("===================================================================");
run_optimizer(constraint_2_ipopt, grad_constraint_2_ipopt, lagrangian_hess_constraint_2);
print("===================================================================");


