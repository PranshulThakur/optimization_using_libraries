import numpy as np

# Functions for constraint 1
def constraint_1(x):
    c_val = 1.0 - x[0] - x[1];
    return c_val;

def grad_constraint_1(x):
    dc = np.array([0.0,0.0]);
    dc[0] = -1.0;
    dc[1] = -1.0;
    return dc;

def hess_v_constraint_1(x,v):  # computes (lagrange_multiplier^T Hessian)
    hess_v = np.array([ [0.0,0.0],
                        [0.0,0.0] ]);
    return hess_v;


# Functions for constraint 2
def constraint_2(x):
    c_val = 1.0 - x[0]**2 - x[1]**2;
    return c_val;

def grad_constraint_2(x):
    dc = np.array([0.0,0.0]);
    dc[0] = -2.0*x[0];
    dc[1] = -2.0*x[1];
    return dc;

def hess_v_constraint_2(x,v):  # computes (lagrange_multiplier^T Hessian)
    hess_v = np.array([ [0.0,0.0],
                        [0.0,0.0] ]);
    hess_v[0][0] = -2.0*v;
    hess_v[1][1] = -2.0*v;
    return hess_v;

