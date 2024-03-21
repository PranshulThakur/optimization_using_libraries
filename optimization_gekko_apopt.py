import numpy as np
from objective_function import *
from constraints import *
from gekko import GEKKO

def run_optimizer(constraint):
    m = GEKKO(remote=False); # Initialize gekko
    m.options.SOLVER=1; # Uses SQP from APOPT
    m.options.OTOL = 1.0e-9;
    m.options.RTOL = 1.0e-9;
    m.solver_options = ['minlp_as_nlp 1', 'nlp_maximum_iterations 500'];
    # Note: Gradients and Hessians are computed using automatic differentiation.
    x = m.Array(m.Var,2);
    x[0].value = 0.0;
    x[1].value = 0.0;
    m.Minimize(obj_func(x));
    m.Equation(constraint(x)==0.0); 
    m.solve(disp=True);
    print("Optimal solution = ",x);
    return x;

print("===================================================================");
print("      Optimization with APOPT using GEKKO as the interface         ");
print("===================================================================");
print('\n\n');
print("===================================================================");
print("Running optimization for f(x,y) = (1-x)^2 + 100*(y-x^2)^2");
print("With constraint c1(x,y) = 1-x-y = 0");
print("===================================================================");
run_optimizer(constraint_1);
print("===================================================================");
print('\n\n');
print("===================================================================");
print("Running optimization for f(x,y) = (1-x)^2 + 100*(y-x^2)^2");
print("With constraint c2(x,y) = 1-x^2-y^2 = 0");
print("===================================================================");
run_optimizer(constraint_2);
print("===================================================================");
