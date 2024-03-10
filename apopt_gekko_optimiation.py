import numpy as np
from objective_function import *
from constraints import *
from gekko import GEKKO
m = GEKKO(remote=False); # Initialize gekko
m.options.SOLVER=1; # Uses SQP from APOPT
m.options.OTOL = 1.0e-9;
m.options.RTOL = 1.0e-9;

# Note: Gradients and Hessians are computed using automatic differentiation.
print("===================================================================");
print("      Optimization with APOPT using GEKKO as the interface         ");
print("===================================================================");
print("Running optimization for f(x,y) = (1-x)^2 + 100*(y-x^2)^2");
print("With constraint c1(x,y) = 1-x-y = 0");
x = m.Array(m.Var,2);
m.Minimize(obj_func(x));
m.Equation(constraint_1(x)==0.0); 
m.solve();
print("Optimal solution = ",x);
print("===================================================================");

print("===================================================================");
print("Running optimization for f(x,y) = (1-x)^2 + 100*(y-x^2)^2");
print("With constraint c1(x,y) = 1-x^2-y^2 = 0");
x = m.Array(m.Var,2);
m.Minimize(obj_func(x));
m.Equation(constraint_2(x)==0.0); 
m.solve();
print("Optimal solution = ",x);
print("===================================================================");

