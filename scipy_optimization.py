import numpy as np
from scipy import optimize
from scipy.optimize import minimize

x0 = np.array([0.5, 0]);
res = minimize(rosen, x0, method='SLSQP', jac=rosen_der,
               constraints=[eq_cons, ineq_cons], options={'ftol': 1e-9, 'disp': True},
               bounds=bounds);
print(res.x);




