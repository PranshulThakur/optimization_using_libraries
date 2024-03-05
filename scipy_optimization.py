import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds

bounds = Bounds([0, -0.5], [1.0, 2.0])
def rosen(x):
    """The Rosenbrock function"""
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
def rosen_der(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
    der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    der[-1] = 200*(x[-1]-x[-2]**2)
    return der
def rosen_hess_p(x, p):
    x = np.asarray(x)
    Hp = np.zeros_like(x)
    Hp[0] = (1200*x[0]**2 - 400*x[1] + 2)*p[0] - 400*x[0]*p[1]
    Hp[1:-1] = -400*x[:-2]*p[:-2]+(202+1200*x[1:-1]**2-400*x[2:])*p[1:-1] \
               -400*x[1:-1]*p[2:]
    Hp[-1] = -400*x[-2]*p[-2] + 200*p[-1]
    return Hp
def rosen_hess(x):
    x = np.asarray(x)
    H = np.diag(-400*x[:-1],1) - np.diag(400*x[:-1],-1)
    diagonal = np.zeros_like(x)
    diagonal[0] = 1200*x[0]**2-400*x[1]+2
    diagonal[-1] = 200
    diagonal[1:-1] = 202 + 1200*x[1:-1]**2 - 400*x[2:]
    H = H + np.diag(diagonal)
    return H
ineq_cons = {'type': 'ineq',
             'fun' : lambda x: np.array([1 - x[0] - 2*x[1],
                                         1 - x[0]**2 - x[1],
                                         1 - x[0]**2 + x[1]]),
             'jac' : lambda x: np.array([[-1.0, -2.0],
                                         [-2*x[0], -1.0],
                                         [-2*x[0], 1.0]])}
eq_cons = {'type': 'eq',
           'fun' : lambda x: np.array([2*x[0] + x[1] - 1]),
           'jac' : lambda x: np.array([2.0, 1.0])}


def f(x):
    f_val = 0.0;
    f_val = (1.0-x[0])**2 + 100.0*(x[1] - x[0]**2)**2;
    return f_val;

def df(x):
    # df_val = np.zeros_like(x);
    df_val = np.array([0.0,0.0]);
    df_val[0] = -2*(1-x[0]) - 400*x[0]*(x[1] - x[0]**2);
    df_val[1] = 200*(x[1] - x[0]**2);
    return df_val;

def d2f(x):
    d2f_val = np.array( [[0.0,0.0], 
                         [0.0,0.0]] );
    d2f_val[0][0] = 2.0 + 800.0*x[0];
    d2f_val[0][1] = -400.0*x[0];
    d2f_val[1][0] = -400.0*x[0];
    d2f_val[1][1] = 200;
    return d2f_val;


x0 = np.array([0.5, 0.0]);
res = minimize(f, x0, method='SLSQP', jac=df, hess=d2f,
               constraints=[eq_cons, ineq_cons], options={'ftol': 1e-9, 'disp': True},
               bounds=bounds);
print(res.x);





