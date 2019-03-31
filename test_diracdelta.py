from sympy import DiracDelta
from scipy import integrate

def f(x):
     return x*DiracDelta(x-1)

b, err = integrate.quad(f, 0, 5)    
print b
