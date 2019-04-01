from __future__ import print_function
import numpy as np
from scipy.constants import hbar
import matplotlib.pyplot as plt


def numerov(xgrid, y_0, y_1, k, S):
    """
    Numerov algorithm
   
    Solve 
   
        d^2
       ------y(x) + k(x)^2 * y(x) = S(x)
        dx^2
   
    by approximating the 2nd derivative 
    by the 3-point formula,
   
        yn+1 - 2yn + yn-1            h^2
       ------------------- = y''n + -----y''''n + O(h^4) .
               h^2                   12
   
    By some rearrangement,
   
             h^2                      5h^2               h^2
       (1 + -----kn+1^2)yn+1 - 2(1 - ------kn^2)yn + (1 + -----kn-1^2)yn-1
             12                        12                12
   
         h^2
      = -----(Sn+1 + 10*Sn + Sn-1) + O(h^6)
         12
   
   
    *** One should know the the two initial points, k and S terms.
   
   
    - Case 1: Two initial values are given.
   
        y(0) = y0, y(1) = y1 --> y(2) is directly calculated.
   
   
    - Case 2: One initial value and its gradient is given.
   
        y(0) = y0, y'(0) = y'0 --> y(1) should be estimated by some formula.
   
        Then one can follow "Case 1".


    Inputs
    ------
    - xgrid : x (1-D array)
    - y_0   : y, initial condition 1
    - y_1   : y, initial condition 2
    - k     : oscillartory function (1-D array, len(k)==len(xgrid))
    - S     : driving term          (1-D array, len(S)==len(xgrid))


    Output
    ------
    - y : solution of the differential equation (1-D array)

    """

    # initialize y
    ngrid = len(xgrid)
    h = xgrid[1] - xgrid[0]
    y = np.zeros(ngrid)
    y[0] = y_0
    y[1] = y_1

    # main loop: evaluate y[j]
    for j in np.arange(2, ngrid):

        y1 = y[j-2]; y2 = y[j-1]
        k1 = k[j-2]; k2 = k[j-1]; k3 = k[j]
        s1 = S[j-2]; s2 = S[j-1]; s3 = S[j] 

        term_S = 1/12. * h**2 * (s3 + 10*s2 + s1)
        term_3 =      (1 + 1/12. *   h**2 * k3**2)
        term_2 = -2 * (1 - 5/12. * 5*h**2 * k2**2) * y2
        term_1 =      (1 + 1/12. *   h**2 * k1**2) * y1

        y3 = (term_S - term_2 - term_1) / term_3
        y[j] = y3

    return y


def numerov_inward(xgrid, y_0, y_1, k, S):

    # initialize y
    ngrid = len(xgrid)
    h = np.abs(xgrid[1] - xgrid[0])
    y = np.zeros(ngrid)
    y[-1] = y_0
    y[-2] = y_1

    # main loop: evaluate y[j]
    for j in np.arange(2, ngrid):
        #print (j, 1-j, 0-j, -1-j)

        y2 = y[0-j]; y3 = y[1-j]
        k1 = k[-1-j]; k2 = k[0-j]; k3 = k[1-j]
        s1 = S[-1-j]; s2 = S[0-j]; s3 = S[1-j] 

        term_S = 1/12. * h**2 * (s3 + 10*s2 + s1)
        term_3 =      (1 + 1/12. *   h**2 * k3**2) * y3
        term_2 = -2 * (1 - 5/12. * 5*h**2 * k2**2) * y2
        term_1 =      (1 + 1/12. *   h**2 * k1**2)

        y1 = (term_S - term_2 - term_3) / term_1
        y[-1-j] = y1

    return y


def greensfunction(xgrid, y_0, y_1, k, broad=1.e-5):
    """
    GF solution

    The solution of linear problem as 

                      |inf.
        phi(x) = intg |     G(x,x'),S(x')dx' 
                      |0

    that satisfying

       -                 -
       |  d^2            |                
       | ------ + k(x)^2 | * G(x,x') = delta(x-x') ,
       |  dx^2           |
       -                 -
   
    similar to
   
        d^2
       ------y(x) + k(x)^2 * y(x) = S(x) .
        dx^2

    Two solutions are phi_< and phi_>, 

    satisfying booundary conditions at x=0 and x=inf., respectively

    and normalized so that their Wronskian

             d phi_>             d phi_<
        W = --------- phi_<  -  --------- phi_> = 1 .
             d x                 d x

    Then tha Green's function is given by

        G(x,x') = phi_<(x_<) * phi_>(x_>)

    """

    x_ = 0.0
    S = delta(xgrid, x_, broad)
    y = numerov(xgrid, y_0, y_1, k, S)
    return y


def phi_gt(xgrid, l):
    return xgrid**(l+1)

def phi_lt(xgrid, l):
    return (-1./(2*l+1))*(xgrid**-l)

def delta(x, x_, broad): 
    A = broad/(2*np.pi)
    B = (xgrid-x_)**2 + (broad/2)**2
    return A/B

def rho(xgrid):
    return (1./(8.*np.pi))*np.exp(-xgrid)

def y_exact(xgrid):
    return 1 - 0.5*(xgrid+2)*np.exp(-xgrid)


# domain
xgrid = np.linspace(0., 100., 1001)
ngrid = len(xgrid)

# initial values
#y_0 = 0.; y_1 = y_exact(xgrid)[1]; l=0
#k = -l*(l+1) * 1./(xgrid)**2
#S = np.zeros(ngrid); S += -4*np.pi * xgrid * rho(xgrid)
#y1 = numerov(xgrid, y_0, y_1, k, S)
#y2 = greensfunction(xgrid, y_0, y_1, k, S)


# Figure
fig = plt.figure(figsize=(10,3))
fig1 = fig.add_subplot(111)
#fig1.plot(xgrid[1:], y1[1:]/xgrid[1:], 'k--', label='Exact')
#fig1.plot(xgrid[1:], y2[1:]/xgrid[1:], 'k--', label='Exact')

#fig1.plot(xgrid[501:],  phi_gt(xgrid, 0)[501:],  label='phi_gt')
#fig1.plot(xgrid[1:501], phi_lt(xgrid, 0)[1:501], label='phi_lt')
fig1.plot(xgrid[501:],  phi_gt(xgrid, 0)[501:] /xgrid[501:],  label='PHI_gt')
fig1.plot(xgrid[1:501], phi_lt(xgrid, 0)[1:501]/xgrid[1:501], label='PHI_lt')
fig1.legend()
plt.show()
