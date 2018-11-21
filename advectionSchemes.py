# The numpy package for numerical functions and pi\
import numpy as np
from math import floor
from initialConditions import *
from diagnostics import *

def FTCS(phiOld, c, nt):
    "Linear advection of profile in phiOld using FTCS, Courant number c"
    "for nt time-steps"

    nx = len(phiOld)

    # new time-step array for phi
    phi = phiOld.copy()

    # FTCS for each time-step
    for it in range(nt):
        # Loop through all space using remainder after division (%)
        # to cope with periodic boundary conditions
        for j in range(nx):
            phi[j] = phiOld[j] - 0.5*c*\
                     (phiOld[(j+1)%nx] - phiOld[(j-1)%nx])

        # update arrays for next time-step
        phiOld = phi.copy()

    return phi

def FTBS(phiExact, phiOld, c, nt):
    "Linear advection of profile in phiOld using FTBS, Courant number c"
    "for nt time-steps"

    nx = len(phiOld)
    error = np.zeros(nt, dtype=np.float)
    # new time-step array for phi
    phi = phiOld.copy()

    # FTBS for each time-step
    for it in range(nt):
        # Loop through all space using remainder after division (%)
        # to cope with periodic boundary conditions
        for j in range(nx):
            phi[j] = phiOld[j] - c*\
                     (phiOld[j] - phiOld[j-1])
        # calculate l2 error between phi at time it and the exact solution
        error[it]=l2ErrorNorm(phi, phiExact[it])
        # update arrays for next time-step
        phiOld = phi.copy()

    return phi, error

def CTCS(phiExact, phiOld, c, nt):
    "Linear advection of profile in phiOld using CTCS, Courant number c"
    "for nt time-steps"

    nx = len(phiOld)
    error = np.zeros(nt, dtype=np.float)
    # new time-step array for phi
    prevphiOld = phiOld.copy()
    phiOld, ftbserror = FTBS(phiExact, phiOld, c, 1)
    phi = phiOld.copy()
    error[0]=ftbserror[0]

    # CTCS for each time-step
    for it in range(1, nt):
        # Loop through all space using remainder after division (%)
        # to cope with periodic boundary conditions
        for j in range(nx):
            phi[j] = prevphiOld[j] - c*\
                     (phiOld[(j+1)%nx] - phiOld[(j-1)%nx])
        # calculate l2 error between phi at time it and the exact solution
        error[it]=l2ErrorNorm(phi, phiExact[it])
        # update arrays for next time-step
        prevphiOld=phiOld.copy()
        phiOld = phi.copy()

    return phi, error

def BTCS(phiExact, phiOld, c, nt):
    "Linear advection of profile in phiOld using BTCS, Courant number c"
    "for nt time-steps"

    nx = len(phiOld)
    error = np.zeros(nt, dtype=np.float)
    matrix= np.zeros((nx,nx), dtype=np.float) #initialise matrix as 0
    for j in range(nx): #define matrix such that Matrix*phi(n+1)=phi(n)
        matrix[j][j]=1
        matrix[(j+1)%(nx)][j]=-c/2
        matrix[(j-1)%(nx)][j]=c/2
    # new time-step array for phi
    phi = phiOld.copy()

    # BTCS for each time-step
    for j in range(nt):
        phi = np.linalg.solve(matrix, phiOld) #this would be the same as phi = inversematrix@phiOld, but is less expensive than inverting matrix M
        # calculate l2 error between phi at time j and the exact solution
        error[j]=l2ErrorNorm(phi, phiExact[j])
        # update arrays for next time-step
        phiOld = phi.copy()

    return phi, error

def CLI(beta, phi, k): #cubic lagrangian interpolation where f(x)=phi[x]
    nx= len(phi)
    phiNewj = -1/6*beta*(1-beta)*(2-beta)*phi[(k-1)%nx]+1/2*(1+beta)*(1-beta)*(2-beta)*phi[k%nx]+\
        1/2*(1+beta)*beta*(2-beta)*phi[(k+1)%nx]-1/6*(1+beta)*beta*(1-beta)*phi[(k+2)%nx]
    return phiNewj

def SL(phiExact, phiOld, c, nt):
    "Linear advection of profile in phiOld using Semi-Lagrangian, courant number c"
    "for nt time-steps"

    nx = len(phiOld)
    error = np.zeros(nt, dtype=np.float)
    phi= phiOld.copy()

    # Semi Legrangian for each time-step
    for t in range(nt):
        for j in range(nx):
            k = floor(j-c) %nx
            beta = (j-c-k)%nx
            phi[j] = CLI(beta,phiOld,k) #Lagrangian interpolation using x_(k-1),x_k,x_(k+1),x_(k+2)
        # calculate l2 error between phi at time t and the exact solution
        error[t]=l2ErrorNorm(phi, phiExact[t])
        phiOld=phi.copy()
    return phi, error
