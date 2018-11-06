# The numpy package for numerical functions and pi
import numpy as np
from math import floor

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

def FTBS(phiOld, c, nt):
    "Linear advection of profile in phiOld using FTBS, Courant number c"
    "for nt time-steps"

    nx = len(phiOld)

    # new time-step array for phi
    phi = phiOld.copy()

    # FTBS for each time-step
    for it in range(nt):
        # Loop through all space using remainder after division (%)
        # to cope with periodic boundary conditions
        for j in range(nx):
            phi[j] = phiOld[j] - c*\
                     (phiOld[j] - phiOld[j-1])

        # update arrays for next time-step
        phiOld = phi.copy()

    return phi

def CTCS(phiOld, c, nt):
    "Linear advection of profile in phiOld using CTCS, Courant number c"
    "for nt time-steps"

    nx = len(phiOld)

    # new time-step array for phi
    phi = phiOld.copy()
    prevphiOld = phi.copy()

    # CTCS for each time-step
    for it in range(nt):
        # Loop through all space using remainder after division (%)
        # to cope with periodic boundary conditions
        for j in range(nx):
            phi[j] = prevphiOld[j] - c*\
                     (phiOld[(j+1)%nx] - phiOld[(j-1)%nx])

        # update arrays for next time-step
        prevphiOld=phiOld.copy()
        phiOld = phi.copy()

    return phi

def BTCS(phiOld, c, nt):
    "Linear advection of profile in phiOld using CTCS, Courant number c"
    "for nt time-steps"

    nx = len(phiOld)
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
        # update arrays for next time-step
        phiOld = phi.copy()

    return phi

def CLI(beta, phi, k): #cubic lagrangian interpolation where f(x)=phi[x]
    nx= len(phi)
    phiNewj = -1/6*beta*(1-beta)*(2-beta)*phi[(k-1)%nx]+1/2*(1+beta)*(1-beta)*(2-beta)*phi[k%nx]+\
        1/2*(1+beta)*beta*(2-beta)*phi[(k+1)%nx]-1/6*(1+beta)*beta*(1-beta)*phi[(k+2)%nx]
    return phiNewj

def SL(phiOld, c, nt):
    "Linear advection of profile in puiOld using Semi-Lagrangian, courant number c"
    "for nt time-steps"

    nx = len(phiOld)
    phi= phiOld.copy()
    #Semi Legrangian for each time-step
    for t in range(nt):
        for j in range(nx):
            k = floor(j-c) %nx
            beta = (j-c-k)%nx
            phi[j] = CLI(beta,phiOld,k) #Lagrangian interpolation using x_(k-1),x_k,x_(k+1),x_(k+2)
        phiOld=phi.copy()
    return phi
