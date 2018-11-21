import matplotlib.pyplot as plt

from initialConditions import *
from advectionSchemes import *
from diagnostics import *

### The main code is inside a function to avoid global variables    ###
def main():
    "Advect the initial conditions using various advection schemes and"
    "compare results"

    # Parameters
    xmin = 0
    xmax = 1
    nx = 40
    nt = 40
    c = 0.2

    # Derived parameters
    dx = (xmax - xmin)/nx

    # Spatial points for plotting and for defining initial conditions
    x = np.arange(xmin, xmax, dx)

    # Initial conditions
    phiOld = cosBell(x, 0, 0.75)
    phiOldSquare = squareWave(x, 0, 0.75)
    # Exact solution is the initial condition shifted around the domain
    phiAnalytic = cosBell((x - c*nt*dx)%(xmax - xmin), 0, 0.75)
    phiAnalyticSquare = squareWave((x - c*nt*dx)%(xmax - xmin), 0, 0.75)
    # Exact solution
    phiExact = np.zeros((nt,nx))
    for t in range(nt):
        phiExact[t] = cosBell((x - c*t*dx)%(xmax - xmin), 0, 0.75)
    phiExactSquare = np.zeros((nt,nx))
    for t in range(nt):
        phiExactSquare[t] = squareWave((x - c*t*dx)%(xmax - xmin), 0, 0.75)

    # Advect the profile using finite difference for all the time steps for inital bell curve
    phiFTBS, errorFTBS = FTBS(phiExact, phiOld.copy(), c, nt)
    phiBTCS, errorBTCS = BTCS(phiExact, phiOld.copy(), c, nt)
    phiCTCS, errorCTCS = CTCS(phiExact, phiOld.copy(), c, nt)
    phiSL, errorSL = SL(phiExact, phiOld.copy(), c, nt)

    # Advect the profile using finite difference for all the time steps for sqare wave initial
    phiFTBSSquare, errorFTBSSquare = FTBS(phiExactSquare, phiOldSquare.copy(), c, nt)
    phiBTCSSquare, errorBTCSSquare = BTCS(phiExactSquare, phiOldSquare.copy(), c, nt)
    phiCTCSSquare, errorCTCSSquare = CTCS(phiExactSquare, phiOldSquare.copy(), c, nt)
    phiSLSquare, errorSLSquare = SL(phiExactSquare, phiOldSquare.copy(), c, nt)

    # Calculate the error at the last timestep for different Courant numbers
    phiFTBScourant = np.zeros(41)
    phiBTCScourant = np.zeros(41)
    phiCTCScourant = np.zeros(41)
    phiSLcourant = np.zeros(41)
    courantnumber = np.linspace(-2, 2, 41)
    for i in range(41):
        cphiFTBS, cerrorFTBS = FTBS(phiExact, phiOld.copy(), courantnumber[i], nt)
        cphiBTCS, cerrorBTCS = BTCS(phiExact, phiOld.copy(), courantnumber[i], nt)
        cphiCTCS, cerrorCTCS = CTCS(phiExact, phiOld.copy(), courantnumber[i], nt)
        cphiSL, cerrorSL = SL(phiExact, phiOld.copy(), courantnumber[i], nt)

        phiFTBScourant[i]=cerrorFTBS[(nt-2)]
        phiBTCScourant[i]=cerrorBTCS[(nt-2)]
        phiCTCScourant[i]=cerrorCTCS[(nt-2)]
        phiSLcourant[i]=cerrorSL[(nt-2)]

    # Calculate and print out error norms
    print("FTCS l2 error norm = ", l2ErrorNorm(phiFTBS, phiAnalytic))
    print("FTCS linf error norm = ", lInfErrorNorm(phiFTBS, phiAnalytic))

    # Plot the solutions, to give an advection plot for the square wave
    font = {'size'   : 20}
    plt.rc('font', **font)
    plt.figure(1)
    plt.clf()
    plt.ion()
    plt.plot(x, phiOldSquare, label='Initial', color='black')
    plt.plot(x, phiAnalyticSquare, label='Analytic', color='black',
             linestyle='--', linewidth=2)
    #plt.plot(x, phiFTBSSquare, label='FTBS', color='blue')
    #plt.plot(x, phiCTCSSquare, label='CTCS', color='red')
    plt.plot(x, phiBTCSSquare, label='BTCS', color='orange')
    plt.plot(x, phiSLSquare, label='SL', color='green')
    plt.axhline(0, linestyle=':', color='black')
    plt.ylim([-0.2,1.2])
    plt.legend()
    #plt.legend(bbox_to_anchor=(0.6 , 0.6))
    plt.xlabel('$x$')
    input('press return to save file and continue')
    plt.savefig('BTCSSLSquare.pdf')

    # Plot the solutions, to give an advection plot for the Bell curve
    plt.figure(2)
    plt.plot(x, phiOld, label='Initial', color='black')
    plt.plot(x, phiAnalytic, label='Analytic', color='black',
             linestyle='--', linewidth=2)
    plt.plot(x, phiFTBS, label='FTBS', color='blue')
    plt.plot(x, phiCTCS, label='CTCS', color='red')
    plt.plot(x, phiBTCS, label='BTCS', color='orange')
    plt.plot(x, phiSL, label='SL', color='green')
    plt.axhline(0, linestyle=':', color='black')
    plt.ylim([-0.2,1.2])
    plt.legend()
    #plt.legend(bbox_to_anchor=(0.6 , 0.6))
    plt.xlabel('$x$')
    plt.savefig('FTBSCTCSBTCSSL.pdf')

    # Plot the errors against the time steps for initial square wave
    plt.figure(3)
    #plt.plot(errorFTBSSquare, label='error of FTBS', color='blue')
    #plt.plot(errorCTCSSquare, label='error of CTCS', color='red')
    plt.plot(errorBTCSSquare, label='error of BTCS', color='orange')
    plt.plot(errorSLSquare, label='error of SL', color='green')
    plt.legend()
    plt.xlabel('$t$')
    plt.ylabel('error')
    plt.savefig('errorBTCSSLSquare.pdf')    #Plot the errors against the time steps

    # Plot the errors against the time steps for initial Bell curve
    plt.figure(4)
    plt.plot(errorFTBS, label='error of FTBS', color='blue')
    plt.plot(errorCTCS, label='error of CTCS', color='red')
    #plt.plot(errorBTCSSquare, label='error of BTCS', color='orange')
    #plt.plot(errorSLSquare, label='error of SL', color='green')
    plt.legend()
    plt.xlabel('$t$')
    plt.ylabel('error')
    plt.savefig('errorFTBSCTCS.pdf')

    # Plot the errors agains the Courant number
    plt.figure(5)
    plt.plot(courantnumber, phiBTCScourant, label='error of BTCS', color='orange')
    plt.plot(courantnumber, phiSLcourant, label='error of SL', color='green')
    plt.legend()
    plt.xlabel('$c$')
    plt.ylabel('error')
    plt.xlim([-2,2])
    plt.savefig('couranterrorBTCSSL.pdf')

    # Plot the log of errors against the Courant numbers
    plt.figure(6)
    plt.plot(courantnumber, phiFTBScourant, label='error of FTBS', color='blue')
    plt.plot(courantnumber, phiCTCScourant, label='error of CTCS', color='red')
    plt.legend()
    plt.yscale("log")
    plt.xlabel('$c$')
    plt.ylabel('error')
    plt.xlim([-2,2])
    plt.savefig('couranterrorFTBSCTCS.pdf')

main()
