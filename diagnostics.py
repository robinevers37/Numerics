# Various function for plotting results and for calculating error measures
import numpy as np


def l2ErrorNorm(phi, phiExact):
    "Calculates the l2 error norm (RMS error) of phi in comparison to"
    "phiExact"

    # calculate the error and the RMS error norm
    phiError = phi - phiExact
    l2 = np.sqrt(sum(phiError**2)/sum(phiExact**2))

    return l2


def lInfErrorNorm(phi, phiExact):
    "Calculates the linf error norm (maximum normalised error) in comparison"
    "to phiExact"
    phiError = phi - phiExact
    return np.max(np.abs(phiError))/np.max(np.abs(phiExact))
