import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigsh

####Local Hamiltonian Procedure
def doApplyHam(psiIn: np.ndarray,
               hloc: np.ndarray,
               N: int,
               usePBC: bool):
    """
    Applies local Hamiltonian, given as sum of nearest neighbor terms, to
    an input quantum state.
    Args:
    psiIn: vector of length d**N describing the quantum state.
    hloc: array of ndim=4 describing the nearest neighbor coupling.
    N: the number of lattice sites.
    usePBC: sets whether to include periodic boundary term.
    Returns:
    np.ndarray: state psi after application of the Hamiltonian.
    """
    d = hloc.shape[0]
    psiOut = np.zeros(psiIn.size)
    for k in range(N - 1):
        # apply local Hamiltonian terms to sites [k,k+1]
        psiOut += np.tensordot(hloc.reshape(d**2, d**2),
                               psiIn.reshape(d**k, d**2, d**(N - 2 - k)),
                               axes=[[1], [1]]).transpose(1, 0, 2).reshape(d**N)

    if usePBC:
        # apply periodic term
        psiOut += np.tensordot(hloc.reshape(d, d, d, d),
                               psiIn.reshape(d, d**(N - 2), d),
                               axes=[[2, 3], [2, 0]]
                               ).transpose(1, 2, 0).reshape(d**N)

    return psiOut


def diagonalize(numval,Nsites,hloc,usePBC):
    
    def doApplyHamClosed(psiIn):
        return doApplyHam(psiIn, hloc, Nsites, usePBC)
    
    H = LinearOperator((2**Nsites,2**Nsites), matvec=doApplyHamClosed)
    Energy, psi = eigsh(H,k=numval,which='SA')
    
    return Energy, psi

def diagonalize_2(numval,Nsites,hloc,usePBC,vtr):
    
    def doApplyHamClosed(psiIn):
        return doApplyHam(psiIn, hloc, Nsites, usePBC)
    
    H = LinearOperator((2**Nsites,2**Nsites), matvec=doApplyHamClosed)
    Energy, psi = eigsh(H,k=numval,which='SA',v0=vtr)
    
    return Energy, psi
#### ising model
sX = np.array([[0, 1], [1, 0]])
sY = np.array([[0, -1j], [1j, 0]])
sZ = np.array([[1, 0], [0,-1]])
sI = np.array([[1, 0], [0, 1]])
def isingmodel(lam,h=0.00):
    return (np.real(-lam*np.kron(sX,sX) - np.kron(sZ,sI)- np.kron(sI,sZ)+h*np.kron(sX,sI))).reshape(2,2,2,2)
def isingmodel_zz_x(lam,h=0.00):
    return (np.real(-np.kron(sZ,sZ) - (lam/2)*np.kron(sX,sI)-(lam/2)*np.kron(sI,sX)+h*np.kron(sZ,sI))).reshape(2,2,2,2)
def isingmodel_zz_x_2(lam,h=0.00):
    return (np.real(-np.kron(sZ,sZ) - (lam/2)*np.kron(sX,sI)-(lam/2)*np.kron(sI,sX)+h*np.kron(sZ,sI))).reshape(2,2,2,2)
def isingmodel_rev(lam,h=0.00):
    return (np.real(-np.kron(sX,sX) - (lam/2)*np.kron(sZ,sI)- (lam/2)*np.kron(sI,sZ)+h*np.kron(sX,sI))).reshape(2,2,2,2)
def isingmodel_rev_frac2(lam,h=0.00):
    return (1/2)*(np.real(-np.kron(sX,sX) - (lam/2)*np.kron(sZ,sI)- (lam/2)*np.kron(sI,sZ) + h*np.kron(sX,sI))).reshape(2,2,2,2)
def isingmodel_rev_frac2_2(lam,h=0.00):
    return (1/2)*(np.real(-np.kron(sX,sX) - (lam)*np.kron(sZ,sI)- (lam)*np.kron(sI,sZ) + h*np.kron(sX,sI))).reshape(2,2,2,2)
