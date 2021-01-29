import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigsh

####Local Hamiltonian Procedure
def doApplyHam(psiIn, hloc, N, usePBC):
    
    d = hloc.shape[0]
    psiOut = np.zeros(len(psiIn))
    for k in range(1,N):
        # apply local Hamiltonian terms
        psiOut = psiOut + (hloc.reshape(d**2,d**2) @ psiIn.reshape(d**(k-1),d**2,d**(N-k-1)).transpose(1,0,2).reshape(
                d**2,d**(N-2))).reshape(d**2,d**(k-1),d**(N-k-1)).transpose(1,0,2).reshape(d**N)
                                                      
    if usePBC:
        # apply periodic term
        psiOut = psiOut + (hloc.reshape(d**2,d**2) @ psiIn.reshape(d,d**(N-2),d).transpose(2,0,1).reshape(
                d**2,d**(N-2))).reshape(d,d,d**(N-2)).transpose(1,2,0).reshape(d**N)

    return psiOut


def diagonalize(numval,Nsites,hloc,usePBC):
    
    def doApplyHamClosed(psiIn):
        return doApplyHam(psiIn, hloc, Nsites, usePBC)
    
    H = LinearOperator((2**Nsites,2**Nsites), matvec=doApplyHamClosed)
    Energy, psi = eigsh(H,k=numval,which='SA')
    
    return Energy, psi

#### ising model
sX = np.array([[0, 1], [1, 0]])
sY = np.array([[0, -1j], [1j, 0]])
sZ = np.array([[1, 0], [0,-1]])
sI = np.array([[1, 0], [0, 1]])
def isingmodel(lam,h=0.001):
    return (np.real(-lam*np.kron(sX,sX) - np.kron(sZ,sI)+h*np.kron(sX,sI))).reshape(2,2,2,2)

def isingmodel_rev(lam,h=0.00):
    return (np.real(-np.kron(sX,sX) - lam*np.kron(sZ,sI)+h*np.kron(sX,sI))).reshape(2,2,2,2)
def isingmodel_rev_frac2(lam,h=0.00):
    return (1/2)*(np.real(-np.kron(sX,sX) - lam*np.kron(sZ,sI) + h*np.kron(sX,sI))).reshape(2,2,2,2)

