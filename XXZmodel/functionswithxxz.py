import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigsh
import qutip as q

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
sX = np.array([[0, 0.5], [0.5, 0]])
sY = np.array([[0, -0.5*j], [0.5*j, 0]])
sZ = np.array([[0.5, 0], [0,-0.5]])
sI = np.array([[0.5, 0], [0, 0.5]])
def isingmodel(lam,h=0.00):
    return (np.real(-lam*np.kron(sX,sX) - np.kron(sZ,sI)- np.kron(sI,sZ)+h*np.kron(sX,sI))).reshape(2,2,2,2)

def xymodel(lam,gamma=1,h=0.00):
    return (1/2)*(np.real(-((1+gamma)/2)*np.kron(sX,sX) - ((1-gamma)/2)*np.kron(sY,sY)- (lam/2)*np.kron(sZ,sI)- (lam/2)*np.kron(sI,sZ) + h*np.kron(sX,sI))).reshape(2,2,2,2)

def xxzmodel(lam,Jx=1,Jy=1,Jz=1,h=0.00):
    return -(np.real(Jx*np.kron(sX,sX) + Jy*np.kron(sY,sY) + Jz*np.kron(sZ,sZ) + (lam)*np.kron(sZ,sI) + (lam)*np.kron(sI,sZ) + h*np.kron(sX,sI))).reshape(2,2,2,2)


def construct_ham(N, h):

    si = q.qeye(2)
    sx = q.sigmax()
    sy = q.sigmay()
    sz = q.sigmaz()

    sx_list = []
    sy_list = []
    sz_list = []

    for n in range(N):
        op_list = []
        for m in range(N):
            op_list.append(si)

        op_list[n] = sx
        sx_list.append(q.tensor(op_list))

        op_list[n] = sy
        sy_list.append(q.tensor(op_list))

        op_list[n] = sz
        sz_list.append(q.tensor(op_list))

    # construct the hamiltonian
    H = 0

    # energy splitting terms
    for n in range(N):
        H += - 0.5 * h * sz_list[n]

    # interaction terms
    for n in range(N-1):
        H += - 0.5 * sx_list[n] * sx_list[n+1]


    return H

def construct_distant_ham_2sites(N, h):

    si = q.qeye(2)
    sx = q.sigmax()
    sy = q.sigmay()
    sz = q.sigmaz()

    sx_list = []
    sy_list = []
    sz_list = []

    for n in range(N):
        op_list = []
        for m in range(N):
            op_list.append(si)

        op_list[n] = sx
        sx_list.append(q.tensor(op_list))

        op_list[n] = sy
        sy_list.append(q.tensor(op_list))

        op_list[n] = sz
        sz_list.append(q.tensor(op_list))

    # construct the hamiltonian
    H = 0

    
    H += - 0.5 * h * sz_list[0]
    H += - 0.5 * h * sz_list[-1]
 


    return H

def construct_ham_xxz(N, h):

    si = q.qeye(2)
    sx = q.sigmax()
    sy = q.sigmay()
    sz = q.sigmaz()

    sx_list = []
    sy_list = []
    sz_list = []

    for n in range(N):
        op_list = []
        for m in range(N):
            op_list.append(si)

        op_list[n] = sx
        sx_list.append(q.tensor(op_list))

        op_list[n] = sy
        sy_list.append(q.tensor(op_list))

        op_list[n] = sz
        sz_list.append(q.tensor(op_list))

    # construct the hamiltonian
    H = 0

    # energy splitting terms


    # interaction terms
    for n in range(N-1):
        H += - 0.25 * sx_list[n] * sx_list[n+1]
        
    for n in range(N-1):
        H += - 0.25 * sy_list[n] * sy_list[n+1]
        
    for n in range(N-1):
        H += - 0.25* h * sz_list[n] * sz_list[n+1]
    return H
