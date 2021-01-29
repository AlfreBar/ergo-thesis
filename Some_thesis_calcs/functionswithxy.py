import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigsh

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

def doApplyHamClosed(psiIn):
    return doApplyHam(psiIn, hloc, Nsites, usePBC)
    
if __name__ == '__main__':

    # Simulation parameters
    model = 'XX'  # select 'XX' model of 'ising' model
    Nsites = 18  # number of lattice sites
    usePBC = True  # use periodic or open boundaries
    numval = 1  # number of eigenstates to compute

    # Define Hamiltonian (quantum XX model)
    d = 2  # local dimension
    sX = np.array([[0, 1.0], [1.0, 0]])
    sY = np.array([[0, -1.0j], [1.0j, 0]])
    sZ = np.array([[1.0, 0], [0, -1.0]])
    sI = np.array([[1.0, 0], [0, 1.0]])
    if model == 'XX':
        hloc = (np.real(np.kron(sX, sX) + np.kron(sY, sY))).reshape(2, 2, 2, 2)
        EnExact = -4 / np.sin(np.pi / Nsites)  # Note: only for PBC
    elif model == 'ising':
        hloc = (-np.kron(sX, sX) + 0.5 * np.kron(sZ, sI) + 0.5 * np.kron(sI, sZ)
                ).reshape(2, 2, 2, 2)
        EnExact = -2 / np.sin(np.pi / (2 * Nsites))  # Note: only for PBC


    # cast the Hamiltonian 'H' as a linear operator


    from timeit import default_timer as timer

    H = LinearOperator((2**Nsites, 2**Nsites), matvec=doApplyHamClosed)

    # do the exact diag
    start_time = timer()
    Energy, psi = eigsh(H, k=numval, which='SA')
    diag_time = timer() - start_time

    # check with exact energy
    EnErr = Energy[0] - EnExact  # should equal to zero

    print('NumSites: %d, Time: %1.2f, Energy: %e, EnErr: %e' %
        (Nsites, diag_time, Energy[0], EnErr))

