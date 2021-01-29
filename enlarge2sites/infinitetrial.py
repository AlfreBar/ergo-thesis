#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.sparse import kron, identity
from scipy.sparse.linalg import eigsh  # Lanczos routine from ARPACK
from matplotlib import pyplot as plt


# Definiamo degli oggetti BLOCK che hanno come attributi: lunghezza della catena di siti nel blocco, dimensione dello spazio di Hilbert del blocco, $H_B$ e $H_{BS}$

# In[2]:


from collections import namedtuple
Block = namedtuple("Block", ["length", "basis_size", "operator_dict"])
EnlargedBlock = namedtuple("EnlargedBlock", ["length", "basis_size", "operator_dict"])


# In[3]:


def is_valid_block(block):
    for op in block.operator_dict.values():
        if op.shape[0] != block.basis_size or op.shape[1] != block.basis_size:
            return False
    return True
is_valid_enlarged_block = is_valid_block


model_d = 2  # single-site basis size


sX = np.array([[0, 1], [1, 0]], dtype='complex') 
sY = np.array([[0, -1j], [1j, 0]], dtype="complex") 
Id = np.array([[1, 0], [0, 1]], dtype='complex')

def H2(Sx1, Sx2, Sy1, Sy2,gamma=1):  # two-site part of H
    
    return     -0.25 * ((1+gamma)*kron(Sx1, Sx2)+(1-gamma)*kron(Sy1, Sy2)) 



h=0.0001


def enlarge_block(block,site,H2):

    dblock = block.basis_size
    b = block.operator_dict
    dsite = site.basis_size
    s=site.operator_dict
    
    enlarged_operator_dict = {
        "H": kron(b["H"], identity(dsite)) + kron(identity(dblock), s["H"]) + H2(b["conn_Sx"], s["conn_Sx"],b["conn_Sy"], s["conn_Sy"]),
        "conn_Sx": kron(identity(dblock), s["conn_Sx"]),
        "conn_Sy": kron(identity(dblock), s["conn_Sy"]),

    }

    return EnlargedBlock(length=(block.length + 1),
                         basis_size=(dblock * model_d),
                         operator_dict=enlarged_operator_dict)


def enlarge_2_block(block,site,H2):

    dblock = block.basis_size
    b = block.operator_dict
    dsite = site.basis_size
    s=site.operator_dict
    
    enlarged_operator_dict = {
        "H": kron(b["H"], identity(dsite)) + kron(identity(dblock), s["H"]) + H2(b["conn_Sx"], s["conn_Sx"],b["conn_Sy"], s["conn_Sy"]),
        "conn_Sx": kron(identity(dblock), s["conn_Sx"]),
        "conn_Sy": kron(identity(dblock), s["conn_Sy"]),

    }

    return EnlargedBlock(length=(block.length + site.length),
                         basis_size=(dblock * dsite),
                         operator_dict=enlarged_operator_dict)

# In[7]:


def rotate_and_truncate(operator, transformation_matrix):
    """Transforms the operator to the new (possibly truncated) basis given by
    `transformation_matrix`.
    """
    return transformation_matrix.conjugate().transpose().dot(operator.dot(transformation_matrix))


# Ora bisogna fare un DMRG step: Creare blocco allargato, connettere, superblocco, trovare lo stato di base e poi costruire matrice densità

# In[8]:


def get_superblock(sys_enl, env_enl, H2):
  
    assert is_valid_block(sys_enl)
    assert is_valid_block(env_enl)

    # Enlarge each block by a single site.

     # Construct the full superblock Hamiltonian.
    m_sys_enl = sys_enl.basis_size
    m_env_enl = env_enl.basis_size
    sys_enl_op = sys_enl.operator_dict
    env_enl_op = env_enl.operator_dict
    superblock_hamiltonian= kron(sys_enl_op["H"], identity(m_env_enl)) + kron(identity(m_sys_enl), env_enl_op["H"]) + \
                             H2(sys_enl_op["conn_Sx"], env_enl_op["conn_Sx"],sys_enl_op["conn_Sy"], env_enl_op["conn_Sy"])

    return superblock_hamiltonian



# In[9]:

# Diagonalizziamo e otteniamo matrice densità

def get_reduced_density_matrix(enl,psi0):
    # Construct the reduced density matrix of the system by tracing out the
    # environment
    # psi=psi_{ij}|i>|j>
    # We want to make the (sys, env) indices correspond to (row, column) of a
    # matrix, respectively.  Since the environment (column) index updates most
    # quickly in our Kronecker product structure, psi0 is thus row-major ("C style")
    # esempio 3 siti
    #          12345678->  12   000 and 001
    #                      34   010 and 011
    #                      56   100 and 101
    #                      78   110 and 111
    
    #-1 means to be inferred
    psi0 = psi0.reshape([enl.basis_size, -1], order="C")
    rho = np.dot(psi0, psi0.conjugate().transpose())
    return rho


# In[14]:


def get_transformation_matrix(rho,m,enl):
    # Diagonalize the reduced density matrix and sort the eigenvectors by
    # eigenvalue.


    evals, evecs = np.linalg.eigh(rho)
    possible_eigenstates = []
    for eval, evec in zip(evals, evecs.transpose()):
        possible_eigenstates.append((eval, evec))
    possible_eigenstates.sort(reverse=True, key=lambda x: x[0])  # largest eigenvalue first
    
    # Build the transformation matrix from the `m` overall most significant
    # eigenvectors.
    my_m = min(len(possible_eigenstates), m)
    transformation_matrix = np.zeros((enl.basis_size, my_m), dtype='d', order='F')
    for i, (eval, evec) in enumerate(possible_eigenstates[:my_m]):
        transformation_matrix[:, i] = evec
    return transformation_matrix, my_m


# In[15]:


def DMRG_step(sys,site,env,m,H2):
    
    assert is_valid_block(sys)
    assert is_valid_block(env)

    # Enlarge each block by a single site.
    sys_enl = enlarge_2_block(sys,site,H2)

    if sys is env:  # no need to recalculate a second time
        env_enl = sys_enl
    else:
        env_enl = enlarge_2_block(env,site,H2)

    assert is_valid_enlarged_block(sys_enl)
    assert is_valid_enlarged_block(env_enl)

    
    superblock=get_superblock(sys_enl,env_enl,H2)
    
    startpsi=np.zeros(superblock.shape[0])
    
    startpsi[0]=1/np.sqrt(2)
    startpsi[1]=1/np.sqrt(2)
    #startpsi[-1]=1/np.sqrt(2)
    
    energies, psis = eigsh(superblock, k=1, which="SA",v0=startpsi)
    
    energy=energies[0]
    
    
    psi0=psis[:,0]
    
    
    reshapedpsi0=psi0.reshape(sys.basis_size,4,sys.basis_size,4).transpose(1,3,0,2).reshape(4*4,-1)
    

    
    rhomagn0 = np.dot(reshapedpsi0, reshapedpsi0.conjugate().transpose())
    


    
    rho=get_reduced_density_matrix(sys_enl,psi0)
    


    transformation_matrix, my_m =get_transformation_matrix(rho,m,sys_enl)
    
    #truncation_error = 1 - sum([x[0] for x in possible_eigenstates[:my_m]])
    #print("truncation error:", truncation_error)
    
    # Rotate and truncate each operator.
    new_operator_dict = {}
    for name, op in sys_enl.operator_dict.items():
        new_operator_dict[name] = rotate_and_truncate(op, transformation_matrix)
        
    new_env_operator_dict = {}
    for name, op in env_enl.operator_dict.items():
        new_env_operator_dict[name] = rotate_and_truncate(op, transformation_matrix)
        
    newblock = Block(length=sys_enl.length,
                     basis_size=my_m,
                     operator_dict=new_operator_dict)
    
    newenv = Block(length=env_enl.length,
                     basis_size=my_m,
                     operator_dict=new_env_operator_dict) 
    
    
    return newblock, energy,rhomagn0


# In[16]:


def infinite_system_algorithm(block,site, L, m,H2):
    # Repeatedly enlarge the system by performing a single DMRG step, using a
    # reflection of the current block as the environment.
    while 2 * block.length < L:
        #print("L =", block.length * 2 + 2)
        block, energy , rhomagn0 = DMRG_step(block, site ,block, m,H2)
        #print("E/L =", energy / (block.length * 2))
    return rhomagn0


#%%
#fblock,fenergy,frho=infinite_system_algorithm(site,site,100,20,H2)

H1 = -np.array([[1, 0], [0, -1]], dtype='d') #-h*np.array([[0, 1], [1, 0]], dtype='d') # single-site portion of H 

site = Block(length=2, basis_size=4, operator_dict={
    "H": kron(H1,H1),
    "conn_Sx": kron(identity(2),sX),
    "conn_Sy": kron(identity(2),sY),
    })


infinite_system_algorithm(site,site, 10, 10,H2)
