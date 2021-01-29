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


# In[4]:

model_d = 2  # single-site basis size


# In[6]:


def enlarge_block(block,site,H2):

    dblock = block.basis_size
    b = block.operator_dict
    dsite = site.basis_size
    s=site.operator_dict
    
    enlarged_operator_dict = {
        "H": kron(b["H"], identity(dsite)) + kron(identity(dblock), s["H"]) + H2(b["conn_Sx"], s["conn_Sx"]),
        "conn_Sx": kron(identity(dblock), s["conn_Sx"])
    }

    return EnlargedBlock(length=(block.length + 1),
                         basis_size=(dblock * model_d),
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


    assert is_valid_enlarged_block(sys_enl)
    assert is_valid_enlarged_block(env_enl)
     # Construct the full superblock Hamiltonian.
    m_sys_enl = sys_enl.basis_size
    m_env_enl = env_enl.basis_size
    sys_enl_op = sys_enl.operator_dict
    env_enl_op = env_enl.operator_dict
    superblock_hamiltonian= kron(sys_enl_op["H"], identity(m_env_enl)) + kron(identity(m_sys_enl), env_enl_op["H"]) + \
                             H2(sys_enl_op["conn_Sx"], env_enl_op["conn_Sx"])

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
    
    sys_enl = enlarge_block(sys,site,H2)

    env_enl = enlarge_block(env,site,H2)

    superblock=get_superblock(sys_enl,env_enl,H2)
    
    energies, psis = eigsh(superblock, k=2, which="SA")
    
    energy=energies[0]
    
    energy1=energies[1]
    
    psi0=psis[:,0]
    
    psi1=psis[:,1]
    

    rho=get_reduced_density_matrix(sys_enl,psi0)
    
    rho1=get_reduced_density_matrix(sys_enl,psi1)


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
    
    
    return newblock,energy,energy1


# In[16]:


def infinite_system_algorithm(block,site, L, m,H2):
    # Repeatedly enlarge the system by performing a single DMRG step, using a
    # reflection of the current block as the environment.
    while 2 * block.length < L:
        #print("L =", block.length * 2 + 2)
        block, energy ,energy1 = DMRG_step(block, site ,block, m,H2)
        #print("E/L =", energy / (block.length * 2))
    return energy1-energy


#%%
#fblock,fenergy,frho=infinite_system_algorithm(site,site,100,20,H2)