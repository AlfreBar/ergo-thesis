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


'''
# Model-specific code for the Heisenberg XXZ chain
h=0.0
Sz1 = np.array([[0.5, 0], [0, -0.5]], dtype='d')  # single-site S^z
Sp1 = np.array([[0, 1], [0, 0]], dtype='d')  # single-site S^+
H1 = -2*h*np.array([[0.5, 0], [0, -0.5]], dtype='d')  # single-site portion of H 



site = Block(length=1, basis_size=model_d, operator_dict={
    "H": H1,
    "conn_Sz": Sz1,
    "conn_Sp": Sp1,
})



def H2(Sz1, Sp1, Sz2, Sp2,J=1,Delta=2):  # two-site part of H
    """Given the operators S^z and S^+ on two sites in different Hilbert spaces
    (e.g. two blocks), returns a Kronecker product representing the
    corresponding two-site term in the Hamiltonian that joins the two sites.
    """
    return (
        -(J) *( 0.5*(kron(Sp1, Sp2.conjugate().transpose()) + kron(Sp1.conjugate().transpose(), Sp2)) + \
        Delta * kron(Sz1, Sz2)))

    
'''
# Questa funzione dati 4 operatori Sz1 Sp1 Sz2 Sp2 e due parametri J,Jz restituisce:
# \begin{equation}
# \frac{J}{2}(S_{p1}\otimes S_{p2}^\dagger + S_{p1}^{\dagger}\otimes S_{p2} )+ J_{z} (S_{z1}\otimes S_{z2})
# \end{equation}
# 
# Ricordiamo che:
# \begin{equation}
#  S^{x}\otimes S^{x} + S^{y}\otimes S^{y} = \frac{1}{2}(a \otimes a^\dagger + a^{\dagger}\otimes a)
#  \end{equation}

# In[5]:



# Ora definiamo una funzione che aggiunge un sito al blocco 

# In[6]:


def enlarge_block(block,site,H2):

    dblock = block.basis_size
    b = block.operator_dict
    dsite = site.basis_size
    s=site.operator_dict
    
    enlarged_operator_dict = {
        "H": kron(b["H"], identity(dsite)) + kron(identity(dblock), s["H"]) + H2(b["conn_Sx"], s["conn_Sx"],b["conn_Sy"],s["conn_Sy"]),
        "conn_Sx": kron(identity(dblock), s["conn_Sx"]),
        "conn_Sy": kron(identity(dblock), s["conn_Sy"]),
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


def get_superblock(sys, site, env, H2):
  
    assert is_valid_block(sys)
    assert is_valid_block(env)

    # Enlarge each block by a single site.
    sys_enl = enlarge_block(sys,site,H2)
    if sys is env:  # no need to recalculate a second time
        env_enl = sys_enl
    else:
        env_enl = enlarge_block(env,site,H2)

    assert is_valid_enlarged_block(sys_enl)
    assert is_valid_enlarged_block(env_enl)
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
    transformation_matrix = np.zeros((enl.basis_size, my_m), dtype='complex', order='F')
    for i, (eval, evec) in enumerate(possible_eigenstates[:my_m]):
        transformation_matrix[:, i] = evec
    return transformation_matrix, my_m


# In[15]:


def DMRG_step(sys,site,env,m,H2):
    
    sys_enl = enlarge_block(sys,site,H2)

    env_enl = enlarge_block(env,site,H2)

    superblock=get_superblock(sys,site,env,H2)
    
    energies, psis = eigsh(superblock, k=2, which="SA")
    
    energy=energies[0]
    
    energy1=energies[1]
    
    psi0=psis[:,0]
    
    psi1=psis[:,1]
    
    reshapedpsi0=psi0.reshape(sys.basis_size,2,sys.basis_size,2).transpose(1,3,0,2).reshape(4,-1)
    
    reshapedpsi1=psi1.reshape(sys.basis_size,2,sys.basis_size,2).transpose(1,3,0,2).reshape(4,-1)

    
    rhomagn0 = np.dot(reshapedpsi0, reshapedpsi0.conjugate().transpose())
    
    rhomagn1 = np.dot(reshapedpsi1, reshapedpsi1.conjugate().transpose())

    
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
    
    
    return newblock, energy,rhomagn0,rhomagn1,rho,rho1,energy1


# In[16]:


def infinite_system_algorithm(block,site, L, m,H2):
    # Repeatedly enlarge the system by performing a single DMRG step, using a
    # reflection of the current block as the environment.
    while 2 * block.length < L:
        #print("L =", block.length * 2 + 2)
        block, energy , rhomagn0,rhomagn1,rho,rho1,energy1 = DMRG_step(block, site ,block, m,H2)
        #print("E/L =", energy / (block.length * 2))
    return rhomagn0,rhomagn1,rho,rho1,energy,energy1


#%%
#fblock,fenergy,frho=infinite_system_algorithm(site,site,100,20,H2)