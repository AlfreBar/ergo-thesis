#!/usr/bin/env python
#
# Simple DMRG tutorial.  This code integrates the following concepts:
#  - Infinite system algorithm
#  - Finite system algorithm
#
# Copyright 2013 James R. Garrison and Ryan V. Mishmash.
# Open source under the MIT license.  Source code at
# <https://github.com/simple-dmrg/simple-dmrg/>

# This code will run under any version of Python >= 2.6.  The following line
# provides consistency between python2 and python3.
from __future__ import print_function, division  # requires Python >= 2.6

# numpy and scipy imports
import numpy as np
from scipy.sparse import kron, identity
from scipy.sparse.linalg import eigsh  # Lanczos routine from ARPACK

# We will use python's "namedtuple" to represent the Block and EnlargedBlock
# objects
from collections import namedtuple

Block = namedtuple("Block", ["length", "basis_size", "operator_dict"])
EnlargedBlock = namedtuple("EnlargedBlock", ["length", "basis_size", "operator_dict"])

def is_valid_block(block):
    for op in block.operator_dict.values():
        if op.shape[0] != block.basis_size or op.shape[1] != block.basis_size:
            return False
    return True

# This function should test the same exact things, so there is no need to
# repeat its definition.
is_valid_enlarged_block = is_valid_block

# Model-specific code for the Heisenberg XXZ chain
model_d = 2  # single-site basis size

'''

sX = np.array([[0, 1], [1, 0]], dtype='d') 


def H2(Sx1, Sx2):  # two-site part of H
    
    return -0.5*(kron(Sx1, Sx2))

lam=0.5

H1 = -0.5*lam*np.array([[1, 0], [0, -1]], dtype='d')

#-h*np.array([[0, 1], [1, 0]], dtype='d') # single-site portion of H 

site = Block(length=1, basis_size=model_d, operator_dict={
    "H": H1,
    "conn_Sx": sX,
    })

'''
# conn refers to the connection operator, that is, the operator on the edge of
# the block, on the interior of the chain.  We need to be able to represent S^z
# and S^+ on that site in the current basis in order to grow the chain.
#initial_block = Block(length=1, basis_size=model_d, operator_dict={
#    "H": H1,
#    "conn_Sz": Sz1,
#    "conn_Sp": Sp1,
#})

def enlarge_block(block,site,H2):
    
    dblock = block.basis_size
    b = block.operator_dict
    dsite = site.basis_size
    s=site.operator_dict
    
    enlarged_operator_dict = {
        "H": kron(b["H"], identity(dsite)) + kron(identity(dblock), s["H"]) + H2(b["conn_Sx"], s["conn_Sx"]),
        "conn_Sx": kron(identity(dblock), s["conn_Sx"]),
    }

    return EnlargedBlock(length=(block.length + 1),
                         basis_size=(dblock * model_d),
                         operator_dict=enlarged_operator_dict)

def rotate_and_truncate(operator, transformation_matrix):
    """Transforms the operator to the new (possibly truncated) basis given by
    `transformation_matrix`.
    """
    return transformation_matrix.conjugate().transpose().dot(operator.dot(transformation_matrix))

def single_dmrg_step_with_rho(sys,site,env,H2,m):
    

    sys_enl = enlarge_block(sys,site,H2)

    env_enl = enlarge_block(env,site,H2)
    
    m_sys_enl = sys_enl.basis_size
    m_env_enl = env_enl.basis_size
    sys_enl_op = sys_enl.operator_dict
    env_enl_op = env_enl.operator_dict
    
    assert is_valid_block(sys)
    assert is_valid_block(env)


    assert is_valid_enlarged_block(sys_enl)
    assert is_valid_enlarged_block(env_enl)
    
    superblock_hamiltonian= kron(sys_enl_op["H"], identity(m_env_enl)) + kron(identity(m_sys_enl), env_enl_op["H"]) + \
                             H2(sys_enl_op["conn_Sx"], env_enl_op["conn_Sx"])
    
    
    energy, psi0 = eigsh(superblock_hamiltonian, k=1, which="SA")
    
   # energy=energies[0]
    
   # energy1=energies[1]
    
  #  psi0=psis[:,0]
    
   # psi1=psis[:,1]
    
    reshapedpsi0=psi0.reshape(sys.basis_size,2,env.basis_size,2).transpose(1,3,0,2).reshape(4,-1)
    
   # reshapedpsi1=psi1.reshape(sys.basis_size,2,sys.basis_size,2).transpose(1,3,0,2).reshape(4,-1)

    
    rhomagn0 = np.dot(reshapedpsi0, reshapedpsi0.conjugate().transpose())
    
    #rhomagn1 = np.dot(reshapedpsi1, reshapedpsi1.conjugate().transpose())

    
    psi0 = psi0.reshape([sys_enl.basis_size, -1], order="C")
    rho = np.dot(psi0, psi0.conjugate().transpose())

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
    transformation_matrix = np.zeros((sys_enl.basis_size, my_m), dtype='d', order='F')
    for i, (eval, evec) in enumerate(possible_eigenstates[:my_m]):
        transformation_matrix[:, i] = evec

    truncation_error = 1 - sum([x[0] for x in possible_eigenstates[:my_m]])
    #print("truncation error:", truncation_error)

    # Rotate and truncate each operator.
    new_operator_dict = {}
    for name, op in sys_enl.operator_dict.items():
        new_operator_dict[name] = rotate_and_truncate(op, transformation_matrix)

    newblock = Block(length=sys_enl.length,
                     basis_size=my_m,
                     operator_dict=new_operator_dict)
    
    return newblock,energy,rhomagn0

def single_dmrg_step(sys,site,env,H2,m):
    

    sys_enl = enlarge_block(sys,site,H2)

    env_enl = enlarge_block(env,site,H2)
    
    m_sys_enl = sys_enl.basis_size
    m_env_enl = env_enl.basis_size
    sys_enl_op = sys_enl.operator_dict
    env_enl_op = env_enl.operator_dict
    
    assert is_valid_block(sys)
    assert is_valid_block(env)


    assert is_valid_enlarged_block(sys_enl)
    assert is_valid_enlarged_block(env_enl)
    
    superblock_hamiltonian= kron(sys_enl_op["H"], identity(m_env_enl)) + kron(identity(m_sys_enl), env_enl_op["H"]) + \
                             H2(sys_enl_op["conn_Sx"], env_enl_op["conn_Sx"])
    
    
    energy, psi0 = eigsh(superblock_hamiltonian, k=1, which="SA")
    
   # energy=energies[0]
    
   # energy1=energies[1]
    
  #  psi0=psis[:,0]
    
   # psi1=psis[:,1]
    
    #reshapedpsi0=psi0.reshape(sys.basis_size,2,env.basis_size,2).transpose(1,3,0,2).reshape(4,-1)
    
   # reshapedpsi1=psi1.reshape(sys.basis_size,2,sys.basis_size,2).transpose(1,3,0,2).reshape(4,-1)

    
    #rhomagn0 = np.dot(reshapedpsi0, reshapedpsi0.conjugate().transpose())
    
    #rhomagn1 = np.dot(reshapedpsi1, reshapedpsi1.conjugate().transpose())

    
    psi0 = psi0.reshape([sys_enl.basis_size, -1], order="C")
    rho = np.dot(psi0, psi0.conjugate().transpose())

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
    transformation_matrix = np.zeros((sys_enl.basis_size, my_m), dtype='d', order='F')
    for i, (eval, evec) in enumerate(possible_eigenstates[:my_m]):
        transformation_matrix[:, i] = evec

    truncation_error = 1 - sum([x[0] for x in possible_eigenstates[:my_m]])
    #print("truncation error:", truncation_error)

    # Rotate and truncate each operator.
    new_operator_dict = {}
    for name, op in sys_enl.operator_dict.items():
        new_operator_dict[name] = rotate_and_truncate(op, transformation_matrix)

    newblock = Block(length=sys_enl.length,
                     basis_size=my_m,
                     operator_dict=new_operator_dict)
    
    return newblock,energy
def graphic(sys_block, env_block, sys_label="l"):
    """Returns a graphical representation of the DMRG step we are about to
    perform, using '=' to represent the system sites, '-' to represent the
    environment sites, and '**' to represent the two intermediate sites.
    """
    assert sys_label in ("l", "r")
    graphic = ("=" * sys_block.length) + "**" + ("-" * env_block.length)
    if sys_label == "r":
        # The system should be on the right and the environment should be on
        # the left, so reverse the graphic.
        graphic = graphic[::-1]
    return graphic

def infinite_system_algorithm(block,site,L, H2,m):
    block = initial_block
    # Repeatedly enlarge the system by performing a single DMRG step, using a
    # reflection of the current block as the environment.
    while 2 * block.length < L:
        print("L =", block.length * 2 + 2)
        block, energy,rhomagn0 = single_dmrg_step(block,site, block,H2, m=m)
        print("E/L =", energy / (block.length * 2))
    return rhomagn0

def finite_system_algorithm(initial_block,site,H2,L, m_warmup, m_sweep_list):
    assert L % 2 == 0  # require that L is an even number

    # To keep things simple, this dictionary is not actually saved to disk, but
    # we use it to represent persistent storage.
    block_disk = {}  # "disk" storage for Block objects

    # Use the infinite system algorithm to build up to desired size.  Each time
    # we construct a block, we save it for future reference as both a left
    # ("l") and right ("r") block, as the infinite system algorithm assumes the
    # environment is a mirror image of the system.
    block = initial_block
    block_disk["l", block.length] = block
    block_disk["r", block.length] = block
    while 2 * block.length < L:
        # Perform a single DMRG step and save the new Block to "disk"
        #print(graphic(block, block))
        block, energy = single_dmrg_step(block,site, block,H2, m=m_warmup)
        #print("E/L =", energy / (block.length * 2))
        block_disk["l", block.length] = block
        block_disk["r", block.length] = block

    # Now that the system is built up to its full size, we perform sweeps using
    # the finite system algorithm.  At first the left block will act as the
    # system, growing at the expense of the right block (the environment), but
    # once we come to the end of the chain these roles will be reversed.
    sys_label, env_label = "l", "r"
    sys_block = block; del block  # rename the variable
    for m in m_sweep_list[:-1]:
        while True:
            # Load the appropriate environment block from "disk"
            env_block = block_disk[env_label, L - sys_block.length - 2]
            if env_block.length == 1:
                # We've come to the end of the chain, so we reverse course.
                sys_block, env_block = env_block, sys_block
                sys_label, env_label = env_label, sys_label

            # Perform a single DMRG step.
            #print(graphic(sys_block, env_block, sys_label))
            sys_block, energy = single_dmrg_step(sys_block,site, env_block,H2, m=m)

            #print("E/L =", energy / L)

            # Save the block from this step to disk.
            block_disk[sys_label, sys_block.length] = sys_block

            # Check whether we just completed a full sweep.
            if sys_label == "l" and 2 * sys_block.length == L:
                #print(sys_block.length)

                break  # escape from the "while True" loop
    while True:
        env_block = block_disk[env_label, L - sys_block.length - 2]
        if env_block.length == 1:
            # We've come to the end of the chain, so we reverse course.
            sys_block, env_block = env_block, sys_block
            sys_label, env_label = env_label, sys_label

        # Perform a single DMRG step.
        sys_block, energy,rhomagn0 = single_dmrg_step_with_rho(sys_block,site, env_block,H2, m=m_sweep_list[-1])

        #print("E/L =", energy / L)

        # Save the block from this step to disk.
        block_disk[sys_label, sys_block.length] = sys_block

        if sys_label == "l" and 2 * sys_block.length == L:
            #print(graphic(sys_block, env_block, sys_label))
            break
    return rhomagn0           

if __name__ == "__main__":
    np.set_printoptions(precision=10, suppress=True, threshold=10000, linewidth=300)

    #infinite_system_algorithm(L=100, m=20)
    rhom=finite_system_algorithm(site,site,H2,L=20, m_warmup=10, m_sweep_list=[10, 20, 30])
    a="helo"