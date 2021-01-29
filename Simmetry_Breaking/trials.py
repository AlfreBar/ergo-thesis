# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 13:55:46 2020

@author: alfre
"""
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from dmrgwaytorho import *
import qutip as q
from scipy import integrate,special
from scipy.linalg import eig,eigh,eigvals,eigvalsh
from scipy.sparse.linalg import eigs
import pickle 
from functions import doApplyHam,diagonalize,diagonalize_2
from functions import isingmodel_zz_x,isingmodel_rev_frac2,isingmodel_zz_x_2
import qutip as q

lambdarange=np.linspace(0,0.5,20)
sX = np.array([[0, 1], [1, 0]], dtype='d') 
Id = np.array([[1, 0], [0, 1]], dtype='d')
sZ = np.array([[1, 0], [0, -1]], dtype='d')

def H2(Sx1, Sx2):  # two-site part of H
    
    return -(kron(Sx1, Sx2.conjugate().transpose()))

h=0.00

# dmrg_dict={}
# for i,lam in enumerate([0,0.1,0.2]):

#     H1 = -lam*np.array([[0, 1], [1, 0]], dtype='d') #-h*np.array([[0, 1], [1, 0]], dtype='d') # single-site portion of H 

#     site = Block(length=1, basis_size=model_d, operator_dict={
#         "H": H1,
#         "conn_Sx": sZ,
#         })

#     dmrg_dict[i]=infinite_system_algorithm(site,site,30, 20,H2)

#     print(lam)
fig,ax=plt.subplots(ncols=4)
for i,l in enumerate([0,0.1,0.3,0.5]):    
    dmrg_sizes={}
    vecsizes=np.arange(4,60,2)
    for size in vecsizes:
        H1 = -l*np.array([[0, 1], [1, 0]], dtype='d') #-h*np.array([[0, 1], [1, 0]], dtype='d') # single-site portion of H 
    
        site = Block(length=1, basis_size=model_d, operator_dict={
            "H": H1,
            "conn_Sx": sZ,
            })
    
        dmrg_sizes[size]=infinite_system_algorithm(site,site,size, 10,H2)
        
    rho00=[dmrg_sizes[el][0][0] for el in dmrg_sizes]
    rho11=[dmrg_sizes[el][3][3] for el in dmrg_sizes]
        #dipende da rapporto lambda/J, piu lambda grande più regge a lunghezze maggiori
    ax[i].scatter(vecsizes,rho00,label=r"$\rho_{00}$")
    ax[i].scatter(vecsizes,rho11,label=r"$\rho_{11}$")
    ax[i].set_title(r"$\lambda=${}".format(l))
    ax[i].set_xlabel("N")
    ax[i].legend()
    

#%%
fig.savefig("lambdaandsizes")
# qrhomagns=[q.Qobj(dmrg_sizes[x][0]) for x in dmrg_sizes]
# labels=[(x,y) for x in range(2) for y in range(2)]
# for el in qrhomagns:
#     q.hinton(el,xlabels=labels,ylabels=labels)
#print(np.kron(sZ,sZ.conjugate().transpose()))
# print(np.kron(sZ,sZ))