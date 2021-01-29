# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 16:30:57 2020

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
from functions import doApplyHam,isingmodel,diagonalize
from functions import isingmodel_zz_x,isingmodel_rev_frac2,isingmodel_zz_x_2
import qutip as q
length=10
lambdarange=np.linspace(0.1,3)
dict_open0_psis_2={}

for i,l in enumerate(lambdarange):
    
    Energies,psis=diagonalize(3,length,isingmodel_zz_x_2(l,0.00),False)
    
    dict_open0_psis_2[i]=Energies,psis

dict_open0_psis={}

for i,l in enumerate(lambdarange):
    
    Energies,psis=diagonalize(3,length,isingmodel_zz_x(l,0.00),False)
    
    dict_open0_psis[i]=Energies,psis

envec1=[dict_open0_psis[x][0][1]-dict_open0_psis[x][0][0] for x in dict_open0_psis]
envec2=[dict_open0_psis_2[x][0][1]- dict_open0_psis_2[x][0][0] for x in dict_open0_psis_2]

plt.plot(lambdarange,envec1)
plt.plot(lambdarange,envec2)
#perciò si rompe la simmetria capi stupido