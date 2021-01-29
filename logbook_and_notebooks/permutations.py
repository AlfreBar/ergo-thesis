#!/usr/bin/env python
# coding: utf-8

# In[1]:


import itertools as it
import time
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import qutip as q
import pickle 




def tensor(s):
    
    tensdic={
    "X" : np.array([[0, 1],[ 1, 0]]),
    "Y" : np.array([[0, -1j],[1j, 0]]),
    "Z" : np.array([[1, 0],[0, -1]]),
    "I"  : np.array([[1,0],[0,1]])
    }
    
    lst=[tensdic[r] for r in s]
    
    b=np.kron(lst[0],lst[1])
    
    i=2
    while i<len(s):
        a=lst[i]
        b=np.kron(b,a)
        i+=1

    return b

def isvalid(comb,string):
    teststr=string[:]
    for i in range(len(comb)):
        if comb[i][0] in teststr:
            teststr=teststr.replace(comb[i][0],'',1)
            if comb[i][1] in teststr:
                teststr=teststr.replace(comb[i][1],'',1)
            else:
                return False 
        else:
            return False 
    return True


def paritycheck(perm0, perm1):
    """Check if 2 permutations are of equal parity.

    Assume that both permutation lists are of equal length
    and have the same elements. No need to check for these
    conditions.
    """
    perm1 = perm1[:] ## copy this list so we don't mutate the original

    transCount = 0
    for loc in range(len(perm0) - 1):                         # Do (len - 1) transpositions
        p0 = perm0[loc]
        p1 = perm1[loc]
        if p0 != p1:
            sloc = perm1[loc:].index(p0)+loc          # Find position in perm1
            perm1[loc], perm1[sloc] = p0, p1          # Swap in perm1
            transCount += 1

    # Even number of transpositions means equal parity
    if (transCount % 2) == 0:
        return 1
    else:
        return -1







def func1(phi,l,lamb):
    
    part1=(lamb-np.cos(phi))/np.sqrt(1+lamb**2-2*lamb*np.cos(phi))
    
    return part1*np.cos(l*phi)/np.pi

def func2(phi,l,lamb):
    
    part2=(np.sin(phi))/np.sqrt(1+lamb**2-2*lamb*np.cos(phi))
    
    return -part2*np.sin(l*phi)/np.pi

def gfunc(l,lamb):
    
    integ1=integrate.quad(func1,0,np.pi,args=(l,lamb))
    integ2=integrate.quad(func2,0,np.pi,args=(l,lamb))
    
    return integ1[0]-integ2[0]


L=2

def g_s(lambdas):

    g={}
    for n in range(-L,L): 
        for l in lambdas:
            g[n,l]=gfunc(n,l)
    return g




# leggi coppia, se $x>y$ aggiungi un meno e swappa. Calcola differenza d. per  d>0 se primo dispari allora $g_{(d-1)/2} $se pari: $g_{(d+1)/2}$



def assign_g(a,b,g,lamb):
    if a==b:
        return 1
    d=b-a 
    if d%2==1:
        if a%2==1:
            return 1j*g[(d-1)//2,lamb]
        elif a%2==0:
            
            return -1j*g[-((d+1)//2),lamb]
    else:
        return 0






def get_prods():

    prod=[list(el) for el in list(it.product('IXYZ', repeat=L))]

    reduced_prod=[]

    for x in prod:
        temp=0
        for i in range(L):
            if x[i]=="X" or x[i]=="Y":
                temp+=1
        if temp%2==0:
            reduced_prod.append(x)
        

    letter_prod=["".join(el) for el in reduced_prod]

    for i in range(len(reduced_prod)):
        for j in range(L):
            if reduced_prod[i][j]=="I":
                reduced_prod[i][j]=""
            if reduced_prod[i][j]=="Z":
                reduced_prod[i][j]=("{}{}".format(2*j+1,2*j+2))
            elif reduced_prod[i][j]=="X":
                newel=""
                for k in range(j):
                    newel+=("{}{}".format(2*k+1,2*k+2))
                newel+=("{}".format(2*j+1))
                reduced_prod[i][j]=newel
            elif reduced_prod[i][j]=="Y":
                newel=""
                for k in range(j):
                    newel+=("{}{}".format(2*k+1,2*k+2))
                newel+=("{}".format(2*j+2))
                reduced_prod[i][j]=newel

    num_prod=["".join(x) for x in reduced_prod]

    return num_prod, letter_prod
def get_dicts():

    joined_prod,letter_prod=get_prods()

    dict_comb={}
    for trial in joined_prod:
        reduced_comb=[]
        for x in list(it.combinations(trial,2)):
            if int(x[0])==int(x[1]) or (int(x[0])+int(x[1]))%2==1:
                if int(x[0])>int(x[1]):
                    reduced_comb.append((x[1],x[0]))
                else:
                    reduced_comb.append(x)
        dict_comb[trial]=reduced_comb

    final_dict={}
    perm_dict={}
    for strings in dict_comb:
        
        valid_comb=set(longcomb for longcomb in it.combinations(dict_comb[strings],int(len(strings)/2)) if isvalid(longcomb,strings))

        final_dict[strings]=[list(it.chain(*el)) for el in valid_comb]

        def stringparity(string):
            return paritycheck(string,list(strings))   
 
        perm_dict[strings]=list(map(stringparity,final_dict[strings]))
        # if strings=="1124":
        #     print(list(set(s) for s in valid_comb))
        
        
    return final_dict, perm_dict

def build_gdict(final_dict, perm_dict,g,lam):

    joined_prod,letter_prod=get_prods()

    gdict={}

    n=0
    for strings in final_dict:

        finalg=0
        a=time.time()
        for j,perm in enumerate(final_dict[strings]):

            gval=1
            
            for i in range(0,len(perm),2):
                
                gval=gval*assign_g(int(perm[i]),int(perm[i+1]),g,lam)
            
            gval=gval*perm_dict[strings][j]

            finalg=finalg+gval
        b=time.time()
        #print("{:.3f}".format(b-a))
        if abs(finalg)>1e-6:
            gdict[letter_prod[n]]=finalg
        else:
            gdict[letter_prod[n]]=0

        n+=1
    return gdict



# $\rho_2=\frac{1}{4}\left(\left(\sigma_1^0 \otimes\sigma_2^0\right)-g_{-1} \left(\sigma_1^x \otimes\sigma_2^x\right) - g_{1} \left(\sigma_1^y \otimes\sigma_2^y\right)+\left(g_{0}^2-g_1g_{-1}\right)\left(\sigma_1^z \otimes\sigma_2^z\right)+g_0\left(\sigma_1^0 \otimes\sigma_2^z+\sigma_1^z \otimes\sigma_2^0\right)\right)$

# $\rho_3=\frac{1}{8}\left(\left(\sigma_1^0 \otimes\sigma_2^0\right)-g_{-1} \left(\sigma_1^x \otimes\sigma_2^x\right) - g_{1} \left(\sigma_1^y \otimes\sigma_2^y\right)+\left(g_{0}^2-g_1g_{-1}\right)\left(\sigma_1^z \otimes\sigma_2^z\right)+g_0\left(\sigma_1^0 \otimes\sigma_2^z+\sigma_1^z \otimes\sigma_2^0\right)\right)$



def assign_i(word):
    f=1
    for i,x in enumerate(word):
        if x=="I":
            pass
        elif x=="X" or x=="Y":
            f=f*(-1j)**i
        elif x=="Z":    
            f=f*(-1j)
    return f

def rho_2new(final_dict,perm_dict,g,l):

    start=time.time()
    gd=build_gdict(final_dict, perm_dict,g,l)
    mid=time.time()
    rho=0
    for s in gd:
        if len(s)==2:
            rho=rho+(1/4)*assign_i(s)*gd[s]*tensor(s)

    return rho

def rho_3new(final_dict,perm_dict,g,l):

    start=time.time()
    gd=build_gdict(final_dict, perm_dict,g,l)
    mid=time.time()
    rho=0
    for s in gd:
        if len(s)==3:
            rho=rho+(1/8)*assign_i(s)*gd[s]*tensor(s)
    return rho

def rho_4new(final_dict,perm_dict,g,l):

    start=time.time()
    gd=build_gdict(final_dict, perm_dict,g,l)
    mid=time.time()
    rho=0
    for s in gd:
        if len(s)==4:
            rho=rho+(1/16)*assign_i(s)*gd[s]*tensor(s)

    return rho

def rho_3(final_dict,perm_dict,g,l):

    start=time.time()
    gd=build_gdict(final_dict, perm_dict,g,l)
    mid=time.time()

    rho= (1/8)*(tensor( I ,I ,I ))-1j*gd["IIZ"]*tensor(I,I,Z) + \
    (-1+0j)*gd["IXX"]*tensor(I,X,X) + \
    (-1+0j)*gd["IXY"]*tensor(I,X,Y) + \
    (-1+0j)*gd["IYX"]*tensor(I,Y,X) + \
    (-1+0j)*gd["IYY"]*tensor(I,Y,Y) + \
    (-1+0j)*gd["IZZ"]*tensor(I,Z,Z) + \
    (-1+0j)*gd["XIX"]*tensor(X,I,X) + \
    (-1+0j)*gd["XIY"]*tensor(X,I,Y) + \
    1j*gd["XXZ"]*tensor(X,X,Z) + \
    1j*gd["XYZ"]*tensor(X,Y,Z) + \
    1j*gd["XZX"]*tensor(X,Z,X) + \
    1j*gd["XZY"]*tensor(X,Z,Y) + \
    (-1+0j)*gd["YIX"]*tensor(Y,I,X) + \
    (-1+0j)*gd["YIY"]*tensor(Y,I,Y) + \
    1j*gd["YXZ"]*tensor(Y,X,Z) + \
    1j*gd["YYZ"]*tensor(Y,Y,Z) + \
    1j*gd["YZX"]*tensor(Y,Z,X) + \
    1j*gd["YZY"]*tensor(Y,Z,Y) + \
    (-1+0j)*gd["ZIZ"]*tensor(Z,I,Z) + \
    1j*gd["ZXX"]*tensor(Z,X,X) + \
    1j*gd["ZXY"]*tensor(Z,X,Y) + \
    1j*gd["ZYX"]*tensor(Z,Y,X) + \
    1j*gd["ZYY"]*tensor(Z,Y,Y) + \
    1j*gd["ZZZ"]*tensor(Z,Z,Z)
    end=time.time()

    return rho

if __name__ == "__main__":
    num,lett=get_prods()
    dictr,permtr=get_dicts()
    lambdarange=np.linspace(0.1,10,10)
    gtr=g_s(lambdarange)
    finalg=build_gdict(dictr,permtr,gtr,0.1)
    rhotr=rho_2new(dictr,permtr,gtr,10)
    print(rhotr)
#print([(x,assign_i(x))  for x in lett])


'''

joi,lett=get_prods()
print(joi,lett)




for l in lett:

    s1="{}*gd[\"{}\"]*".format(assign_i(l),l)
    s2="tensor({},{},{}) + \\".format(l[0],l[1],l[2])

    print(s1+s2)
i_assigned=list(map(assign_i,lett))
print(i_assigned)


# %%
def assign_g_debug(a,b):
    if a>b:
        d=a-b   
        if d%2==1:     
            if b%2==1:
 
                return "-ig[{}]".format((d-1)//2)
            elif b%2==0:
                
                return "ig[{}]".format(-(d+1)//2) 
        else:
            return 0
    elif a<b:
        d=b-a 
        if d%2==1:
            if a%2==1:
 
                return "ig[{}]".format((d-1)//2)
            elif a%2==0:
 
                return "-ig[{}]".format(-(d+1)//2)
        else:
            return 0
    elif a==b:        
        return "1"
mat = [[None for c in range(1,7)] for r in range(1,7)]
for x in range(1,7):
    for y in range(1,7):
        mat[x-1][y-1]=assign_g_debug(x,y)
    # %%
mat=np.array(mat)
# %%
def bmatrix(a):
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv +=  [r'\end{bmatrix}']
    return '\n'.join(rv)

a=bmatrix(mat)
print(a)
# %%
'''