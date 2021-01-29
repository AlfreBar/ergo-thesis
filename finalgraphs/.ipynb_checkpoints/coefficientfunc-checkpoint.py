from scipy import integrate,special
import numpy as np

def func1(phi,l,lamb,gamma):
    
    part1=(lamb-np.cos(phi))/np.sqrt((lamb-np.cos(phi))**2+(gamma*np.sin(phi))**2)
    
    return part1*np.cos(l*phi)/np.pi

def func2(phi,l,lamb,gamma):
    
    part2=(gamma*(np.sin(phi)))/np.sqrt((lamb-np.cos(phi))**2+(gamma*np.sin(phi))**2)
    
    return -part2*np.sin(l*phi)/np.pi

def gfunc(l,lamb,gamma=1):
    
    integ1=integrate.quad(func1,0,np.pi,args=(l,lamb,gamma))
    integ2=integrate.quad(func2,0,np.pi,args=(l,lamb,gamma))
    
    return integ1[0]-integ2[0]

def g_s(lambdas,gamma=1):

    g={}
    for n in range(-L,L): 
        for l in lambdas:
            g[n,l]=gfunc(n,l,gamma)
    return g

def assign_g(a,b,g,lamb):
    if a==b:
        return 0
    d=b-a 
    if d%2==1:
        if a%2==1:
            return 1j*g[(d-1)//2,lamb]
        elif a%2==0:
            
            return -1j*g[-((d+1)//2),lamb]
    else:
        return 0
    
def assign_g_debug(a,b):
    if a==b:
        return 0
    d=b-a 
    if d%2==1:
        if a%2==1:
            return "ig[{}]".format((d-1)//2)
        elif a%2==0:
            
            return "-ig[{}]".format(-(d+1)//2)
    else:
        return 0

def get_mat(gdic,lam):
    mat = [[None for c in range(1,2*L+1)] for r in range(1,2*L+1)]
    for x in range(1,2*L+1):
        for y in range(1,2*L+1):
            mat[x-1][y-1]=assign_g(x,y,gdic,lam)  
    return np.array(mat)