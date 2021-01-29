
import numpy as np
import cmath
import math
import time
import matplotlib.pyplot as plt
from scipy import sparse
import os
from scipy.linalg import eigh
#generate the basis of ground subspace:
#(Using bitwise operators)
def generate_map():
    state_list=[]
    state_dict={}
    k1=-1
    for j in range(2<<(n-1)):         #Use the '<<' to walk through all the basis
        S=sum((j>>k)&1 for k in range(n))   #Use '>>' and '&' to sum up all the 1 of one basis
        if S==n//2:
            k1+=1
            state_list.append(j)
            state_dict[j]=k1
    return state_list,state_dict

def generate_hsub():
    H_sub=sparse.lil_matrix((matrix_dim,matrix_dim))
    for x in range(0,matrix_dim):
        x1=state_list[x]
        H_sub[x,x]=sum(j_1*(float((x1>>k)&1)-1/2)*(float((x1>>((k+1)%n))&1)-1/2)+
        j_2*(float((x1>>k)&1)-1/2)*(float((x1>>((k+2)%n))&1)-1/2)+
        j_4*(float((x1>>k)&1)-1/2)*(float((x1>>((k+4)%n))&1)-1/2) for k in range(n))
        #if k and k+1 is the same ,the value is 1/4 and if different ,is -1/4,sum them up
        for k in range(n):
            if ((x1>>k)&1)+((x1>>((k+1)%n))&1)==1:
                #if the basis is like 01 or 10,then the element generated is 1/2
                x2=state_dict[x1^(1<<k)^(1<<((k+1)%n))]
                H_sub[x,x2]=j_1/2
            if ((x1>>k)&1)+((x1>>((k+2)%n))&1)==1:
                x3=state_dict[x1^(1<<k)^(1<<((k+2)%n))]
                H_sub[x,x3]=j_2/2
            if ((x1>>k)&1)+((x1>>((k+4)%n))&1)==1:
                x4=state_dict[x1^(1<<k)^(1<<((k+4)%n))]
                H_sub[x,x4]=j_4/2
    return H_sub

def lanczos(A,matrix_dim,v0):
    steps = 100
    
    T = np.zeros((steps,steps))
    # v0 = np.random.rand(matrix_dim)
    # v0 = 1/np.linalg.norm(v0) * v0
    v1 = A@v0
    for j in range(steps):
        a = v0@v1
        T[j,j] = a
        v1= -a*v0 + v1
        b = np.linalg.norm(v1)
        if j+1 <= steps - 1:
            T[j,j+1] = b
            T[j+1,j] = b
        v1=(1/b) * v1
        v0 , v1 = v1, -b * v0
        v1 = A@v0 + v1
    eigen_value , eigen_vector = eigh(T)
    eigen_value=min(eigen_value)
    return eigen_value,eigen_vector,steps

def lanczos_activate(A,matrix_dim,v0,ground_state):
    steps = 100
    T = np.zeros((steps,steps))
    # v0 = np.random.rand(matrix_dim)
    # v0 = 1/np.linalg.norm(v0) * v0
    # v0=v0-(v0@ground_state)*ground_state
    v0 = 1/np.linalg.norm(v0) * v0
    v1 = A@v0
    for j in range(steps):
        a = v0@v1
        b = ground_state@v1
        T[j,j] = a
        v1= -a*v0 -b*ground_state + v1
        c = np.linalg.norm(v1)
        if j+1 <= steps - 1:
            T[j,j+1] = c
            T[j+1,j] = c
        v1=(1/c) * v1
        v0 , v1 = v1, -c * v0
        v0=v0-(v0@ground_state)*ground_state
        v1 = A@v0 + v1
    eigen_value , eigen_vector = eigh(T)
    eigen_value=min(eigen_value)
    return eigen_value,eigen_vector,steps

def lanczos_state(hm,matrix_dim,eig_vector,v0,step):
    ground_state=np.zeros((matrix_dim))
    ground_state=eig_vector[0,0]*v0
    v1=hm@v0
    for i in range(1,step):
        a0=v0@v1
        v1=-a0*v0+v1
        b1=np.linalg.norm(v1)
        v1=1/b1*v1
        ground_state=ground_state+eig_vector[i,0]*v1
        v0,v1=v1,-b1*v0
        v1=hm@v0+v1
    ground_state=ground_state/np.linalg.norm(ground_state)
    return ground_state

if __name__ == '__main__':
    # print('Please input the length of your chain:n=')
    # n=int(input())
    # print("Heisenberg Chain,n=",n)
    length=10
    t1=time.time()
    eigen_value_list=[]
    delta_list=[]
    # j_4_list=[]
    # for i in range(0,100,1):
    j_1=1
    j_2=0.5
    j_4=0
    
    n_list=[]
    for n in range(4,length,2):
        state_list,state_dict=generate_map()
        # print(state_list,state_dict)
        matrix_dim=np.size(state_list)
        H_sub=generate_hsub()
        vinit=np.random.rand(matrix_dim)
        vinit=1/np.linalg.norm(vinit)*vinit
        eigen_value0,eig_vector0,step=lanczos(H_sub,matrix_dim,vinit)
        print("Ground State Energy=",eigen_value0,"n=",n,"matrix_dim=",matrix_dim,eig_vector0.shape)
        print("ground_state is ",eig_vector0)
        ground_state=lanczos_state(H_sub,matrix_dim,eig_vector0,vinit,step)
    #print(ground_state)
    #e1,v1=eigh(H_sub.todense())
    #print(e1)
    #print(v1[:,0])
        eigen_value1,eig_vector1,step=lanczos_activate(H_sub,matrix_dim,vinit,ground_state)
        print("eigen_value1=",eigen_value1)
        # print("activate_state=",eig_vector1)
        delta=eigen_value1-eigen_value0
        delta_list.append(delta)
    #print(H_sub)
    #eigen_value=lanczos(H_sub,matrix_dim,ground_state)
    # eigen_value_n=eigen_value/n
    # eigen_value_list.append(eigen_value_n)
    # j_4_list.append(j_4)
        n_list.append(1/n)
    t2=time.time()
    
    # delta_list.append(0)
    # n_list.append(0)
    #print('the first activate state energy=',eigen_value1)
    y=np.array(delta_list)
    # print(delta_list,n_list)
    #y=np.reshape(y,[15,])
    #now let's plot the graph to show the correlation function on every site.
    plt.title("Graph of delta-1/n when j1=2,j2=1")
    plt.xlabel("1/n",)
    plt.ylabel("delta")
    plt.axis([0.0,0.25,0,2])
    # plt.plot(0,0,'k')
    plt.plot(n_list,y,"bo-")
    plt.show()
    print("Time Spent=",t2-t1,"s")