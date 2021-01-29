import numpy as np
Sy = np.array([[0, -1j*0.5], [1j*0.5, 0]]) 
print(np.kron(Sy,Sy))