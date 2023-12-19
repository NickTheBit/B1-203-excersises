import numpy as np
import matplotlib.pyplot as plt



step=0.01
Max_h=8 # Max height
N_=Max_h+step
steepnes=3 # Steepnes of the non-linear region (steepness of the sigmoid function)
N_fine=7
spread=1


#Boundaries
lowerBoundary=3
higherBoundary=5

#Number of discrete states
N=N_fine*2+2+8-(higherBoundary+spread)
print(N)

# This is a pice wise continuous function from x=[0,Max_h], and consists of two linear regions 
# and two sigmoid functions near the boundaries
Z=2*N_fine+spread
def Non_linear_mapping(x):
    if x <=lowerBoundary-spread:
        y=x
    elif x <=lowerBoundary+spread:
        z = 1/(1 + np.exp(-steepnes*(x-lowerBoundary))) 
        y=N_fine*z+lowerBoundary-spread
        
    elif x <=higherBoundary+spread:
        z = 1/(1 + np.exp(-steepnes*(x-higherBoundary))) 
        y=N_fine*z+N_fine+spread
    else:
        y=x-(higherBoundary+spread)+Z
    return int(y) # Discretize output
        

y=[]
discrete_step=[]
y_prev=0
index=0

for x in np.arange(0, N_, step):
    
    y.append(Non_linear_mapping(x))
    
    if y[index] > y_prev:
        discrete_step.append(x)
    
    y_prev=y[index]
    index=index+1


x = np.arange(0, N_, step)

if spread >= (higherBoundary-lowerBoundary)/2:
    print(f"Carful, maximum spread= {(higherBoundary-lowerBoundary)/2} for the set boundaries")

plt.style.use('dark_background')

plt.figure(1)
plt.plot(x,y)
plt.title("Non-linear mapping")
plt.xlabel("Whater level continuous") 
plt.ylabel("Water level discreete") 

# Plot discretization
plt.figure(2)

for i in range(0,len(discrete_step)):
    plt.axhline(y=discrete_step[i], color = 'b', linestyle = ':') 
plt.ylabel("Water level continuous") 
plt.title("Discreete levels")
  
plt.show() 
