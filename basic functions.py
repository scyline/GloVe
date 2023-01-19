#compute the weigth function
def weight(x): 
  max = 100
  alpha = 0.75
  if x < 100:
    return (x/max)**alpha
  else:
    return 1
  
#compute the loss function
def cost(M,W1,W2,b1,b2,V): 
    J = 0
    for i in range(V):
      for j in range(V):
        J = J + weight(M[i,j])*(np.dot(W1[i,],np.transpose(W2[j,])) + B1[i,0] + B2[j,0] - np.log(M[i,j]+1))**2
    return J[0,0] 
  
  
  
