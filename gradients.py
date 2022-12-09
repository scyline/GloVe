#compute the coefficient matrix
def coefficient(M, W1, W2, B1, B2, V): 
  fM = np.vectorize(weight)(M)
  logM = np.log(M+1) #add 1 to avoid log(0)
  ones = np.matrix([1 for x in range(V)]) #row of one
  return 2*np.multiply(fM,
                       np.dot(np.transpose(W1),W2)+np.dot(B1,ones)+np.dot(np.transpose(ones),np.transpose(B2))-logM)

#compute gradient of the biais
def grad_B(coeff, V): 
  ones = np.matrix([1 for x in range(V)]) 
  return np.dot(coeff, np.transpose(ones))

#compute gradient of the weights
def grad_W(coeff, W2, V): 
  return np.dot(W2, np.transpose(coeff))
