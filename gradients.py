#compute the coefficient matrix
def coefficient(M, W1, W2, B1, B2, V): #OK
  fM = np.vectorize(weight)(M)
  logM = np.log(M+1) #add 1 to avoid log(0)
  ones = np.matrix([1 for x in range(V)]) #row of one
  return 2*np.multiply(fM, np.dot(W1,np.transpose(W2))+np.dot(B1,ones)+np.dot(np.transpose(ones),np.transpose(B2))-logM)

#compute gradient of the biais
def grad_B(coeff, V): #OK
  ones = np.matrix([1 for x in range(V)]) #row of one
  return np.dot(coeff, np.transpose(ones))

#compute gradient of the weights
def grad_W(coeff, W2, V): #OK
  return np.dot(coeff, W2)
