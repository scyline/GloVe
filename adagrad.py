
def adagrad_loop(M, W1, W2, B1, B2, G, eta, epsilon, itr, V, N):

  I_W = np.matrix(np.ones((V,D), dtype=np.float64))
  I_B = np.matrix(np.ones((V,1), dtype=np.float64))
  G_W1 = I_W
  G_W2 = I_W
  G_B1 = I_B
  G_B2 = I_B

  for i in range(itr):
    
    #calculate gradients
    coeff = coefficient(M, W1, W2, B1, B2, V)
    g_W1 = grad_W(coeff, W2, V)
    g_W2 = grad_W(coeff, W1, V)
    g_B1 = grad_B(coeff, V)
    g_B2 = grad_B(coeff, V)

    #update G
    G_W1 = G_W1 + np.multiply(g_W1,g_W1)
    G_W2 = G_W2 + np.multiply(g_W2,g_W2)
    G_B1 = G_B1 + np.multiply(g_B1,g_B1)
    G_B2 = G_B2 + np.multiply(g_B2,g_B2)
    
    #compute M
    M_W1 = eta*np.power(G_W1 + epsilon*I_W, 0.5)
    M_W2 = eta*np.power(G_W2 + epsilon*I_W, 0.5)
    M_B1 = eta*np.power(G_B1 + epsilon*I_B, 0.5)
    M_B2 = eta*np.power(G_B2 + epsilon*I_B, 0.5)

    #update weight
    W1 = W1 - np.multiply(M_W1, g_W1)
    W2 = W2 - np.multiply(M_W2, g_W2)
    B1 = B1 - np.multiply(M_B1, g_B1)
    B2 = B2 - np.multiply(M_B2, g_B2)
  
  return [W1,W2,B1,B2]
