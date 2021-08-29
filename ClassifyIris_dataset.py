import numpy as np
## Support Vector Machine
def fit(X,Y):
    X = np.c_[np.ones((X.shape[0], 1)), X]

    epochs = 1
    alpha = 0.001
    m = X.shape[0]
    w = np.zeros(X.shape[1])

    while (epochs < 10000):

      lambda_ = 1 / epochs
      i=0

      for x in X:

              check=Y[i]*(np.dot(x,w.T))


              if check>=1:
               w=w-alpha*(2*lambda_*w)

              else:

                w =w+ alpha * (x*Y[i]-2  *lambda_*w)
              i += 1
      epochs+=1



    y_pred =  np.dot(X,w)

    return y_pred,w