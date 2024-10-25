import numpy as np
import random

#Calculates sigmoid
def sigmoid(z):
    return 1/(1+np.exp(-z))

#Returns prediction based on weights w with features
def y(w,x):
    return sigmoid(np.dot(x,w.T))

def ybin(w,x):
    return (y(w,x)>0.5).astype(int)

#Cross-entropy loss function for logistic regression
def entropy(w,x,y_train):
    #prediction
    y_pred=y(w,x)
    N=np.shape(x)[0]
    #cross-entropy
    E=(-np.dot(y_train.T,np.log(y_pred))-np.dot((1-y_train).T,np.log(1-y_pred)))/N
    return E

#Calculates gradient dE/dw of cross-entropy loss function
def gradient(w,x_train,y_train,batch):
    N=np.shape(x_train)[0] 
    #Selects random batch of size batch from N
    batchlist=random.sample(range(N),batch)
    #selects batch from x
    x_batch=[x_train[i] for i in batchlist]
    #selects batch from
    y_batch=[y_train[i] for i in batchlist]
    #prediction
    y_pred=y(w,x_batch)
    #dE/dw
    dw=np.dot((y_pred-y_batch).T,x_batch)/batch
    return dw
    
#Minimizes error function using gradient descent
def optimize(x_train,y_train,iters,step,batch,update):
    N=np.shape(x_train)[0] 
    m=np.shape(x_train)[1]
    #initializes weights w from random distribution
    w=np.random.randn(1,m)
    #gradient descent
    for i in range(iters):
        E=entropy(w,x_train,y_train)
        dw = gradient(w,x_train,y_train,batch)
        w-=step*dw
        if i%update==0:
            print(f"Step {i}: CE loss {E}")
    print(f"Final results: CE loss {E}")
    return w
    