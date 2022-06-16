#Train a simplified MLP for regression with backpropagation and Gradient Descent optmization (Only for didactic purposes)
#X: the n x p matrix containing the input data (where p is the number of input variables)
#y: the n x 1 vector of continuous responses 
#epochs: a constant for the number of epochs (e.g. 1000)
#lr: a constant for the learning rate (e.g. 0.01)
#act_h: a vector of strings for the hidden layers activaction functions, only linear, 
#rectified linear unit (relu), sigmoid, and hyperbolic tangent (tanh) are avaliable at this moment.
#Example: act_h = c("sigmoid), act_h = c("relu", "relu"); or act_h = c("tanh","sigmoid", "linear")
#act_o: the string with the output layer, only linear function is avaliable at this moment ("linear")
#momentum: a constant between 0 and 1 for the momentum term, e.g. 0.97 (helps gradient descent algorithm to escape from local minima)
#lambda: a constant betwen 0 and ∞ for the L2 regularization factor (e.g. 0.01, 1, 10...), higher values imply in more regularized models and 0 value means no regularization

train.nn = function(X, y, neurons, epochs, lr, act_h, act_o, momentum, lambda){
  #Get dimensions
  n_h = neurons
  n = dim(X)[1]
  p = dim(X)[2]
  y_n = dim(y)[2]
  
  
  ############## Pre-processing functions ###################
  
  #Check the provided strings for the activaction functions
  act = function(act.fun, z){
    if(act.fun == 'relu'){z[which(z<0)] = 0;return(z)
    }else if(act.fun == 'linear'){return(z)  
    }else if (act.fun == 'sigmoid'){return(1/(1+exp(-z)))
    }else if (act.fun == 'tanh'){return((exp(z) - exp(-z))/(exp(z) + exp(-z)))  
    }else {print('Invalid activation function!')}
  }
  
  #get first-order derivative for the hidden layer activation functions
  der_act = function(act.fun, A){
    if(act.fun == 'relu'){d = A/A; d[which(is.nan(d))] = 0; return(d)
    }else if(act.fun == 'linear'){return(abs(A)-abs(A)+1)  
    }else if (act.fun == 'sigmoid'){return(A*(1-A))
    }else if (act.fun == 'tanh'){return(1 - (A^2))  
    }else {print('Invalid activation function!')}
  }  
  
  #A function for initializing the weights and biases for the Backpropagation algorithm
  ini_par = function(n_h, n, p, y_n){
    W = list()
    W[[1]] = matrix(runif(p * n_h[1], 0, 1), nrow = p, ncol = n_h[1], byrow = T)*0.01
    for (i in 1:length(n_h)){
      if (length(n_h) - i!=0){W[[i+1]] = matrix(runif(n_h[i]*n_h[i+1], 0, 1), nrow = n_h[i], ncol = n_h[i+1], byrow = T)*0.01
      }else{W[[i+1]] = matrix(runif(n_h[i] * y_n, 0, 1), nrow = n_h[i], ncol = y_n, byrow = T)*0.01}
    }
    
    b = list()
    
    for(i in 1:(length(n_h)+1)){
      if (length(n_h) - i>=0){b[[i]] = rep(0, n_h[i])
      }else{
        b[[i]] = c(rep(0,y_n))
      }
    }
    
    pars = c(W, b)
    return(pars)
  }
  
  #The foward propagation function 
  foward_prop = function(X, pars,n_h, n, p, y_n, act_h, act_o){
    W = list()
    b = list()
    for (i in 1:(length(pars)/2)){
      W[[i]] = pars[[i]]
      b[[i]] = pars[[i+length(pars)/2]]
    }
    
    Z = list()
    Z[[1]] = X%*%W[[1]] + matrix(rep(b[[1]],n), nrow = n, ncol = length(b[[1]]), byrow = T)
    Z[[1]] = act(act.fun = act_h[1], z = Z[[1]])
    for (i in 2:length(W)){
      if(length(n_h)-i>=0){Z[[i]] = Z[[i-1]]%*%W[[i]] + matrix(rep(b[[i]],n), nrow = n, ncol = length(b[[i]]), byrow = T)
      Z[[i]] = act(act.fun = act_h[i],z = Z[[i]])
      }else{
        Z[[i]]  = Z[[i-1]]%*%W[[i]] + matrix(rep(b[[i]],n), nrow = n, ncol = length(b[[i]]), byrow = T)
        Z[[i]] = act(act.fun = act_o, z = Z[[i]])
      }
    }
    return (Z)
  }
  
  #Get the first-order derivative of the cost function  
  computeCost <- function(y, Z, pars) {
    w2 = NULL
    for (i in 1:(length(pars)/2)){
      w2 = cbind(w2, t(c(diag(crossprod (pars[[i]])))))
    }
    penalty = 1/2*lambda*sum(w2)
    n <- dim(y)[1]
    penalty = t(t(rep(penalty, n)))
    
    y_hat <- Z[[length(n_h)+1]]
    dL <- (2*(y_hat - y) + penalty)
    return (dL)
  }
  
  ########################################################
  #First step: Initializing parameters
  pars = ini_par(n_h=n_h, n=n, p=p, y_n=y_n)
  
  #Second step: Fowarded propagation with initialized parameters 
  Z = foward_prop(X, pars,n_h, n, p, y_n, act_h, act_o)
  
  #Third step: Compute the cost function
  dL = computeCost(y = y, Z = Z, pars = pars)
  
  #Storing paramters initialized in the first step
  W = list()
  b = list()
  for (i in 1:(length(pars)/2)){
    W[[i]] = pars[[i]]
    b[[i]] = pars[[i+length(pars)/2]]
    
  }  
  
  #creating a list for storing biases and weights momentum parameters
  vb = list()
  vw = list()
  for (i in length(W):1){
    vw[[length(W)-i+1]] = pars[[i]]*0
    vb[[length(W)-i+1]] = pars[[i+length(pars)/2]]*0
  }
  
  #get the user-specified constant for the learning rate
  lr = lr
  
  acc = NULL
  mse = NULL
  epochs = epochs
  
  
  #Fourth step: Begin Backpropagation 
  for (i in 1:epochs){
    K = list()
    K[[1]] = tcrossprod(dL,W[[length(W)]])*der_act(act_h[length(W)-1], Z[[length(W)-1]])
    if (length(n_h)>1){
      for (j in 2:(length(W)-1)){
        K[[j]] =  tcrossprod(K[[j-1]],W[[length(W)+1-j]])*der_act(act_h[length(W)-j], Z[[length(W)-j]])
      }
    }
    dw = list()
    dw[[1]] = 1/n*crossprod(Z[[length(W)-1]],dL*der_act(act_o, Z[[length(W)]]))
    
    for (j in 2:(length(W))){
      if (j < length(W)){dw[[j]] =  1/n*crossprod(Z[[length(W)-j]], K[[j-1]])
      }else{dw[[j]] =  1/n*crossprod(X, K[[length(W)-1]])}}
    
    db = list()
    db[[1]] = 1/n*crossprod(matrix(rep(1,n),n,1), dL*der_act(act_o, Z[[length(W)]]))
    for (j in 2:(length(b))){
      db[[j]] =  1/n*matrix(rep(1,n),1,n)%*%K[[j-1]]
    }
    
    for (j in 1:length(W)){
      W[[j]] = (1-lambda*lr)*W[[j]] - (momentum*vw[[length(W)+1-j]] + lr*dw[[length(W)+1-j]])
      b[[j]] = b[[j]] - (momentum*vb[[length(b)+1-j]] + lr*db[[length(b)+1-j]])
    }
    
    #Store new parameters
    pars = c(W,b)  
    
    #Feed the neural network with new parameters
    new = foward_prop(X = X, pars = pars, n_h, n, p, y_n, act_h, act_o)
    Z = new
    #Compute the cost function given the new parameters
    dL = computeCost(y = y, Z = Z, pars = pars)
    #get predicted values given the new parameters  
    y_hat = Z[[length(W)]]
    #Store the new values for the momentum  
    for (j in 1:length(vw)){
      vw[[j]] = (momentum*vw[[j]] + lr*(dw[[j]]))
      vb[[j]] = (momentum*vb[[j]] + lr*(db[[j]]))
    }
    
    #Compute model godness-of-fitting   
    acc[i] = cor(y_hat, y)
    mse[i] = mean((y_hat - y)^2)
    #Print the actual iteration
    cat('epoch:',i,'|', 'mse --->', mse[i],'\n');
    if(is.na(acc[i])){
      cat("Stop! Unstable training process, try to use different values for lr and/or momentum", '\n')
      break()  
    }
    
  }
  #end of the backpropagation
  
  return(list(acc, mse, pars, n_h, act_h, act_o))
}
####################end of train.nn ########################################

#Use a trained MLP object for predicting unobserved values
#Xtest: a n x p matrix with new input variables
#model: a train.nn object
predict.nn = function(Xtest, model){
  n_h = model[[4]]
  act_h = model[[5]]
  act_o = model[[6]]
  pars = model[[3]]
  n = dim(Xtest)[1]
  p = dim(Xtest)[2]
  out = length(pars)
  y_n = length(pars[[out]])
  
  act = function(act.fun, z){
    if(act.fun == 'relu'){z[which(z<0)] = 0;return(z)
    }else if(act.fun == 'linear'){return(z)  
    }else if (act.fun == 'sigmoid'){return(1/(1+exp(-z)))
    }else if (act.fun == 'tanh'){return((exp(z) - exp(-z))/(exp(z) + exp(-z)))  
    }else {print('Invalid activation function!')}
  }
  
  foward_prop = function(X, pars,n_h, n, p, y_n, act_h, act_o){
    W = list()
    b = list()
    for (i in 1:(length(pars)/2)){
      W[[i]] = pars[[i]]
      b[[i]] = pars[[i+length(pars)/2]]
    }
    
    Z = list()
    Z[[1]] = X%*%W[[1]] + matrix(rep(b[[1]],n), nrow = n, ncol = length(b[[1]]), byrow = T)
    Z[[1]] = act(act.fun = act_h[1], z = Z[[1]])
    for (i in 2:length(W)){
      if(length(n_h)-i>=0){Z[[i]] = Z[[i-1]]%*%W[[i]] + matrix(rep(b[[i]],n), nrow = n, ncol = length(b[[i]]), byrow = T)
      Z[[i]] = act(act.fun = act_h[i],z = Z[[i]])
      }else{
        Z[[i]]  = Z[[i-1]]%*%W[[i]] + matrix(rep(b[[i]],n), nrow = n, ncol = length(b[[i]]), byrow = T)
        Z[[i]] = act(act.fun = act_o, z = Z[[i]])
      }
    }
    return (Z)
  }
  
  Zhat = foward_prop(Xtest, pars,n_h, n, p, y_n, act_h, act_o)
  y_hat <- Zhat[[length(n_h)+1]]
  return(y_hat)
}
########################end of nn.predict############################
#END
