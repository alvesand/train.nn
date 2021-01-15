############################################################
#Toy example
############################################################

#CREATING THE INPUT MATRIX
X = matrix(NA, 3000, 100)

set.seed(182)
for (i in 1:dim(X)[1]){
  X[i,] <- runif(100, 0, 1)
}
#########################

#Simulate the responses vector
set.seed(182)
y = (1-exp(X[,2])^2) + X[,2] -  X[,10]^2*(.2) + (X[,2]*X[,3]*-3.4)+ 
  X[,50]^3*(-0.9)+X[,1]*X[,9]+X[,5]^2*3+tan(X[,9])*X[,7]*(2) - exp(X[,100])*X[,10]*3.2 + rnorm(3000, 0,2)

#A simple function for normalizing y
normalize <- function(x) {
  num <- x - min(x)
  denom <- max(x) - min(x)
  return (num/denom)
}

#get y normalized
y_norm = as.matrix(normalize(y),3000,1)
hist(y_norm)

#split the training data
xtrain = X[1:2500,]
ytrain = as.matrix(y_norm[1:2500,1],2500,1)

#Training a MLP with the train.nn function
#2 hidden layers
#10 and 5 neurons in the first and second hidden layers
#Activaction function = Rectified Linear Unit

start.time <- Sys.time()
mod1 = train.nn(X = xtrain, y = ytrain, neurons = c(10,5), act_h = c('relu', 'relu'), act_o = 'linear',  epochs = 1000, lr = 0.01, momentum = 0.97, lambda = 0)
end.time <- Sys.time()
print(end.time - start.time) #Time to run: 11.26 secs

#get the training history
acc1 = mod1[[1]]
mse1 = mod1[[2]]

#plot the training history
plot(acc1,type = 'l')
plot(mse1,type = 'l')

#Predict unobserved data
yhat_nn = predict.nn(X[2501:3000,], mod1)
cor(y_norm[2501:3000], yhat_nn) #r = 0.781


#Running a similar model with keras
library(keras)
library(tensorflow)

model = keras_model_sequential()%>%
  layer_dense(units = 10, activation = 'relu', input_shape = c(100))%>%
  layer_dropout(rate = 0)%>%
  layer_dense(units = 5, activation = 'relu')%>%
  layer_dropout(rate = 0)%>%
  layer_dense(units = 1, activation = 'linear')

model%>%compile(loss = "mse",
                optimizer = 'sgd', lr = 0.01)

start.time2 <- Sys.time()
mod_DL <- model%>%fit(X[1:2500,], as.matrix(y_norm[1:2500,]), epochs = 1000, validation_split = 0, verbose = F)
end.time2 <- Sys.time()
print(end.time2 - start.time2) #Time to run: 2.74 mins

cor (y_norm[2501:3000], model%>%predict(X[2501:3000,])) #r = 0.786


