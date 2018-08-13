library(tidyverse)
library(caret)
library(randomForest)
library(rpart)
library(xgboost)
library(gbm)
library(glmnet)
library(nnet)

# Any input dataset must be "tibble" and contain a column named "Y", a column named "D" 
# and a column named "Z" if instrumental variable is used.
# If D or Z is not of the type required, some MLmethods need change setting to keep valid.

# MachineLearningmethods are: 
# lasso, regression tree, randomforest, boosting, boosting2, neuralnet;
# they are implemented by R-package:
# glmnet, rpart, randomForest, xgboost, gbm, nnet,
# respectively.

# Interactive Regression Model, for binary D
IRMestimator <- function(data, k, MLmethod, DML = 2, nsplit = 100, mergemethod = "median"){
  t0 <- proc.time()
  N <- nrow(data)
  X <- select(data, -D, -Y)
  XD <- select(data, -Y)
  thetalst <- c()
  sd1lst <- c()
  for(s in 1:nsplit){
    folds <- createFolds(data[,"Y", drop = TRUE], k = k)
    psilst <- c(NA, NA)
    for(i in 1:k){
      idx <- folds[[i]]
      dtest <- data[idx, "D", drop = TRUE]
      dtrain <- data[-idx, "D", drop = TRUE]
      ytest <- data[idx, "Y", drop = TRUE]
      ytrain <- data[-idx, "Y", drop = TRUE]
      if(MLmethod == "randomForest"){
        Dforest <- randomForest(X[-idx,], y = dtrain, ntree = 1000)
        m <- predict(Dforest, X[idx,])
        Yforest <- randomForest(XD[-idx,], y = ytrain, ntree = 1000)
        g1 <- predict(Yforest, mutate(X[idx,],D = 1))
        g0 <- predict(Yforest, mutate(X[idx,],D = 0))
      } else if(MLmethod == "Reg.tree"){
        Dtree <- rpart(D~., data = XD[-idx,], method = "class",
                       control = rpart.control(minbucket = 5,cp = 0.001))
        m <- predict(Dtree, X[idx,])[,2]
        m <- 0.9*m + 0.05
        Ytree <- rpart(Y~., data = data[-idx,])
        g1 <- predict(Ytree, mutate(X[idx,],D = 1))
        g0 <- predict(Ytree, mutate(X[idx,],D = 0))
      } else if(MLmethod == "Boosting"){
        Dboost <- xgboost(data = as.matrix(X[-idx,]), 
                          label = dtrain, nrounds = 100, 
                          objective = "binary:logistic", 
                          max_depth = 2, eta = 1, nthread = 2)
        m <- predict(Dboost, as.matrix(X[idx,]))
        Yboost <- xgboost(data = as.matrix(XD[-idx,]), label = ytrain, 
                          nrounds = 100, max_depth = 2, eta = 1, nthread = 2)
        g1 <- predict(Yboost, as.matrix(mutate(X[idx,],D = 1)))
        g0 <- predict(Yboost, as.matrix(mutate(X[idx,],D = 0)))
      } else if(MLmethod == "Boosting2"){
        Dboost <- gbm(D~., data = XD[-idx,], cv.folds = 10, distribution = "huberized")
        m <- predict(Dboost, XD[idx,], n.trees = gbm.perf(Dboost, plot.it = FALSE))
        Yboost <- gbm(Y~., data = XY[-idx,], cv.folds = 10)
        g1 <- predict(Yboost, as.matrix(mutate(X[idx,],D = 1)), 
                      n.trees = gbm.perf(Yboost, plot.it = FALSE))
        g0 <- predict(Yboost, as.matrix(mutate(X[idx,],D = 0)), 
                      n.trees = gbm.perf(Yboost, plot.it = FALSE))
      } else if(MLmethod == "Lasso"){
        Dlambda <- cv.glmnet(as.matrix(X[-idx,]), dtrain, 
                             family = "binomial", type.measure = "auc")$lambda.min
        Dlasso <- glmnet(as.matrix(X[-idx,]), dtrain, lambda = Dlambda)
        m <- predict(Dlasso, as.matrix(X[idx,]))
        Ylambda <- cv.glmnet(as.matrix(XD[-idx,]), ytrain)$lambda.min
        Ylasso <- glmnet(as.matrix(XD[-idx,]), ytrain, lambda = Ylambda)
        g1 <- predict(Ylasso, as.matrix(mutate(X[idx,],D = 1)))
        g0 <- predict(Ylasso, as.matrix(mutate(X[idx,],D = 0)))
      } else if(MLmethod == "NeuralNet"){
        Dnet <- nnet(D~., data = XD[-idx,], size = 2, decay = 0.02)
        m <- predict(Dnet, X[idx,])
        Ynet <- nnet(Y~., data = data[-idx,], size = 2, decay = 0.02, linout = TRUE)
        g1 <- predict(Ynet, as.matrix(mutate(X[idx,],D = 1)))
        g0 <- predict(Ynet, as.matrix(mutate(X[idx,],D = 0)))
      }
      psi_b <- g1 - g0 + dtest*(ytest - g1)/m - (1 - dtest)*(ytest - g0)/(1 - m)
      psilst <- rbind(psilst, c(mean(psi_b), mean(psi_b^2)))
    }
    psilst <- psilst[-1, drop = FALSE]
    epsi <- apply(psilst, 2, mean)
    if(DML == 2) theta_s <- -epsi[1]
    else if(DML == 1) theta_s <- -epsi[1]
    sd1_s <- sqrt((epsi[2] - epsi[1]^2)/N)
    thetalst <- c(thetalst, theta_s)
    sd1lst <- c(sd1lst, sd1_s)
  }
  if(mergemethod == "median"){
  theta <- median(thetalst)
  sd1 <- median(sd1lst)
  sd2 <- sqrt(median(sd1lst^2 + (thetalst - theta)^2))
  }
  if(mergemethod == "mean"){
    theta <- mean(thetalst)
    sd1 <- mean(thetalst)
    sd2 <- sqrt(mean(sd1lst^2 + (thetalst - theta)^2))
  }
  t1 <- proc.time()
  list("theta" = theta, "sd1" = sd1, "sd2" = sd2, "time" = t1 - t0)
}


# Partial Linear Regression Model, for binary D
PLRMestimator <- function(data, k, MLmethod, DML = 2, nsplit = 100, mergemethod = "median"){
  t0 <- proc.time()
  N <- nrow(data)
  X <- select(data, -D, -Y)
  XD <- select(data, -Y)
  XY <- select(data, -D)
  thetalst <- c()
  sd1lst <- c()
  for(s in 1:nsplit){
    folds <- createFolds(data[,"Y", drop = TRUE], k = k)
    psilst <- c(NA, NA, NA, NA, NA)
    for(i in 1:k){
      idx <- folds[[i]]
      dtrain <- data[-idx, "D", drop = TRUE]
      dtest <- data[idx, "D", drop = TRUE]
      ytrain <- data[-idx, "Y", drop = TRUE]
      ytest <- data[idx, "Y", drop = TRUE]
      if(MLmethod == "randomForest"){
        Dforest <- randomForest(X[-idx,], y = dtrain, ntree = 1000)
        m <- predict(Dforest, X[idx,])
        Yforest <- randomForest(X[-idx,], y = ytrain, ntree = 1000)
        l <- predict(Yforest, X[idx,])
      } else if(MLmethod == "Reg.tree"){
        Dtree <- rpart(D~., data = XD[-idx,], method = "class",
                       control = rpart.control(minbucket = 5,cp = 0.001))
        m <- predict(Dtree, X[idx,])[,2]
        m <- 0.9*m + 0.05
        Ytree <- rpart(Y~., data = XY[-idx,])
        l <- predict(Ytree, X[idx,])
      } else if(MLmethod == "Boosting"){
        Dboost <- xgboost(data = as.matrix(X[-idx,]), label = dtrain, 
                          nrounds = 100, objective = "binary:logistic", 
                          max_depth = 2, eta = 1, nthread = 2)
        m <- predict(Dboost, as.matrix(X[idx,]))
        Yboost <- xgboost(data = as.matrix(X[-idx,]), label = ytrain, 
                          nrounds = 100, max_depth = 2, eta = 1, nthread = 2)
        l <- predict(Yboost, as.matrix(X[idx,]))
      } else if(MLmethod == "Boosting2"){
        Dboost <- gbm(D~., data = XD[-idx,], cv.folds = 10, distribution = "huberized")
        m <- predict(Dboost, XD[idx,], n.trees = gbm.perf(Dboost, plot.it = FALSE))
        Yboost <- gbm(Y~., data = XY[-idx,], cv.folds = 10)
        l <- predict(Yboost, XY[idx,], n.trees = gbm.perf(Yboost, plot.it = FALSE))
      } else if(MLmethod == "Lasso"){
        Dlambda <- cv.glmnet(as.matrix(X[-idx,]), dtrain, 
                             family = "binomial", type.measure = "auc")$lambda.min
        Dlasso <- glmnet(as.matrix(X[-idx,]), dtrain, lambda = Dlambda)
        m <- predict(Dlasso, as.matrix(X[idx,]))
        Ylambda <- cv.glmnet(as.matrix(X[-idx,]), ytrain)$lambda.min
        Ylasso <- glmnet(as.matrix(X[-idx,]), ytrain, lambda = Ylambda)
        l <- predict(Ylasso, as.matrix(X[idx,]))
      } else if(MLmethod == "NeuralNet"){
        Dnet <- nnet(D~., data = XD[-idx,], size = 2, decay = 0.02)
        m <- predict(Dnet, X[idx,])
        Ynet <- nnet(Y~., data = XY[-idx,], size = 2, decay = 0.02, linout = TRUE)
        l <- predict(Ynet, X[idx,])
      }
      psi_b <- (ytest - l)*(dtest - m)
      psi_a <- -(dtest - m)^2
      psilst <- rbind(psilst, c(mean(psi_a), mean(psi_b), 
                                mean(psi_a^2), mean(psi_b^2), mean(psi_a*psi_b)))
    }
    psilst <- psilst[-1,]
    epsi <- apply(psilst, 2, mean)
    if(DML == 2) theta_s <- -epsi[2]/epsi[1]
    else if(DML == 1) theta_s <- mean(-psilst[,2]/psilst[,1])
    sd1_s <- sqrt((epsi[3]*theta_s^2 + epsi[4] + 2*epsi[5]*theta_s)/(epsi[1]^2)/N)
    thetalst <- c(thetalst, theta_s)
    sd1lst <- c(sd1lst, sd1_s)
  }
  if(mergemethod == "median"){
  theta <- median(thetalst)
  sd1 <- median(sd1lst)
  sd2 <- sqrt(median(sd1lst^2 + (thetalst - theta)^2))
  }
  if(mergemethod == "mean"){
    theta <- mean(thetalst)
    sd1 <- mean(thetalst)
    sd2 <- sqrt(mean(sd1lst^2 + (thetalst - theta)^2))
  }
  t1 <- proc.time()
  list("theta" = theta, "sd1" = sd1, "sd2" = sd2, "time" = t1 - t0)
}


# Partial Linear IV Regression Model, for continuous D & continuous Z
PLIVRMestimator <- function(data, k, MLmethod, DML = 2, nsplit = 100, mergemethod = "median"){
  t0 <- proc.time()
  N <- nrow(data)
  X <- select(data, -D, -Y, -Z)
  XD <- elect(data, -Y, -Z)
  XY <- select(data, -D, -Z)
  XZ <- select(data, -D, -Y)
  thetalst <- c()
  sd1lst <- c()
  for(s in 1:nsplit){
    folds <- createFolds(data[,"Y", drop = TRUE], k = k)
    psilst <- c(NA, NA, NA, NA, NA)
    for(i in 1:k){
      idx <- folds[[i]]
      dtest <- data[idx, "D", drop = TRUE]
      ytest <- data[idx, "Y", drop = TRUE]
      ztest <- data[idx, "Z", drop = TRUE]
      dtrain <- data[id-x, "D", drop = TRUE]
      ytrain <- data[-idx, "Y", drop = TRUE]
      ztrain <- data[-idx, "Z", drop = TRUE]
      if(MLmethod == "randomForest"){
        Dforest <- randomForest(X[-idx,], y = dtrain, ntree = 1000)
        r <- predict(Dforest, X[idx,])
        Yforest <- randomForest(X[-idx,], y = ytrain, ntree = 1000)
        l <- predict(Yforest, X[idx,])
        Zforest <- randomForest(X[-idx,], y = ztrain, ntree = 1000)
        m <- predict(Zforest, X[idx,])
      } else if(MLmethod == "Reg.tree"){
        Dtree <- rpart(D~., data = XD[-idx,])
        r <- predict(Dtree, X[idx,])
        Ytree <- rpart(Y~., data = XY[-idx,])
        l <- predict(Ytree, X[idx,])
        Ztree <- rpart(Z~., data = XZ[-idx,])
        m <- predict(Ztree, X[idx,])
      } else if(MLmethod == "Boosting"){
        Dboost <- xgboost(data = as.matrix(X[-idx,]), label = dtrain, nrounds = 100)
        r <- predict(Dboost, as.matrix(X[idx,]))
        Yboost <- xgboost(data = as.matrix(X[-idx,]), label = ytrain, nrounds = 100)
        l <- predict(Yboost, as.matrix(X[idx,]))
        Zboost <- xgboost(data = as.matrix(X[-idx,]), label = ztrain, nrounds = 100)
        m <- predict(Zboost, as.matrix(X[idx,]))
      } else if(MLmethod == "Boosting2"){
        Dboost <- gbm(D~., data = XD[-idx,], cv.folds = 10)
        r <- predict(Dboost, XD[idx,], n.trees = gbm.perf(Dboost, plot.it = FALSE))
        Yboost <- gbm(Y~., data = XY[-idx,], cv.folds = 10)
        l <- predict(Yboost, XY[idx,], n.trees = gbm.perf(Yboost, plot.it = FALSE))
        Zboost <- gbm(Z~., data = XZ[-idx,], cv.folds = 10)
        m <- predict(Zboost, XZ[idx,], n.trees = gbm.perf(Zboost, plot.it = FALSE))
      } else if(MLmethod == "Lasso"){
        Dlambda <- cv.glmnet(as.matrix(X[-idx,]), dtrain)$lambda.min
        Dlasso <- glmnet(as.matrix(X[-idx,]), dtrain, lambda = Dlambda)
        r <- predict(Dlasso, as.matrix(X[idx,]))
        Ylambda <- cv.glmnet(as.matrix(X[-idx,]), ytrain)$lambda.min
        Ylasso <- glmnet(as.matrix(X[-idx,]), ytrain, lambda = Ylambda)
        l <- predict(Ylasso, as.matrix(X[idx,]))
        Zlambda <- cv.glmnet(as.matrix(X[-idx,]), ztrain)$lambda.min
        Zlasso <- glmnet(as.matrix(X[-idx,]), ztrain, lambda = Zlambda)
        m <- predict(Zlasso, as.matrix(X[idx,]))
      } else if(MLmethod == "NeuralNet"){
        Dnet <- nnet(D~., data = XD[-idx,], size = 2, decay = 0.02, linout = TRUE)
        r <- predict(Dnet, X[idx,])
        Ynet <- nnet(Y~., data = XY[-idx,], size = 2, decay = 0.02, linout = TRUE)
        l <- predict(Ynet, X[idx,])
        Znet <- nnet(Z~., data = XZ[-idx,], size = 2, decay = 0.02, linout = TRUE)
        m <- predict(Znet, X[idx,])
      }
      psi_b <- (ytest - l)*(ztest - m)
      psi_a <- -(dtest - r)*(ztest - m)
      psilst <- rbind(psilst, c(mean(psi_a), mean(psi_b), 
                                mean(psi_a^2), mean(psi_b^2), mean(psi_a*psi_b)))
    }
    psilst <- psilst[-1,]
    epsi <- apply(psilst, 2, mean)
    if(DML == 2) theta_s <- -epsi[2]/epsi[1]
    else if(DML == 1) theta_s <- mean(-psilst[,2]/psilst[,1])
    sd1_s <- sqrt((epsi[3]*theta_s^2 + epsi[4] + 2*epsi[5]*theta_s)/(epsi[1]^2)/N)
    thetalst <- c(thetalst, theta_s)
    sd1lst <- c(sd1lst, sd1_s)
  }
  if(mergemethod == "median"){
  theta <- median(thetalst)
  sd1 <- median(sd1lst)
  sd2 <- sqrt(median(sd1lst^2 + (thetalst - theta)^2))
  }
  if(mergemethod == "mean"){
    theta <- mean(thetalst)
    sd1 <- mean(thetalst)
    sd2 <- sqrt(mean(sd1lst^2 + (thetalst - theta)^2))
  }
  t1 <- proc.time()
  list("theta" = theta, "sd1" = sd1, "sd2" = sd2, "time" = t1 - t0)
}

######### running part ###########


# # reading PUBE(Pennsylvania Unemployment Bonus Experiment) data
# recsfile <- read_csv('recsfile.txt')
# data1 <- recsfile %>% 
#   filter(dem_inel==1, revasamp==1) %>%
#   mutate(lusd = site1+site4+site12,
#          durable = ifelse(sic %in% c(24, 25) | (31<sic & sic<40), 1, 0),
#          logdur = log(inuidur1)) %>%
#   select("agelt35", "agegt54", "black", "hispanic", "othrace", "female", 
#          "recall", "dep", "lusd", "durable", "logdur", "q1", "q2", 
#          "q3", "q4", "q5", "t1", "t2","t3", "t4", "t5", "t6") %>%
#   na.omit %>%
#   filter(t1==0, t2==0, t3==0, t5==0, t6==0) %>%
#   select(-t1, -t2, -t3, -t5, -t6) %>%
#   rename(D = t4, Y = logdur)

counts <- 1
reslst <- list()
for(k in c(2, 5)){
  for(MLmethod in c("Lasso", "NeuralNet","Boosting", "Reg.tree", "randomForest")){
    reslst[[counts]] <- IRMestimator(data1, k, MLmethod)
    counts <- counts + 1
  }
}
print(reslst)


# # reading Institutions Effect data(raw data were pre-processed in Stata)
# data2 <- read_csv("maketable2.csv")

# counts <- 1
# reslst <- list()
# for(k in c(2, 5)){
#   for(MLmethod in c("Lasso", "NeuralNet","Boosting", "Reg.tree", "randomForest")){
#     reslst[[counts]] <- PLIVRMestimator(data2, k, MLmethod)
#     counts <- counts + 1
#   }
# }
# print(reslst)