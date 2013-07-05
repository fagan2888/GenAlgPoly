#########################################################################
# initial values, can be overriden
max.monoids <- 4  # poly = monoid1 + monoid2 + ...
max.degree  <- 3  # meaning exponents up to 2^max.degree - 1

#########################################################################
# Root mean square error
rsme <- function(v1, v2=c(0)) {
  sqrt(mean((v1-v2)^2))
}

#########################################################################
# convert a binary vector into an integer
bin2dec <- function(x) 
  sum(2^(which(rev(unlist(strsplit(as.character(x), "")) == 1))-1))

#########################################################################
# The chromosome coding will as follow:
# + an initial segment detailing which monoids are active (the 1st is always active)
# + the chromosome is split into `max.monoids` sets of bits of equal size 
#   (a monoid)
# + each monoid is split into `n.vars` sets of `max.degree` size each 
#   (a variable)
# + for each variable, the bits give the binary description of the variable's degree 
# 
# Eg: consider polynomial $x_1^3 \times x_3 + x_3^7 + x_1 \times x_2$
# 
# One possible coding -- for the values in the previous code snippet -- would be:
# 
# $$ 1111 011,000,001;000,000,111;001,001,000;000,000,000$$
# 
# (the semicolons separate monoids, the commas separate variables)
# 
# The next two functions transform the binary vector into a polynomial formula 
# that `lm()` understands:

make.monoid <- function(bits, n.vars, vars) {
  elements <- split(bits, ceiling(seq_along(bits)/(length(bits)/n.vars)))
  monoid <- ""
  for(i in 1:length(elements)) {
    power <- bin2dec(elements[[i]])
    if (power == 1)
      if (monoid=="")
        monoid = vars[i]
      else  
        monoid = paste0(monoid, "*", vars[i])
    if (power > 1)
      if (monoid=="")
        monoid = paste0(vars[i], "^", power)
      else  
        monoid = paste0(monoid, "*", vars[i], "^", power)
  }
  monoid
}

make.formula <- function(bits, n.vars, max.monoids) {
  vars          <- paste0("x",1:n.vars)    # c("x1","x2"...)
  active.bits   <- 1:(max.monoids-1)       # the initial activation bits
  active.monoid <- c(1,bits[active.bits])  # get those bits
  bits          <- bits[-active.bits]      # the remainder is the info about the monoids
  monoids       <- split(bits, ceiling(seq_along(bits)/(length(bits)/max.monoids)))
  formula <- ""
  for(i in 1:length(monoids)) {
    if (active.monoid[i]==1) {
      monoid <- make.monoid(monoids[[i]], n.vars, vars)
      if (monoid!="")
        if (formula=="")
          formula = paste0("I(", monoid, ")")
        else
          formula = paste0(formula," + I(", monoid,")")
    }
  }
  if (formula=="")     # the extreme case where the polynomial is empty
    formula = "I(x1)"  # default polynomial formula (by convention)
  formula
}

#########################################################################
# This is a factory for the evaluation function that the GA function
# needs to execute. 

# The factory is a closure that includes the dataframe, the number of
# variable and the max number of componentes, all required information
# to perform the polynomial regression and then compute the residuals' rsme 
# which will be the fitness of the respective chromosome
evalFuncFactory <- function(df, n.vars, max.monoids, lambda=1.05) {
  
  function(chromosome) {
    formula <- paste0("y ~ ", make.formula(chromosome, n.vars, max.monoids))
    model <- lm(formula, data=df)
    # for regularization: find how many monoids are there:
    n.monoids <- length(strsplit(formula,"[+]")[[1]]) + 1 # check how many '+' are there
    # rbga.bin() minimizes, so the rmse will be smaller as the interations advance
    return( sqrt(mean(residuals(model)^2)) * lambda^n.monoids ) 
  }
}

monitorEvalFactory <- function(n.vars, max.monoids) {

  function(obj) {
    minEval <- min(obj$evaluations)
    filter  <- obj$evaluations == minEval
    bestObjectCount = sum(rep(1, obj$popSize)[filter]);
    # ok, deal with the situation that more than one object is best
    if (bestObjectCount > 1) {
      bestSolution = obj$population[filter,][1,];
    } else {
      bestSolution = obj$population[filter,];
    }
    print(bestSolution)
    print(make.formula(bestSolution, n.vars, max.monoids))
  }
}

#########################################################################
# Returns a dataset with the rsme for each one of the tested ML methods
# including GA Poly

# pre: both sets must have columns named x1...xn and the last one is y (the output)
make.report <- function(my.data, 
                        n.vars,
                        population=100, 
                        iterations=25, 
                        runs=10, 
                        mutation.rate=0.05) {
  
  library(genalg)
  library(e1071)
  library(rpart)   
  library(randomForest)
  library(party)
  
  train.p.size <- 0.7 # percentage of training set
  n.vars       <- ncol(my.data)-1
  
  ga.error     <- rep(0,runs)
  lm.error     <- rep(0,runs)
  svm.error    <- rep(0,runs)
  rpart.error  <- rep(0,runs)
  rf.error     <- rep(0,runs)
  citree.error <- rep(0,runs)
  
  best.ga.model <- NULL
  best.ga.error <- 10e10
  
  for (i in 1:runs) {
    
    # make train & test set
    inTrain   <- sample(1:nrow(my.data), train.p.size * nrow(my.data))
    train.set <- my.data[inTrain,]
    test.set  <- my.data[-inTrain,]
    
    GAmodel <- rbga.bin(size = max.monoids + max.degree*n.vars*max.monoids, 
                        popSize = population, 
                        iters = iterations, 
                        mutationChance = mutation.rate, 
                        elitism = TRUE, 
                        evalFunc = evalFuncFactory(train.set, n.vars, max.monoids),
                        monitorFunc = monitorEvalFactory(n.vars, max.monoids))
      
    best.solution <- GAmodel$population[1,]
    best.formula <- paste0("y ~ ", make.formula(best.solution, n.vars, max.monoids))
    ga.model <- lm(best.formula, data=my.data)
    ga.pred  <- predict(ga.model, test.set[,-ncol(test.set)]) 
    ga.error[i] <- rsme(ga.pred,test.set[,ncol(test.set)])
    
    if(ga.error[i] < best.ga.error) { # save best ga.model
      best.ga.model <- ga.model
      best.ga.error <- ga.error[i]
    }
    
    lm.model <- lm(y ~ ., data = train.set)
    lm.pred  <- predict(lm.model, test.set[,-ncol(test.set)]) 
    lm.error[i] <- rsme(lm.pred,test.set[,ncol(test.set)])
    
    
    svm.model <- svm(y ~ ., data=train.set, kernel='linear')
    svm.pred  <- predict(svm.model, test.set[,-ncol(test.set)]) 
    svm.error[i] <- rsme(svm.pred,test.set[,ncol(test.set)])

    rpart.model <- rpart(y ~ ., data=train.set)
    rpart.pred  <- predict(rpart.model, test.set[,-ncol(test.set)]) 
    rpart.error[i] <- rsme(rpart.pred,test.set[,ncol(test.set)])

    rf.model <- randomForest(y ~ ., data=train.set, importance=TRUE, do.trace=100, ntree=100)
    rf.pred  <- predict(rf.model, test.set[,-ncol(test.set)]) 
    rf.error[i] <- rsme(rf.pred,test.set[,ncol(test.set)])

    citree.model <- ctree(y ~ ., data = train.set)
    citree.pred  <- predict(citree.model, test.set[,-ncol(test.set)]) 
    citree.error[i] <- rsme(citree.pred,test.set[,ncol(test.set)])    
  }
  
  list(ga.error=ga.error,         # make the errors report into a list
       lm.error=lm.error,
       svm.error=svm.error,
       rpart.error=rpart.error,
       rf.error=rf.error,
       citree.error=citree.error,
       ga.model = best.ga.model)  # also include the best found ga.model
}

#########################################################################
# Testing different lamdba values (the regularization coefficient)
# it returns a matrix of errors for each lambda value for a number
# of given runs

test.lambda <- function(my.data, lambda.values, 
                        runs=5, 
                        population=100, 
                        iterations=20, 
                        mutation.rate=0.05,
                        verbose=TRUE) {

  library(genalg)

  train.p.size <- 0.7 # percentage of training set
  n.vars       <- ncol(my.data)-1
  
  errors <- matrix( rep(0,length(lambda.values)*runs), ncol= length(lambda.values))
  
  for (k in 1:length(lambda.values)) {  
    
    lambda   <- lambda.values[k]
    
    for (i in 1:runs) {  # execute 'runs' times for each value of lambda
      
      # make train & test set
      inTrain   <- sample(1:nrow(my.data), train.p.size * nrow(my.data))
      train.set <- my.data[inTrain,]
      test.set  <- my.data[-inTrain,]
      
      GAmodel <- rbga.bin(size = max.monoids + max.degree*n.vars*max.monoids, 
                          popSize = population, 
                          iters = iterations, 
                          mutationChance = mutation.rate, 
                          elitism = TRUE, 
                          evalFunc = evalFuncFactory(train.set, n.vars, max.monoids, lambda))
      
      best.solution <- GAmodel$population[1,]
      best.formula <- paste0("y ~ ", make.formula(best.solution, n.vars, max.monoids))
      ga.model <- lm(best.formula, data=my.data)
      ga.pred  <- predict(ga.model, test.set[,-ncol(test.set)]) 

      errors[i,k] <- rsme(ga.pred,test.set[,ncol(test.set)])
    }
    if (verbose)
      print(paste0("#lambda: ",k))
  }
  
  errors  
}
