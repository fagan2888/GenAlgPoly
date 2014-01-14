######################################
# Tuning GAPoly parameters

source("gapolyfuncs.R")

########################################
# functions that compute next values (for each parameter)

values.mutation  <- seq(0.01,0.5,len=50)
next.mutation.rate <- function() { sample(values.mutation, 1) }

values.elitism  <- seq(0,0.1,len=11)
next.elitism    <- function() { sample(values.elitism, 1) }

values.lambda   <- seq(0,2,len=81)
next.lambda     <- function(){ sample(values.lambda,1) }

next.max.monoid <- function(){ sample(2:9,1) }

next.max.degree <- function(){ sample(2:4,1) }

########################################
# Random Grid Search
tune.gapoly.params <- function(my.data,population=100,iterations=50,runs=10) {
  
  #initialize parameters
  mutation.rate <- 0.05
  elitism.perc  <- 0.01
  lambda        <- 0.8
  max.monoids   <- 4
  max.degree    <- 3
  
  # prepare dataset
  train.p.size <- 0.7 # percentage of training set
  n.vars       <- ncol(my.data)-1
  
  # make train & test set (ie, each run has a different train+test sets)
  inTrain   <- sample(1:nrow(my.data), train.p.size * nrow(my.data))
  train.set <- my.data[inTrain,]
  test.set  <- my.data[-inTrain,]
  
  report <- data.frame(mutation=rep(NA,runs),
                       elitism=rep(NA,runs),
                       lambda=rep(NA,runs),
                       monoids=rep(NA,runs),
                       degree=rep(NA,runs),
                       error=rep(NA,runs),
                       model=rep(NA,runs))
  
  for(run in 1:runs) {
    # run GA.poly one time
    GAmodel <- rbga.bin(size = max.monoids + max.degree*n.vars*max.monoids, 
                        iters = iterations, 
                        popSize = population, 
                        mutationChance = mutation.rate, 
                        elitism = round(elitism.perc * population), 
                        evalFunc = evalFuncFactory(train.set, n.vars, max.monoids, lambda))
    
    best.solution <- GAmodel$population[1,]
    best.formula <- paste0("y ~ ", make.formula(best.solution, n.vars, max.monoids))
    ga.model <- lm(best.formula, data=train.set)
    ga.pred  <- predict(ga.model, test.set[,-ncol(test.set)]) 
    
    report[run,]$mutation <- mutation.rate
    report[run,]$elitism <- elitism.perc
    report[run,]$lambda <- lambda
    report[run,]$monoids <- max.monoids
    report[run,]$degree <- max.degree
    report[run,]$error <- rsme(ga.pred, test.set[,ncol(test.set)])
    report[run,]$model <- make.formula(best.solution, n.vars, max.monoids)
    
    # update parameters
    mutation.rate <- next.mutation.rate()
    elitism.perc  <- next.elitism()
    lambda        <- next.lambda()
    max.monoids   <- next.max.monoid()
    max.degree    <- next.max.degree()

    # show progress
    cat(paste0(" ",run,"."))
  }
  report
}

#######################################
# Test:

filename <- "Housing/housing.data"  # where is the data (under dataset folder)
filepath <- paste0("datasets/",filename)
my.data <- read.csv(filepath, header=FALSE)
names(my.data) <- c(paste0("x",1:(ncol(my.data)-1)),"y")
my.data <- as.data.frame(scale(my.data))
head(my.data,8)

report <- tune.gapoly.params(my.data, population=100, iterations=75, runs=50)
report

write.table(report, "grid.search.Housing2.txt")

# cf: 
# http://stats.stackexchange.com/questions/3328/given-a-10d-mcmc-chain-how-can-i-determine-its-posterior-modes-in-r

#######################################
# Iterative Search

# In this search, we first test for one parameter, find the best result, then
# pass to the next parameter, and so on until some convergence is seen

tune.gapoly.iterative <- function(my.data,population=100,iterations=50,runs=10) {
  
  #initialize parameters
  mutation.rate <- 0.05
  elitism.perc  <- 0.01
  lambda        <- 0.8
  max.monoids   <- 4
  max.degree    <- 3
  
  # prepare dataset
  train.p.size <- 0.7 # percentage of training set
  n.vars       <- ncol(my.data)-1
  
  # make train & test set (ie, each run has a different train+test sets)
  inTrain   <- sample(1:nrow(my.data), train.p.size * nrow(my.data))
  train.set <- my.data[inTrain,]
  test.set  <- my.data[-inTrain,]
  
  report.obs <- 2*runs
  report <- data.frame(mutation=rep(NA,report.obs),
                       elitism=rep(NA,report.obs),
                       lambda=rep(NA,report.obs),
                       monoids=rep(NA,report.obs),
                       degree=rep(NA,report.obs),
                       error=rep(NA,report.obs),
                       model=rep(NA,report.obs))
  
  # values to iterate, for each parameter
  mutation.vals <- seq(0.01,0.25,len=8)
  elitism.vals  <- seq(0.01,0.2,len=5)
  lambda.vals   <- sort(c( seq(0.3,1.5,len=7), c(.74,.78,.8,.82,.84)))
  max.monoid.vals <- 2:6
  max.degree.vals <- 2:4
  
  best.ga.model <- NULL  # the best GA model 
  best.ga.error <- 10e10 # & its error
  i <- 0
  
  for(run in 1:runs) {
    
    i <- i+1
    best.ga.error <- 10e10
    
    # iterate thru mutation
    for (mut in mutation.vals) {
      GAmodel <- rbga.bin(size = max.monoids + max.degree*n.vars*max.monoids, 
                          iters = iterations, 
                          popSize = population, 
                          mutationChance = mut, 
                          elitism = round(elitism.perc * population), 
                          evalFunc = evalFuncFactory(train.set, n.vars, max.monoids, lambda))      
     
      best.solution <- GAmodel$population[1,]
      best.formula <- paste0("y ~ ", make.formula(best.solution, n.vars, max.monoids))
      ga.model <- lm(best.formula, data=train.set)
      ga.pred  <- predict(ga.model, test.set[,-ncol(test.set)]) 
      ga.error <- rsme(ga.pred,test.set[,ncol(test.set)])

      if(ga.error < best.ga.error) { # save best ga.model
        best.ga.model <- best.solution
        best.ga.error <- ga.error
        mutation.rate <- mut
      }
      
      cat(paste0("mut: ",mut,"."))
    }
    
    report[i,]$mutation <- mutation.rate
    report[i,]$elitism <- elitism.perc
    report[i,]$lambda <- lambda
    report[i,]$monoids <- max.monoids
    report[i,]$degree <- max.degree
    report[i,]$error <- best.ga.error
    report[i,]$model <- make.formula(best.ga.model, n.vars, max.monoids)
    
    print(best.ga.error)
    
    ##### next iteration, elitism
    i <- i+1
    best.ga.error <- 10e10
    
    for (eli in elitism.vals) {
      GAmodel <- rbga.bin(size = max.monoids + max.degree*n.vars*max.monoids, 
                          iters = iterations, 
                          popSize = population, 
                          mutationChance = mutation.rate, 
                          elitism = round(eli * population), 
                          evalFunc = evalFuncFactory(train.set, n.vars, max.monoids, lambda))      
      
      best.solution <- GAmodel$population[1,]
      best.formula <- paste0("y ~ ", make.formula(best.solution, n.vars, max.monoids))
      ga.model <- lm(best.formula, data=train.set)
      ga.pred  <- predict(ga.model, test.set[,-ncol(test.set)]) 
      ga.error <- rsme(ga.pred,test.set[,ncol(test.set)])
      
      if(ga.error < best.ga.error) { # save best ga.model
        best.ga.model <- best.solution
        best.ga.error <- ga.error
        elitism.perc  <- eli
      }
      # show progress
      cat(paste0("eli: ",eli,"."))
    }
    
    report[i,]$mutation <- mutation.rate
    report[i,]$elitism <- elitism.perc
    report[i,]$lambda <- lambda
    report[i,]$monoids <- max.monoids
    report[i,]$degree <- max.degree
    report[i,]$error <- best.ga.error
    report[i,]$model <- make.formula(best.ga.model, n.vars, max.monoids)

    print(best.ga.error)  
  
    ##### next iteration, lambda
    i <- i+1
    best.ga.error <- 10e10
    
    for (lamb in lambda.vals) {
      GAmodel <- rbga.bin(size = max.monoids + max.degree*n.vars*max.monoids, 
                          iters = iterations, 
                          popSize = population, 
                          mutationChance = mutation.rate, 
                          elitism = round(elitism.perc * population), 
                          evalFunc = evalFuncFactory(train.set, n.vars, max.monoids, lamb))      
      
      best.solution <- GAmodel$population[1,]
      best.formula <- paste0("y ~ ", make.formula(best.solution, n.vars, max.monoids))
      ga.model <- lm(best.formula, data=train.set)
      ga.pred  <- predict(ga.model, test.set[,-ncol(test.set)]) 
      ga.error <- rsme(ga.pred,test.set[,ncol(test.set)])
      
      if(ga.error < best.ga.error) { # save best ga.model
        best.ga.model <- best.solution
        best.ga.error <- ga.error
        lambda  <- lamb
      }
      # show progress
      cat(paste0("lambda: ",lamb,"."))
    }
    
    report[i,]$mutation <- mutation.rate
    report[i,]$elitism <- elitism.perc
    report[i,]$lambda <- lambda
    report[i,]$monoids <- max.monoids
    report[i,]$degree <- max.degree
    report[i,]$error <- best.ga.error
    report[i,]$model <- make.formula(best.ga.model, n.vars, max.monoids)
    
    print(best.ga.error)
  
    ##### next iteration, max.monoids
    i <- i+1
    best.ga.error <- 10e10
    
    for (max.mnd in max.monoid.vals) {
      GAmodel <- rbga.bin(size = max.mnd + max.degree*n.vars*max.mnd, 
                          iters = iterations, 
                          popSize = population, 
                          mutationChance = mutation.rate, 
                          elitism = round(elitism.perc * population), 
                          evalFunc = evalFuncFactory(train.set, n.vars, max.mnd, lambda))      
      
      best.solution <- GAmodel$population[1,]
      best.formula <- paste0("y ~ ", make.formula(best.solution, n.vars, max.mnd))
      ga.model <- lm(best.formula, data=train.set)
      ga.pred  <- predict(ga.model, test.set[,-ncol(test.set)]) 
      ga.error <- rsme(ga.pred,test.set[,ncol(test.set)])
      
      if(ga.error < best.ga.error) { # save best ga.model
        best.ga.model <- best.solution
        best.ga.error <- ga.error
        max.monoids  <- max.mnd
      }
      # show progress
      cat(paste0("max.mnd: ",max.mnd,"."))
    }
    
    report[i,]$mutation <- mutation.rate
    report[i,]$elitism <- elitism.perc
    report[i,]$lambda <- lambda
    report[i,]$monoids <- max.monoids
    report[i,]$degree <- max.degree
    report[i,]$error <- best.ga.error
    report[i,]$model <- make.formula(best.ga.model, n.vars, max.monoids)
    
    print(best.ga.error)

    ##### next iteration, max.degree
    i <- i+1
    best.ga.error <- 10e10
    
    for (max.dgr in max.degree.vals) {
      GAmodel <- rbga.bin(size = max.monoids + max.dgr*n.vars*max.monoids, 
                          iters = iterations, 
                          popSize = population, 
                          mutationChance = mutation.rate, 
                          elitism = round(elitism.perc * population), 
                          evalFunc = evalFuncFactory(train.set, n.vars, max.monoids, lambda))      
      
      best.solution <- GAmodel$population[1,]
      best.formula <- paste0("y ~ ", make.formula(best.solution, n.vars, max.monoids))
      ga.model <- lm(best.formula, data=train.set)
      ga.pred  <- predict(ga.model, test.set[,-ncol(test.set)]) 
      ga.error <- rsme(ga.pred,test.set[,ncol(test.set)])
      
      if(ga.error < best.ga.error) { # save best ga.model
        best.ga.model <- best.solution
        best.ga.error <- ga.error
        max.degree  <- max.dgr
      }
      # show progress
      cat(paste0("max.dgr: ",max.dgr,"."))
    }
    
    report[i,]$mutation <- mutation.rate
    report[i,]$elitism <- elitism.perc
    report[i,]$lambda <- lambda
    report[i,]$monoids <- max.monoids
    report[i,]$degree <- max.degree
    report[i,]$error <- best.ga.error
    report[i,]$model <- make.formula(best.ga.model, n.vars, max.monoids)
    
    print(best.ga.error)
  } # for(runs)
    
  report
}

report <- tune.gapoly.iterative(my.data, population=10, iterations=8, runs=3)
report

write.table(report, "grid.iterative.search.Housing.txt")