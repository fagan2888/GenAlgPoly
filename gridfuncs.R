######################################
# Tuning GAPoly parameters

source("gapolyfuncs.R")

# Parameters:
# + mutation.rate \in [0,0.5)
# + elitism       percentage
# + lambda        >= 0
# + max.monoids   [1,9]
# + max.degree    [2,9]

#mutation.rate <- elitism.perc <- lambda <- max.monoids <- max.degree <- 0

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
    
#     print(mutation.rate)
#     print(elitism.perc)
#     print(lambda)
#     print(max.monoids)
#     print(max.degree)
    
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

report <- tune.gapoly.params(my.data, population=80, iterations=60, runs=25)
report

# cf: 
# http://stats.stackexchange.com/questions/3328/given-a-10d-mcmc-chain-how-can-i-determine-its-posterior-modes-in-r