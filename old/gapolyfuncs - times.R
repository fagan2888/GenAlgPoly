#########################################################################
# initial values, can be overriden
n.vars     <- 3  # x1, x2,... 
max.compts <- 4  # poly = compt1 + compt2 + ...
max.degree <- 3  # meaning exponents up to 2^max.degree - 1

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
# + the chromosome is split into `max.compts` sets of bits of equal size 
#   (a component)
# + each component is split into `n.vars` sets of `max.degree` size each 
#   (a variable)
# + for each variable, the number of 1s gives the variable's degree 
#   (this means that a given polynomial may have more than one possible 
#   representation)
# 
# Eg: consider polynomial $x_1^3 \times x_3 + x_3^7 + x_1 \times x_2$
# 
# One possible coding -- for the values in the previous code snippet -- would be:
# 
# $$011,000,001;000,000,111;001,001,000;000,000,000$$
# 
# (the semicolons separate components, the commas seperate variables)
# 
# The next functions transform the binary vector into a polynomial formula 
# that `lm()` understands:

make.component <- function(bits, n.vars) {
  vars <- paste0("x",1:n.vars)
  elements <- split(bits, ceiling(seq_along(bits)/(length(bits)/n.vars)))
  str <- ""
  for(i in 1:length(elements)) {
    power <- bin2dec(elements[[i]])
    if (power == 1)
      if (str=="")
        str = vars[i]
    else  
      str = paste0(str, "*", vars[i])
    if (power > 1)
      if (str=="")
        str = paste0(vars[i], "^", power)
    else  
      str = paste0(str, "*", vars[i], "^", power)
  }
  str
}

make.formula <- function(bits, n.vars, max.compts) {
  components <- split(bits, ceiling(seq_along(bits)/(length(bits)/max.compts)))
  str <- ""
  for(i in 1:length(components)) {
    comp <- make.component(components[[i]], n.vars)
    if (comp!="")
      if (str=="")
        str = paste0("I(",comp,")")
    else
      str = paste0(str," + I(", comp ,")")
  }
  str
}

#########################################################################
# This is a factory for the evaluation function that the GA function
# needs to execute. 

# The factory is a closure that includes the dataframe, the number of
# variable and the max number of componentes, all required information
# to perform the polynomial regression and then compute the residuals' rsme 
# which will be the fitness of the respective chromosome
evalFuncFactory <- function(df, n.vars, max.compts) {
  
  function(chromosome) {
    formula <- paste0("y ~ ", make.formula(chromosome, n.vars, max.compts))
    model <- lm(formula, data=df)
    # rbga.bin() minimizes, so the rmse will be smaller as the interations advance
    return( sqrt(mean(residuals(model)^2)) ) 
  }
}

#########################################################################
# Returns a dataset with the rsme for each one of the tested ML methods
# including GA Poly

# pre: both sets must have columns named x1...xn and the last one is y (the output)
make.report <- function(my.data, population=250, iterations=250, runs=10) {
  
  library(genalg)
  library(e1071)
  library(rpart)   
  library(randomForest)
  library(party)
  
  train.p.size <- 0.7 # percentage of training set
  n.vars       <- ncol(my.data)-1
  
  ga.poly.rsme <- rep(0,runs)
  lm.rsme      <- rep(0,runs)
  svm.rsme     <- rep(0,runs)
  rpart.rsme   <- rep(0,runs)
  rf.rsme      <- rep(0,runs)
  citree.rsme  <- rep(0,runs)
  
  for (i in 1:runs) {
    
    # make train & test set
    inTrain   <- sample(1:nrow(my.data), train.p.size * nrow(my.data))
    train.set <- my.data[inTrain,]
    test.set  <- my.data[-inTrain,]
    
    GAmodel <- rbga.bin(size = max.degree*n.vars*max.compts, 
                        popSize = population, 
                        iters = iterations, 
                        mutationChance = 0.02, 
                        elitism = TRUE, 
                        evalFunc = evalFuncFactory(train.set, n.vars, max.compts))
      
    best.solution <- GAmodel$population[1,]
    best.formula <- paste0("y ~ ", make.formula(best.solution, n.vars, max.compts))
    ga.model <- lm(best.formula, data=my.data)
    ga.pred  <- predict(ga.model, test.set[,-ncol(test.set)]) 
    ga.error <- rsme(ga.pred,test.set[,ncol(test.set)])
    
    lm.time <- system.time( lm.model <- lm(y ~ ., data = train.set) )
    lm.pred  <- predict(lm.model, test.set[,-ncol(test.set)]) 
    lm.error <- rsme(lm.pred,test.set[,ncol(test.set)])
    
    
    svm.time <- system.time( svm.model <- svm(y ~ ., data=train.set, kernel='linear') )
    svm.pred  <- predict(svm.model, test.set[,-ncol(test.set)]) 
    svm.error <- rsme(svm.pred,test.set[,ncol(test.set)])

    r.time <- system.time( rpart.model <- rpart(y ~ ., data=train.set) )
    rpart.pred  <- predict(rpart.model, test.set[,-ncol(test.set)]) 
    rpart.error <- rsme(rpart.pred,test.set[,ncol(test.set)])

    rf.time <- system.time( rf.model <- randomForest(y ~ ., data=train.set, importance=TRUE, do.trace=100, ntree=100) )
    rf.pred  <- predict(rf.model, test.set[,-ncol(test.set)]) 
    rf.error <- rsme(rf.pred,test.set[,ncol(test.set)])
    

    c.time <- system.time( citree.model <- ctree(y ~ ., data = train.set) )
    citree.pred  <- predict(citree.model, test.set[,-ncol(test.set)]) 
    citree.error <- rsme(citree.pred,test.set[,ncol(test.set)])
    
    # creating report
    methods <- c("GA Poly", "Linear Regression", "SVM", "Regression Tree", "Random Forest", "Conditional Inference Trees")
    results <- c(ga.error, lm.error, svm.error, rpart.error, rf.error, citree.error)
    results <- signif(results, digits=4)
    times <- c(ga.time[[3]], lm.time[[3]], svm.time[[3]], r.time[[3]], rf.time[[3]], c.time[[3]])
    times <- signif(times, digits=4)
    
  }
  
  
  
    
    list(report = as.data.frame( cbind(methods, results, times)),
         ga.model = ga.model)
}