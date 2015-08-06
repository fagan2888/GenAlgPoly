###############################
### Classify a dataset
###############################

source("gapolyfuncs.R")
set.seed(101)

###############################
### Read and clean data
###############################

read_clean <- function(dataset) {
  if (dataset=="Artificial") { 
    # artificial data to check if algorithm is able to find its formula
    size <- 50
    my.data <- data.frame(x1 = rpois(size,4),
                          x2 = rnorm(size,10),
                          x3 = rpois(size,20),
                          x4 = rnorm(size,4,1),                      
                          y  = x4^2*x2 + x1^2*x3 + 5)
    
    my.data$x1 <- as.numeric(my.data$x1)
  } else {
    filename <- paste0(dataset,"/",dataset,".data")  # read file with dataset
    filepath <- paste0("datasets/",filename)
    my.data  <- read.csv(filepath)
  }
  
  # preprocess data
  names(my.data) <- c(paste0("x",1:(ncol(my.data)-1)),"y") # columns are named as x_i & y
  
  if (dataset=="Abalone") {
    my.data$x1 <- as.numeric(my.data$x1) # x1 was a factor, make it a numeric value
  }
  
  if (dataset=="Auto-Mpg") {
    my.data$x3 <- as.numeric(my.data$x3) # x3 was a factor, make it a numeric value
  }
  
  if (dataset=="Housing") { }
  if (dataset=="Kinematics") { }
  
  as.data.frame(scale(my.data))  # scale data for better inference
}

###############################
### Select dataset (pick one)
###############################

dataset <- "Artificial"  
dataset <- "Abalone"
dataset <- "Auto-Mpg"
dataset <- "Housing"
dataset <- "Kinematics"

my.data <- read_clean(dataset)
sapply(my.data, class)  # check if all are numeric fields
head(my.data,8)

###############################
### Execute ML methods
###############################

pop  <- 300  # GA population
reg  <- 0.8  # amount of regularization
runs <- 25   # number of simulations
iter <- 100  # number of interations for each simulation

# Apply regularization
report_reg <- make.report(my.data, population=pop, iterations=iter, lambda=reg, runs=runs)

df <- data.frame(ga.poly = report_reg$ga.error, 
                 lin.reg = report_reg$lm.error,
                 svm     = report_reg$svm.error,
                 rpart   = report_reg$rpart.error,
                 rf      = report_reg$rf.error,
                 ci.tree = report_reg$citree.error)

write.table(df, paste0(dataset, pop, "_lambda", reg,".txt"))

# Apply standard GA polynomial regression

report_noreg <- make.report(my.data, population=pop, iterations=iter, lambda=1, runs=runs)

df <- data.frame(ga.poly = report_noreg$ga.error, 
                 lin.reg = report_noreg$lm.error,
                 svm     = report_noreg$svm.error,
                 rpart   = report_noreg$rpart.error,
                 rf      = report_noreg$rf.error,
                 ci.tree = report_noreg$citree.error)

write.table(df, paste0(dataset, pop, "_noReg.txt"))

# df <- read.table(paste0("results/", dataset, pop, "_results.txt"), header = TRUE)
# apply(df, 2, summary)

# show results

size <- length(report$ga.error)
list.data <- list(error=c(report_reg$ga.error, 
                          report_noreg$ga.error,
                          report_reg$lm.error,
                          report_reg$svm.error,
                          report_reg$rpart.error,
                          report_reg$rf.error,
                          report_reg$citree.error), 
                  method=c(rep("EPRR",size),
                           rep("EPR",size),
                           rep("Lin.Reg.",size),
                           rep("SVM",size), 
                           rep("RPART",size),
                           rep("Rnd.For.",size),
                           rep("CI.Tree",size)))

library(ggplot2)

qplot(method,error,data=as.data.frame(list.data),geom=c("jitter","boxplot")) + 
  ylab("root-mean-square error") + 
  xlab("regression method") +
  ggtitle(paste(dataset,"dataset"))

###############################
### Lambda Test
### cf. article's figure "error distribution by regularization exponent"
###############################

library(ggplot2)
library(plyr)

lambda.values <- seq(0.7,1.3,by=0.1)
n.runs <- 50

#datasets <- c("Abalone", "Auto-Mpg", "Housing", "Kinematics")
datasets <- c("Housing", "Kinematics")

for (dataset in datasets) {
  # print(paste("### Starting", dataset, "dataset..."))
  
  my.data <- read_clean(dataset)
  
  errors <- test.lambda(my.data, lambda.values, runs=n.runs, population=50, iterations=60)
  
  df <- data.frame(error  = as.vector(matrix(errors,nrow=1)),
                   lambda = as.factor(rep(round(lambda.values,3), each=n.runs)))
  
  write.table(df, paste0(dataset,"_lambda.errors.txt"))
}


# to read the previous table from file:
# df <- read.table(paste0("results/", dataset,"_lambda.errors.txt"))
# df$lambda <- as.factor(df$lambda)

qplot(lambda,error,data=df,geom=c("jitter","boxplot")) + 
  ylab("root-mean-square error") + 
  xlab("lambda value") +
  #coord_cartesian(ylim = c(0, 1))  +
  ggtitle(paste(dataset,"dataset"))

# check the min error for each lambda value (not very useful)
# best.errors <- 
#   colwise(function(x) min(as.numeric(x)))(as.data.frame(errors)) # get min for each column
# plot(lambda.values, best.errors, type="l")

###############################
### Check fitness progress
### cf. article's figure "Learning curve"
###############################

library(ggplot2)

fitness.progress <- c() # needed to reset fitness progress
lambda0.8 <- 
  follow.fitness(my.data,population=200,iterations=20,mutation.rate=0.15,lambda=0.8)

fitness.progress <- c()
lambda0.975 <- 
  follow.fitness(my.data,population=200,iterations=20,mutation.rate=0.15,lambda=0.975)

fitness.progress <- c()
lambda1 <- 
  follow.fitness(my.data,population=200,iterations=20,mutation.rate=0.15,lambda=1)

lambda_Values = list(lambda0.8=lambda0.8, lambda0.975=lambda0.975, lambda1=lambda1)
save(lambda_Values, file=paste0(dataset,"_lambdas.Rda")) # recover with load("<name>.Rda") 

df <- data.frame(iteration = 1:length(lambda0.975),
                 lambda0.8 = as.numeric(lambda0.8),
                 lambda0.975 = as.numeric(lambda0.975),
                 lambda1 = as.numeric(lambda1))

write.table(round(df,5), paste0(dataset,"_lambdas.txt"))

ggplot(df, aes(iteration)) + 
  geom_line(aes(y = lambda0.8,   colour = "lambda = 0.8")) + 
  geom_line(aes(y = lambda0.975, colour = "lambda = 0.975")) + 
  geom_line(aes(y = lambda1,     colour = "lambda = 1.0")) +
  #ylim(0.36, 0.5) +
  ggtitle(paste0(dataset," dataset")) +
  ylab("rmse") + 
  theme(legend.title=element_blank()) # remove legend title

###############################
### Tuning SVM & RPART
###############################

library(e1071)

rsme <- function(v1, v2=c(0)) {
  sqrt(mean((v1-v2)^2))
}

n.runs <- 25
train.p.size <- 0.7 # percentage of training set

svm.error  <- rep(NA,n.runs)
rpart.error <- rep(NA,n.runs)

for (i in 1:n.runs) {
  inTrain   <- sample(1:nrow(my.data), train.p.size * nrow(my.data))
  train.set <- my.data[inTrain,]
  test.set  <- my.data[-inTrain,]
  
  svm.tune     <- tune.svm(y ~ ., data=train.set)
  svm.pred     <- predict(svm.tune$best.model, test.set[,-ncol(test.set)]) 
  svm.error[i] <- rsme(svm.pred,test.set[,ncol(test.set)])
  
  rpart.tune     <- tune.rpart(y ~ ., data=train.set)
  rpart.pred     <- predict(rpart.tune$best.model, test.set[,-ncol(test.set)]) 
  rpart.error[i] <- rsme(rpart.pred,test.set[,ncol(test.set)])
}

summary(svm.error)
summary(rpart.error)


###############################
# check what attributes Random Forest select as the most importants
###############################

train.p.size <- 0.7 # percentage of training set
n.vars       <- ncol(my.data)-1

inTrain   <- sample(1:nrow(my.data), train.p.size * nrow(my.data))
train.set <- my.data[inTrain,]
test.set  <- my.data[-inTrain,]

rf.model <- randomForest(y ~ ., data=train.set, importance=TRUE, do.trace=100, ntree=100)
rf.model$importance

