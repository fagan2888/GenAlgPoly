source("gapolyfuncs.R")

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

pop  <- 300  # GA population
reg  <- 0.7  # amount of regularization
runs <- 75   # number of simulations
iter <- 100  # number of interations for each simulation

digits <- 5 # round errors after number of digits

################## started Monday 13:43 

#datasets <- c("Abalone", "Auto-Mpg", "Housing", "Kinematics")
datasets <- c("Housing", "Kinematics")

for (dataset in datasets) {
  # print(paste("### Starting", dataset, "dataset..."))
        
  my.data <- read_clean(dataset)
  
  # Apply regularization
  report_reg <- make.report(my.data, population=pop, iterations=iter, lambda=reg, runs=runs)
  
  # Apply standard GA polynomial regression
  # report_noreg <- make.report(my.data, population=pop, iterations=iter, lambda=1, runs=runs)

  # build and save table of error results  
  df <- data.frame(ga.eprr = report_reg$ga.error,
                   #ga.epr  = report_noreg$ga.error, 
                   lin.reg = report_reg$lm.error,
                   svm     = report_reg$svm.error,
                   rpart   = report_reg$rpart.error,
                   rf      = report_reg$rf.error,
                   ci.tree = report_reg$citree.error)
  
  df <- round(df, digits)
  
  write.table(df, paste0(dataset, pop, "_eprr_", reg,".txt"))
  
}