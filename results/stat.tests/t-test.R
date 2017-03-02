data <- read.csv("reg_vs_nonreg.txt",header=TRUE,sep=" ")
names(data) <- c("poly","poly.reg")

# cf. http://ww2.coastal.edu/kingw/statistics/R-tutorials/independent-t.html
t.test(data[,1],data[,2])
boxplot(data[,1],data[,2],ylim=c(0.3,.9))

library(ggplot2)
size <- length(data$poly)
list.data <- list(error=c(data$poly, 
                          data$poly.reg), 
                  method=c(rep("poly",size),
                           rep("polyreg",size)))
df <- as.data.frame(list.data)

science_theme = theme(panel.grid.major = element_line(size = 0.25, color = "grey"), 
                      axis.line = element_line(size = 0.7, color = "black"), 
                      legend.position = c(0.85, 0.7), text = element_text(size = 14))

qplot(method,error,data=df,geom=c("boxplot","jitter")) + 
  ylab("root-mean-square error") + 
  xlab("regression method") +
  coord_cartesian(ylim = c(0.35, 1))  +
  ggtitle("Housing dataset") + theme_classic() + science_theme 


######################

model <- lm("poly ~ poly.reg",data=data)
sum(model$coefficients^2)  
  