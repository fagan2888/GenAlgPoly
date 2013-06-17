GA Poly
========================================================



```r
library(genalg)

x1 <- seq(0, 4 * pi, length.out = 201)
x2 <- seq(0, 4 * pi, length.out = 201)

# a latent variable, with a trick expression and some noise
y <- 2 * sin(x1) + cos(2 * x2) + 0.1 * x1 * x2 + 0.1 * rnorm(201)

# Caution: there are **ONLY** 201 observations (and not 201*201)
observations <- data.frame(y = y, x1 = x1, x2 = x2)

##############

make.component <- function(bits, n.vars) {
    vars <- paste0("x", 1:n.vars)
    elements <- split(bits, ceiling(seq_along(bits)/(length(bits)/n.vars)))
    str <- ""
    for (i in 1:length(elements)) {
        power <- length(elements[[i]][elements[[i]] == 1])
        if (power == 1) 
            if (str == "") 
                str = vars[i] else str = paste0(str, "*", vars[i])
        if (power > 1) 
            if (str == "") 
                str = paste0(vars[i], "^", power) else str = paste0(str, "*", vars[i], "^", power)
    }
    str
}

make.formula <- function(bits, n.vars, n.compts) {
    components <- split(bits, ceiling(seq_along(bits)/(length(bits)/n.compts)))
    str <- ""
    for (i in 1:length(components)) {
        comp <- make.component(components[[i]], n.vars)
        if (comp != "") 
            if (str == "") 
                str = paste0("I(", comp, ")") else str = paste0(str, " + I(", comp, ")")
    }
    str
}

############################

max.degree <- 3
n.vars <- 3  # x1, x2,...
n.compts <- 4  # poly = compt1 + compt2 + ...

set.seed(101)
bits <- sample(c(0, 1), max.degree * n.vars * n.compts, replace = TRUE)
paste0(bits, collapse = "")  # just to show as string
```

```
## [1] "001100101111110110001101110001000011"
```

```r

##############################

x1 <- seq(0, 8 * pi, length.out = 201)/pi
x2 <- sin(x1)^2
x3 <- abs(cos(x2))
y <- x1 * x2 - x1 * x3 + x3^3 + rnorm(201, 0, 0.15)  # a latent variable with some noise
df <- data.frame(y = y, x1 = x1, x2 = x2, x3 = x3)  # make data frame

evalFuncFactory <- function(df, n.vars, n.compts) {
    
    function(bits) {
        formula <- paste0("y ~ ", make.formula(bits, n.vars, n.compts))
        model <- lm(formula, data = df)
        return(sqrt(mean(residuals(model)^2)))
    }
}

#############################

GAmodel <- rbga.bin(size = max.degree * n.vars * n.compts, popSize = 100, iters = 100, 
    mutationChance = 0.01, elitism = TRUE, evalFunc = evalFuncFactory(df, n.vars, 
        n.compts))

cat(summary.rbga(GAmodel))
```

```
## GA Settings
##   Type                  = binary chromosome
##   Population size       = 100
##   Number of Generations = 100
##   Elitism               = TRUE
##   Mutation Chance       = 0.01
## 
## Search Domain
##   Var 1 = [,]
##   Var 0 = [,]
## 
## GA Results
##   Best Solution : 0 0 1 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 1 1 0 0 1
```

```r

best.solution <- GAmodel$population[1, ]
best.formula <- paste0("y ~ ", make.formula(best.solution, n.vars, n.compts))
model <- lm(best.formula, data = df)
coef(model)
```

```
##  (Intercept)   I(x1 * x2)        I(x1)   I(x1 * x3) I(x2^2 * x3) 
##      0.99960      0.98987      0.03768     -1.03396     -1.60290
```

```r
plot(df$x1, df$y, type = "l")
title(best.formula)
y1 <- predict(model, df[, -1])
lines(x1, y1, col = "red")  # approx
```

![plot of chunk unnamed-chunk-1](figure/unnamed-chunk-1.png) 


