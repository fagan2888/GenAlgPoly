GA Poly
========================================================

GA package use based on [http://www.r-bloggers.com/genetic-algorithms-a-simple-r-example/](http://www.r-bloggers.com/genetic-algorithms-a-simple-r-example/)

<hr>

Herein we code a polynomial based on variables $x_1, x_2 \ldots x_n$ into a binary vector (ie, made of 0s and 1s).

To achieve this we need to know three values in advance:
+ number of different variables
+ maximum number of different components (being a polynomial a sum of components)
+ maximum degree for any given variable


```r
n.vars <- 3  # x1, x2,...
max.compts <- 4  # poly = compt1 + compt2 + ...
max.degree <- 3
```


The chromosome coding will as follow:
+ the chromosome is split into `max.compts` sets of bits of equal size (a component)
+ each component is split into `n.vars` sets of `max.degree` size each (a variable)
+ for each variable, the number of 1s gives the variable's degree (this means that a given polynomial may have more than one possible representation)

Eg: consider polynomial $x_1^2 \times x_3 + x_3^3 + x_1 \times x_2$

One possible coding -- for the values in the previous code snippet -- would be:

$$011,000,001;000,000,111;001,001,000;000,000,000$$

(the semicolons separate components, the commas seperate variables)

The next functions transform the binary vector into a polynomial formula that `lm()` understands:


```r
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

make.formula <- function(bits, n.vars, max.compts) {
    components <- split(bits, ceiling(seq_along(bits)/(length(bits)/max.compts)))
    str <- ""
    for (i in 1:length(components)) {
        comp <- make.component(components[[i]], n.vars)
        if (comp != "") 
            if (str == "") 
                str = paste0("I(", comp, ")") else str = paste0(str, " + I(", comp, ")")
    }
    str
}
```


Let's check the previous eg, $x_1^2 \times x_3 + x_3^3 + x_1 \times x_2$:


```r
code.eg <- c(0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 
    0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
paste0(code.eg, collapse = "")  # just to show as string
```

```
## [1] "011000001000000111001001000000000000"
```

```r
make.formula(code.eg, n.vars, max.compts)
```

```
## [1] "I(x1^2*x3) + I(x3^3) + I(x1*x2)"
```


Let's create a dataframe for testing. Notice that the data frame columns name must be called $y$ for the output value, and $x_i$ for the input values:


```r
set.seed(102)
x1 <- seq(0, 8 * pi, length.out = 201)/pi
x2 <- sin(x1)^2
x3 <- abs(cos(x1))
y <- x1 * x2 - x1 * x3 + x3^3 + rnorm(201, 0, 0.15)  # a latent variable with some noise
df <- data.frame(y = y, x1 = x1, x2 = x2, x3 = x3)  # make data frame
```


This binary coding is useful to use the binary GA `rbga.bin()` from package `genalg` (see below):


```r
library(genalg)

# This is a factory for the evaluation function that the GA function needs
# to execute.

# The factory is a closure that includes the dataframe, the number of
# variable and the max number of componentes, all required information to
# perform the polynomial regression and then compute the residuals' rsme
# which will be the fitness of the respective chromosome
evalFuncFactory <- function(df, n.vars, max.compts) {
    
    function(chromosome) {
        formula <- paste0("y ~ ", make.formula(chromosome, n.vars, max.compts))
        model <- lm(formula, data = df)
        # rbga.bin() minimizes, so the rmse will be smaller as the interations
        # advance
        return(sqrt(mean(residuals(model)^2)))
    }
}

################################# Unleash the chromosomes of war!
GAmodel <- rbga.bin(size = max.degree * n.vars * max.compts, popSize = 150, 
    iters = 100, mutationChance = 0.01, elitism = TRUE, evalFunc = evalFuncFactory(df, 
        n.vars, max.compts))

cat(summary.rbga(GAmodel))
```

```
## GA Settings
##   Type                  = binary chromosome
##   Population size       = 150
##   Number of Generations = 100
##   Elitism               = TRUE
##   Mutation Chance       = 0.01
## 
## Search Domain
##   Var 1 = [,]
##   Var 0 = [,]
## 
## GA Results
##   Best Solution : 0 0 0 0 0 1 0 0 0 1 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 1 0
```

```r
best.solution <- GAmodel$population[1, ]
best.formula <- paste0("y ~ ", make.formula(best.solution, n.vars, max.compts))
model <- lm(best.formula, data = df)
coef(model)
```

```
## (Intercept)       I(x2)  I(x1 * x2)  I(x1 * x3)  I(x2 * x3) 
##      0.9955     -0.8692      0.9860     -0.9962     -0.5044
```

```r
plot(df$x1, df$y, type = "l")
title(best.formula)
y1 <- predict(model, df[, -1])
lines(x1, y1, col = "red")  # approx
```

![plot of chunk unnamed-chunk-5](figure/unnamed-chunk-5.png) 

