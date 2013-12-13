#scikit-learn

## Supervised Learning

### Generalized Linear Models

> Methods intended for regression in which the target value is expected to be a linear combination of the input variables

### Support Vector Machines

> Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.
>
>The advantages of support vector machines are:
>
> * Effective in high dimensional spaces.
> * Still effective in cases where number of dimensions is greater than the number of samples.
> * Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.
> * Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.
> 
> The disadvantages of support vector machines include:
> 
> * If the number of features is much greater than the number of samples, the method is likely to give poor performances.
> * SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation.

### Stochastic Gradient Descent

> Stochastic Gradient Descent (SGD) is a simple yet very efficient approach to discriminative learning of linear classifiers under convex loss functions such as (linear) Support Vector Machines and Logistic Regression. Even though SGD has been around in the machine learning community for a long time, it has received a considerable amount of attention just recently in the context of large-scale learning.  
> 
> SGD has been successfully applied to large-scale and sparse machine learning problems often encountered in text classification and natural language processing. Given that the data is sparse, the classifiers in this module easily scale to problems with more than 10^5 training examples and more than 10^5 features.
> 
> The advantages of Stochastic Gradient Descent are:
> 
> * Efficiency.
> * Ease of implementation (lots of opportunities for code tuning).
> 
> The disadvantages of Stochastic Gradient Descent include:
> 
> * SGD requires a number of hyperparameters such as the regularization parameter and the number of iterations.
> * SGD is sensitive to feature scaling.
> 

### Nearest Neighbors

> Nearest Neighbors provides functionality for unsupervised and supervised neighbors-based learning methods. Unsupervised nearest neighbors is the foundation of many other learning methods, notably manifold learning and spectral clustering. Supervised neighbors-based learning comes in two flavors: classification for data with discrete labels, and regression for data with continuous labels.
> 
>The principle behind nearest neighbor methods is to find a predefined number of training samples closest in distance to the new point, and predict the label from these. The number of samples can be a user-defined constant (k-nearest neighbor learning), or vary based on the local density of points (radius-based neighbor learning). The distance can, in general, be any metric measure: standard Euclidean distance is the most common choice. Neighbors-based methods are known as non-generalizing machine learning methods, since they simply “remember” all of its training data (possibly transformed into a fast indexing structure such as a Ball Tree or KD Tree.).
> 
> Despite its simplicity, nearest neighbors has been successful in a large number of classification and regression problems, including handwritten digits or satellite image scenes. Being a non-parametric method, it is often successful in classification situations where the decision boundary is very irregular.
> 
>The classes in sklearn.neighbors can handle either Numpy arrays or scipy.sparse matrices as input. For dense matrices, a large number of possible distance metrics are supported. For sparse matrices, arbitrary Minkowski metrics are supported for searches.
>
>There are many learning routines which rely on nearest neighbors at their core. One example is kernel density estimation, discussed in the density estimation section.


### Gaussian Processes

> Gaussian Processes for Machine Learning (GPML) is a generic supervised learning method primarily designed to solve regression problems. It has also been extended to probabilistic classification, but in the present implementation, this is only a post-processing of the regression exercise.
> 
> The advantages of Gaussian Processes for Machine Learning are:
>
> * The prediction interpolates the observations (at least for regular correlation models).
> * The prediction is probabilistic (Gaussian) so that one can compute empirical confidence intervals and exceedance probabilities that might be used to refit (online fitting, adaptive fitting) the prediction in some region of interest.
> * Versatile: different linear regression models and correlation models can be specified. Common models are provided, but it is also possible to specify custom models provided they are stationary.
> 
> The disadvantages of Gaussian Processes for Machine Learning include:
>
> * It is not sparse. It uses the whole samples/features information to perform the prediction.
> * It loses efficiency in high dimensional spaces – namely when the number of features exceeds a few dozens. It might indeed give poor performance and it loses computational efficiency.
> * Classification is only a post-processing, meaning that one first need to solve a regression problem by providing the complete scalar float precision output y of the experiment one attempt to model.
> 
> Thanks to the Gaussian property of the prediction, it has been given varied applications: e.g. for global optimization, probabilistic classification.



### Cross decomposition

> The cross decomposition module contains two main families of algorithms: the partial least squares (PLS) and the canonical correlation analysis (CCA).
> 
> These families of algorithms are useful to find linear relations between two multivariate datasets: the X and Y arguments of the fit method are 2D arrays.
> 
> Cross decomposition algorithms find the fundamental relations between two matrices (X and Y). They are latent variable approaches to modeling the covariance structures in these two spaces. They will try to find the multidimensional direction in the X space that explains the maximum multidimensional variance direction in the Y space. PLS-regression is particularly suited when the matrix of predictors has more variables than observations, and when there is multicollinearity among X values. By contrast, standard regression will fail in these cases.

### Naive Bayes

> Naive Bayes methods are a set of supervised learning algorithms based on applying Bayes’ theorem with the “naive” assumption of independence between every pair of features. 
> 
> In spite of their apparently over-simplified assumptions, naive Bayes classifiers have worked quite well in many real-world situations, famously document classification and spam filtering. They requires a small amount of training data to estimate the necessary parameters. (For theoretical reasons why naive Bayes works well, and on which types of data it does, see the references below.)
> 
> Naive Bayes learners and classifiers can be extremely fast compared to more sophisticated methods. The decoupling of the class conditional feature distributions means that each distribution can be independently estimated as a one dimensional distribution. This in turn helps to alleviate problems stemming from the curse of dimensionality.
> 
> On the flip side, although naive Bayes is known as a decent classifier, it is known to be a bad estimator, so the probability outputs from predict_proba are not to be taken too seriously.

### Decision Trees

> Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.
> 
> For instance, in the example below, decision trees learn from data to approximate a sine curve with a set of if-then-else decision rules. The deeper the tree, the more complex the decision rules and the fitter the model.
>
> Some advantages of decision trees are:
> 
> * Simple to understand and to interpret. Trees can be visualised.
> * Requires little data preparation. Other techniques often require data normalisation, dummy variables need to be created and blank values to be removed. Note however that this module does not support missing values.
> * The cost of using the tree (i.e., predicting data) is logarithmic in the number of data points used to train the tree.
> * Able to handle both numerical and categorical data. Other techniques are usually specialised in analysing datasets that have only one type of variable. See algorithms for more information.
> * Able to handle multi-output problems.
> * Uses a white box model. If a given situation is observable in a model, the explanation for the condition is easily explained by boolean logic. By contrast, in a black box model (e.g., in an artificial neural network), results may be more difficult to interpret. 
> * Possible to validate a model using statistical tests. That makes it possible to account for the reliability of the model.
> * Performs well even if its assumptions are somewhat violated by the true model from which the data were generated.
> 
> The disadvantages of decision trees include:
> 
> * Decision-tree learners can create over-complex trees that do not generalise the data well. This is called overfitting. Mechanisms such as pruning (not currently supported), setting the minimum number of samples required at a leaf node or setting the maximum depth of the tree are necessary to avoid this problem.
> * Decision trees can be unstable because small variations in the data might result in a completely different tree being generated. This problem is mitigated by using decision trees within an ensemble.
> * The problem of learning an optimal decision tree is known to be NP-complete under several aspects of optimality and even for simple concepts. Consequently, practical decision-tree learning algorithms are based on heuristic algorithms such as the greedy algorithm where locally optimal decisions are made at each node. Such algorithms cannot guarantee to return the globally optimal decision tree. This can be mitigated by training multiple trees in an ensemble learner, where the features and samples are randomly sampled with replacement.
> * There are concepts that are hard to learn because decision trees do not express them easily, such as XOR, parity or multiplexer problems.
> * Decision tree learners create biased trees if some classes dominate. It is therefore recommended to balance the dataset prior to fitting with the decision tree.

### Ensemble methods

> The goal of ensemble methods is to combine the predictions of several models built with a given learning algorithm in order to improve generalizability / robustness over a single model.
> 
> Two families of ensemble methods are usually distinguished:
>
> * In **averaging methods**, the driving principle is to build several models independently and then to average their predictions. On average, the combined model is usually better than any of the single model because its variance is reduced.  
> **Examples:** Bagging methods, Forests of randomized trees...
> * By contrast, in **boosting methods**, models are built sequentially and one tries to reduce the bias of the combined model. The motivation is to combine several weak models to produce a powerful ensemble.  
>**Examples:** AdaBoost, Gradient Tree Boosting, ...

### Multiclass and multilabel algorithms

>Those meta-estimators are meant to turn a binary classifier or a regressor into a multi-class/label classifier.
>
> * Multiclass classification means a classification task with more than two classes; e.g., classify a set of images of fruits which may be oranges, apples, or pears. Multiclass classification makes the assumption that each sample is assigned to one and only one label: a fruit can be either an apple or a pear but not both at the same time.
> * Multilabel classification assigns to each sample a set of target labels. This can be thought as predicting properties of a data-point that are not mutually exclusive, such as topics that are relevant for a document. A text might be about any of religion, politics, finance or education at the same time or none of these.
> * Multioutput-multiclass classification and multi-task classification means that an estimators have to handle jointly several classification tasks. This is a generalization of the multi-label classification task, where the set of classification problem is restricted to binary classification, and of the multi-class classification task. The output format is a 2d numpy array.  
>The set of labels can be different for each output variable. For instance a sample could be assigned “pear” for an output variable that takes possible values in a finite set of species such as “pear”, “apple”, “orange” and “green” for a second output variable that takes possible values in a finite set of colors such as “green”, “red”, “orange”, “yellow”...  
>This means that any classifiers handling multi-output multiclass or multi-task classification task supports the multi-label classification task as a special case. Multi-task classification is similar to the multi-output classification task with different model formulations. For more information, see the relevant estimator documentation.
>
> Estimators in this module are meta-estimators. For example, it is possible to use these estimators to turn a binary classifier or a regressor into a multiclass classifier. It is also possible to use these estimators with multiclass estimators in the hope that their generalization error or runtime performance improves.
> 
> You don’t need to use these estimators unless you want to experiment with different multiclass strategies: all classifiers in scikit-learn support multiclass classification out-of-the-box. Below is a summary of the classifiers supported by scikit-learn grouped by strategy:
> 
> * Inherently multiclass: Naive Bayes, sklearn.lda.LDA, Decision Trees, Random Forests, Nearest Neighbors.
> * One-Vs-One: sklearn.svm.SVC.
> * One-Vs-All: all linear models except sklearn.svm.SVC.
> 
> Some estimators also support multioutput-multiclass classification tasks Decision Trees, Random Forests, Nearest Neighbors.


### Feature selection

> Used for feature selection/dimensionality reduction on sample sets, either to improve estimators’ accuracy scores or to boost their performance on very high-dimensional datasets.

### Semi-Supervised

>Semi-supervised learning is a situation in which in your training data some of the samples are not labeled. The semi-supervised estimators are able to make use of this addition unlabeled data to capture better the shape of the underlying data distribution and generalize better to new samples. These algorithms can perform well when we have a very small amount of labeled points and a large amount of unlabeled points.

### Linear and quadratic discriminant analysis

> Linear discriminant analysis and quadratic discriminant analysis are two classic classifiers, with, as their names suggest, a linear and a quadratic decision surface, respectively.
> 
> These classifiers are attractive because they have closed-form solutions that can be easily computed, are inherently multiclass, and have proven to work well in practice. Also there are no parameters to tune for these algorithms.

### Isotonic regression

> The class IsotonicRegression fits a non-decreasing function to data.
> 
> It yields the vector which is composed of non-decreasing elements the closest in terms of mean squared error. In practice this list of elements forms a function that is piecewise linear.
