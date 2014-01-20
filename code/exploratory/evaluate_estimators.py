#!/opt/local/bin/python

import sklearn as sk
import sklearn.cross_validation as cv
import numpy as np
import eprr
import os
import glob

#########################################################################################
#
#   GET DATASETS
#
#########################################################################################

def get_dataset_abalone():
    datafolder = '../../datasets'
    dataset_name = 'Abalone'
    datafile = glob.glob(os.path.join(
        datafolder, dataset_name, '*.data'))[0]
    D = np.loadtxt( open(datafile, 'r'), delimiter = ',', skiprows = 0, usecols = (1,2,3,4,5,6,7,8) )

    D = sk.preprocessing.scale(D)
    X = D[:,:-1]
    y = D[:,-1]

    return X,y

def get_dataset_housing():
    datafolder = '../../datasets'
    dataset_name = 'Housing'
    datafile = glob.glob(os.path.join(
        datafolder, dataset_name, '*.data'))[0]
    D = np.loadtxt( open(datafile, 'r'), delimiter = ',', skiprows = 0 )

    D = sk.preprocessing.scale(D)
    X = D[:,:-1]
    y = D[:,-1]

    return X,y


def get_dataset_fc():
    num_samples = 150
    x1 = np.random.uniform(0,1, num_samples)
    x2 = np.random.uniform(0,1, num_samples)
    e = np.random.uniform(0,0.001, num_samples)    
    y = (x1 ** 2) + (-2 * x1 * x2) + (3 * x2 ** 2) - 4 + e
    X = np.c_[x1,x2]
    return X,y    

def get_dataset_jpn():    
    num_samples = 100
    x1 = np.random.poisson(1, num_samples)
    x2 = np.random.poisson(1, num_samples)
    x3 = np.random.uniform(0,1, num_samples)
    x4 = np.random.uniform(0,1, num_samples)
    y = x2 * (x4 ** 2) + (x1 ** 2) * x3 + 5
    X = np.c_[x1,x2,x3,x4]
    
    return X,y       

#########################################################################################
#
#   GET ESTIMATORS
#
#########################################################################################

def get_estimator_tunedsvr_dataset(X,y):
    X_train, X_test, y_train, y_test = cv.train_test_split(X,y, train_size = 0.66)
    evaluator = cv.StratifiedKFold(y_train, n_folds = 3)
    search_grid = dict(
                C = [0.5, 5, 50, 500, 5000],
                gamma = [0.1, 0.3, 0.9], 
                )
    searcher = sk.grid_search.GridSearchCV(
            sk.svm.SVR(),
            param_grid = search_grid,
            cv = evaluator)

    searcher.fit(X_train, y_train)
    return searcher.best_estimator_
    
def get_estimator_eprr_dataset_jpn():
    return eprr.EPRR(
            regularization_penalty = 0.8,
            epsilon = 1E-1,
            maxnum_terms = 2,
            pop_size = 100,
            mu = 30,
            lambda_ = 30,
            num_generations = 20,
            mutpb = 0.75,
            cxpb = 0.25)

def get_estimator_eprr_dataset_fc():
    return eprr.EPRR(
            regularization_penalty = 0.8,
            epsilon = 1E-1,
            maxnum_terms = 3,
            pop_size = 100,
            mu = 30,
            lambda_ = 30,
            num_generations = 20,
            mutpb = 0.75,
            cxpb = 0.25)

def get_estimator_eprr_dataset_abalone():
    return eprr.EPRR(
            regularization_penalty = 0.8,
            epsilon = 1E-1,
            maxnum_terms = 5,
            train_size = 0.7,
            cross_validations = 3,
            include_dataset = True,
            pop_size = 100,
            mu = 10,
            lambda_ = 50,
            num_generations = 50,
            mutpb = 0.25,
            cxpb = 0.125)

def get_estimator_eprr_dataset_housing():
    return eprr.EPRR(
            regularization_penalty = 0.8,
            epsilon = 1E-1,
            maxnum_terms = 5,
            train_size = 0.7,
            cross_validations = 3,
            include_dataset = True,
            pop_size = 100,
            mu = 10,
            lambda_ = 50,
            num_generations = 50,
            mutpb = 0.25,
            cxpb = 0.125)


#########################################################################################
#
#   EVALUATION FUNCTIONS
#
#########################################################################################

def make_dataset_scorer(X_train, y_train, X_test = None, y_test = None,
        score_fn = sk.metrics.r2_score ):
    '''
    Make a dataset based scorer that accepts a estimator
    '''

    if X_test is None:
        X_test = X_train
    if y_test is None:
        y_test = y_train

    def __tester__(estimator):
        e = estimator.fit(X_train, y_train)
        return score_fn(y_test, e.predict(X_test))

    return __tester__

def scorer(dataset_scorer, estimator, n = 10, verbose = True):
    '''
        Evaluate the score of estimator using the given dataset.
    '''
    s = np.zeros( (n,1) )
    for row in range(n):
        if verbose:
            print("Evaluating #%d of %d"%(row+1,n))
        s[row] = dataset_scorer(estimator)
    return np.percentile(s, [0, 25, 50, 75, 100])

#########################################################################################
#
#   SCORING DATASETS x ESTIMATORS
#
#########################################################################################
def show(f):
    print("%20s: %s"%(f.__name__, f()))

def score_fc_eprr():
    X,y = get_dataset_fc()
    dataset_scorer = make_dataset_scorer(X,y)
    estimator = get_estimator_eprr_dataset_fc()
    return scorer(dataset_scorer, estimator)

def score_jpn_eprr():
    X,y = get_dataset_jpn()
    dataset_scorer = make_dataset_scorer(X,y)
    estimator = get_estimator_eprr_dataset_jpn()
    return scorer(dataset_scorer, estimator)

def score_abalone_eprr():
    X,y = get_dataset_abalone()
    num_samples, _ = X.shape
    rand_rows = np.random.choice(num_samples, 200, replace = False)

    X_train = X[rand_rows,:]
    y_train = y[rand_rows]

    X_test = X [-rand_rows,:]
    y_test = y [-rand_rows]

    dataset_scorer = make_dataset_scorer(X_train, y_train, X_test, y_test)
    estimator = get_estimator_eprr_dataset_abalone()
    return scorer(dataset_scorer, estimator)

def score_housing_eprr():
    X,y = get_dataset_housing()

    dataset_scorer = make_dataset_scorer(X,y)
    estimator = get_estimator_eprr_dataset_housing()
    return scorer(dataset_scorer, estimator)
    
def score_fc_svr():
    X,y = get_dataset_fc()
    dataset_scorer = make_dataset_scorer(X,y)
    estimator = get_estimator_tunedsvr_dataset(X,y)
    return scorer(dataset_scorer, estimator)

def score_jpn_svr():
    X,y = get_dataset_jpn()
    dataset_scorer = make_dataset_scorer(X,y)
    estimator = get_estimator_tunedsvr_dataset(X,y)
    return scorer(dataset_scorer, estimator)

def score_abalone_svr():
    X,y = get_dataset_abalone()
    dataset_scorer = make_dataset_scorer(X,y)
    estimator = get_estimator_tunedsvr_dataset(X,y)
    return scorer(dataset_scorer, estimator)

def score_housing_svr():
    X,y = get_dataset_housing()
    dataset_scorer = make_dataset_scorer(X,y)
    estimator = get_estimator_tunedsvr_dataset(X,y)
    return scorer(dataset_scorer, estimator)



if __name__ == '__main__':
    # FC POLYNOMIAL SET
    #show(score_fc_eprr)
    show(score_fc_svr)
    # JPN POLYNOMIAL SET
    #show(score_jpn_eprr)
    show(score_jpn_svr)
    # ABALONE SET
    #show(score_abalone_eprr)
    show(score_abalone_svr)
    # HOUSING SET
    #show(score_housing_eprr)
    show(score_housing_svr)
