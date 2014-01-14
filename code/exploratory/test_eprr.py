#!/opt/local/bin/python

import sklearn as sk
import numpy as np
import eprr
import os
import glob

def tester(f):
    def deco():
        print('\n********** START TESTING [%s] **********\n'%f.__name__)
        (p,s,e) = f()
        print('\n\tbest poly:\t%s\n\twith score:\t%s\n\tand error:\t%s'%(p, s, e))
        print('\n********** END TESTING [%s] **********\n'%f.__name__)
    return deco

@tester
def test_polyterms():
    poly = PolyTerms([
        [0,1,0],
        [2,0,0],
        [1,1,1],
        [0.5,3,1] ])
    print('\tPolyTerms:\n%s'%poly)
    print('\tVars:\t%d\n\tTerms\t:\t%d'%(poly.num_vars,poly.num_terms))
    X = np.array([
        [1,2,1],
        [2,2,2 ]])
    print('\tData:\n%s'%X)
    print('\tTransform:\n%s'%poly(X))
    #
    print('\nTesting polyterm simplification.')
    p0 = PolyTerms([[0,1], [1,1], [2,1], [0,1]])
    print('\tPolyTerms:\n%s'%p0)
    p0.coef_ = [2, 3, 1E-6, 2]
    p1 = p0.simplify()
    print('\tReduced PolyTerms:\n%s'%p1)
    return (p1,0,0)
 
#        
def get_dataset_fc():
    num_samples = 150
    x1 = np.random.uniform(0,1, num_samples)
    x2 = np.random.uniform(0,1, num_samples)
    e = np.random.uniform(0,0.001, num_samples)    
    y = (x1 ** 2) + (-2 * x1 * x2) + (3 * x2 ** 2) - 4 + e
    X = np.c_[x1,x2]
    return X,y    

#
def get_dataset_jpn():    
    num_samples = 100
    x1 = np.random.poisson(1, num_samples)
    x2 = np.random.poisson(1, num_samples)
    x3 = np.random.uniform(0,1, num_samples)
    x4 = np.random.uniform(0,1, num_samples)
    y = x2 * (x4 ** 2) + (x1 ** 2) * x3 + 5
    X = np.c_[x1,x2,x3,x4]
    
    return X,y       

#
@tester
def test_polysearch_jpn():
    X,y = get_dataset_jpn()

    pop_size = 100
    mu_rate = 30.0/pop_size
    lambda_rate = 30.0/pop_size
    e = eprr.EPRR(
            regularization_penalty = 0.8,
            epsilon = 1E-1,
            verbose = False,
            maxnum_terms = 2,
            pop_size = pop_size,
            mu = int(mu_rate * pop_size),
            lambda_ = int(lambda_rate * pop_size),
            num_generations = 20,
            mutpb = 0.75,
            cxpb = 0.25,
            )
    e = e.fit(X,y)
    error = e.poly_.fitness
    p = e.poly_.simplify()
    return (p,p.score(X,y), error)

@tester
def test_polysearch_fc():
    X,y = get_dataset_fc()

    pop_size = 100
    mu_rate = 30.0/pop_size
    lambda_rate = 30.0/pop_size
    e = eprr.EPRR(
            regularization_penalty = 0.8,
            epsilon = 1E-1,
            verbose = False,
            maxnum_terms = 3,
            pop_size = pop_size,
            mu = int(mu_rate * pop_size),
            lambda_ = int(lambda_rate * pop_size),
            num_generations = 20,
            mutpb = 0.75,
            cxpb = 0.25,
            )
    e = e.fit(X,y)
    error = e.poly_.fitness
    p = e.poly_.simplify()
    return (p,p.score(X,y), error)


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

@tester
def test_abalone():
    X,y = get_dataset_abalone()
    num_samples, _ = X.shape
    rand_rows = np.random.choice(num_samples, 200, replace = False)

    X_train = X[rand_rows,:]
    y_train = y[rand_rows]

    X_test = X [-rand_rows,:]
    y_test = y [-rand_rows]

    pop_size = 100
    mu_rate = 0.1
    lambda_rate = 0.50
    mutpb = 0.25
    cxpb = 0.125
    num_generations = 50

    e = eprr.EPRR(
            verbose = True,
            regularization_penalty = 0.8,
            epsilon = 1E-1,
            maxnum_terms = 5,
            train_size = 0.7,
            cross_validations = 3,
            include_dataset = True,
            pop_size = pop_size,
            mu = int(mu_rate * pop_size),
            lambda_ = int(lambda_rate * pop_size),
            num_generations = num_generations,
            mutpb = mutpb,
            cxpb = cxpb,
            )
    e = e.fit(X_train,y_train)
    error = e.poly_.fitness
    p = e.poly_.simplify()
    return (p, p.score(X_test,y_test), error)
    

def make_dataset_scorer(X_train, y_train, X_test = None, y_test = None ):

    if X_test is None:
        X_test = X_train
    if y_test is None:
        y_test = y_train

    def __tester__(estimator):
        e = estimator.fit(X_train, y_train)
        error = e.poly_.fitness.values[0]
        p = e.poly_.simplify()
        return (p, p.score(X_test, y_test), error)

    return __tester__

def scorer(dataset_scorer, estimator, cols = 1, n = 25, verbose = False):
    '''
        cols is the column mask in the scores array that provides the statistics;
            cols == 0 ==> score (r1)
            cols == 1 ==> error (rmse)
    '''
    s = np.zeros( (n,2) )
    for row in range(n):
        if verbose:
            print("Evaluating #%d of %d"%(row+1,n))
        sc = dataset_scorer(estimator)[1:]
        s[row,:] = sc

    return np.percentile(s[:,cols], [0, 25, 50, 75, 100], axis = 0)


def test_scorer():
    X,y = get_dataset_jpn()
    ds = make_dataset_scorer(X,y)

    pop_size = 100
    mu_rate = 30.0/pop_size
    lambda_rate = 30.0/pop_size
    estimator = eprr.EPRR(
            regularization_penalty = 0.8,
            epsilon = 1E-1,
            verbose = False,
            maxnum_terms = 3,
            pop_size = pop_size,
            mu = int(mu_rate * pop_size),
            lambda_ = int(lambda_rate * pop_size),
            num_generations = 20,
            mutpb = 0.75,
            cxpb = 0.25,
            )
    scores = scorer(ds, estimator, n = 3, cols = 1, verbose = True)
    print(scores)

def score_dataset_jpn():
    X,y = get_dataset_jpn()
    dataset_scorer = make_dataset_scorer(X,y)

    estimator_eprr = get_estimator_eprr_dataset_jpn()
    
    estimators = [estimator_eprr]
    for estimator in estimators:
        print( scorer(
            dataset_scorer,
            estimator,
            verbose = True ) )

def get_estimator_eprr_dataset_jpn():
    pop_size = 100
    mu_rate = 30.0/pop_size
    lambda_rate = 30.0/pop_size
    return eprr.EPRR(
            regularization_penalty = 0.8,
            epsilon = 1E-1,
            verbose = False,
            maxnum_terms = 3,
            pop_size = pop_size,
            mu = int(mu_rate * pop_size),
            lambda_ = int(lambda_rate * pop_size),
            num_generations = 20,
            mutpb = 0.75,
            cxpb = 0.25,
            )




def test(x):
    while x:
        t = x.pop(0)
        if t == 'polyterms':
            test_polyterms()
        if t == 'polysearch_fc':
            test_polysearch_fc()
        if t == 'polysearch_jpn':
            test_polysearch_jpn()        
        if t == 'abalone':
            test_abalone()
        if t == 'scorer':
            test_scorer()
        if t == 'score_dataset_jpn':
            score_dataset_jpn()

if __name__ == '__main__':
   test(['score_dataset_jpn'])
