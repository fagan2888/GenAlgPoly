#!/opt/local/bin/python

import sklearn as sk
import numpy as np
import eprr
import os
import glob

def tester(f):
    def deco():
        print('\n********** START TESTING [%s] **********\n'%f.__name__)
        f()
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
def test_polysearch():
    print("Search JPN poly")
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
    print('\n\tbest poly:\t%s\n\twith score:\t%s\n\tand error:\t%s'%(p,p.score(X,y), error))

    print("Search FC poly")
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
    print('\n\tbest poly:\t%s\n\twith score:\t%s\n\tand error:\t%s'%(p,p.score(X,y), error))


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
    rand_rows = np.random.choice(num_samples, 100, replace = False)

    X = X[rand_rows,:]
    y = y[rand_rows]

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
    e = e.fit(X,y)
    error = e.poly_.fitness
    p = e.poly_.simplify()
    print('\n\tbest poly:\t%s\n\twith score:\t%s\n\tand error:\t%s'%(p,p.score(X,y), error))

    
def test(x):
    if 'polyterms' in x:
        test_polyterms()
    if 'polysearch' in x:
        test_polysearch()
    if 'abalone' in x:
        test_abalone()

if __name__ == '__main__':
   test(['abalone'])
