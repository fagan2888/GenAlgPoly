#!/opt/local/bin/python

import numpy as np
import eprr

def test(f):
    def deco():
        print('\n********** START TESTING [%s] **********\n'%f.__name__)
        f()
        print('\n********** END TESTING [%s] **********\n'%f.__name__)
    return deco

@test
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
@test
def test_search():
    X,y = get_dataset_jpn()
    print('\tX shape:\t%s'%str(X.shape))
    print('\ty shape:\t%s'%str(y.shape))

    pop_size = 100
    mu_rate = 30.0/pop_size
    lambda_rate = 30.0/pop_size
    e = eprr.EPRR(
            regularization_penalty = 0.8,
            epsilon = 1E-1,
            verbose = True,
            maxnum_terms = 2,
            pop_size = pop_size,
            mu = int(mu_rate * pop_size),
            lambda_ = int(lambda_rate * pop_size),
            num_generations = 20,
            mutpb = 0.75,
            cxpb = 0.25,
            )
    print('\tSTART "fit"...')
    e = e.fit(X,y)
    print('\tEND "fit"')
    error = e.poly_.fitness
    p = e.poly_.simplify()
    print('\n\tbest poly:\t%s\n\twith score:\t%s\n\tand error:\t%s'%(p,p.score(X,y), error))

    
def test(x):
    if 'polyterms' in x:
        test_polyterms()
    if 'search' in x:
        test_search()

if __name__ == '__main__':
   test(['search'])
