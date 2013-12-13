#!/usr/bin/env python

#
#   Fitness strategy for sklearn
#
#
#       1. Define a searcher:
#           searcher = GridSearchCV(
#               estimator,
#               param_grid,
#               
#               )
import numpy as np
from sklearn import cross_validation as cv
from sklearn import svm
from sklearn import preprocessing
from sklearn import datasets
from deap import algorithms, base, creator, tools
#
#   Supress deprecated warnings
#
import warnings
warnings.filterwarnings("ignore")
#       
def save_data(filename, data):
    f = open(filename, 'w')
    f.write('Id,Solution\n')
    i=1
    for x in data:
        f.write('%d,%d\n' % (i,x))
        i += 1
    f.close()
#    
def load_data(root_folder = '../../datasets/'):
    '''
    Loads data from project data folder and returns a triple (train, labels, test) where:
        - train and labels are used to setup classifiers
        - test is used to produce predictions
    '''
    load_file = lambda f,s: np.loadtxt( open(f, 'r'), delimiter = ',', skiprows = s )
    data_folder = root_folder + 'Auto-Mpg/'    
    data = load_file( data_folder + 'auto_no_missing.data', 0)
    
    return data
    
class PolyTransform:
    '''Represent a set of monomial transformations.
    
    It is assumed that a set of n variables, X1 .. Xn, is fixed.
    An instance of this class defines a set of transformations, Z1 .. Zm, where
        - each Zj = (X1 ** E1j) * .. *(Xn ** Enj) and
        - the exponents Eij are integers or fractions.
    
    Example:
    
        Given the dataset X1,X2
        
        X1  X2
        0   1
        1   1
        2   2
        
        and the PolyTransform [Z1 = [1,0], Z2 = [2,2], Z3 = [2,0]] we obtain
        Z1  Z2  Z3
        0   0   0
        1   1   1
        2   16  4
    '''
    def __init__(self, degrees, dtype = np.int8):
        '''Defines a monomial transformation.
        
        Args:
            degrees (list of lists of integers/fractions): the set of transformations.
                - each element of degrees represents a single monomial transformation;
                - all the elements of degrees are expected to have the same size;

        Kwargs:
            dtype (numpy type): the numpy.array type to hold the degrees
            
        Returns:
            an instance of PolyTransform
            
        Defines:
            .num_terms (int): the number of monomial terms given in degrees
            .num_vars (int): the number of variables (X1 .. Xn) in each monmial term
            
        No check is performed.
        '''
        self.terms = np.array( degrees, dtype = dtype)
        self.num_terms, self.num_vars = self.terms.shape
                
    def __call__(self, X, extend = False):
        '''Evaluates the transformation on a given dataset, X.
        
            Args:
                X (numpy.array): A dataset whose number of features (columns) must be equal to .num_vars;
            Kwargs:
                extend (boolean): whether the transformation includes the original dataset or not;
        
            Returns:
                the transformed dataset
        '''
        Z = np.array([
            [ np.prod(np.power(x, t)) for t in self.terms ]
            for x in X])
        if extend:
            return np.c_[X,Z]
        else:
            return Z
            
    def __repr__(self):
        '''Computes a string representation of the instance.
            
            Returns:
                a string with the exponents matrix.
        '''
        return self.terms.__repr__()

class AbstractScoreProcess:
    def __init__(self, dataset):
        self.dataset = dataset
        self.score = None
        self._changed_ = False
        
    def _compute_score_(self):
        self.score = cv.cross_val_score(estimator, X_train, y_train, cv = 5)
        
    def score(self):
        if self._changed_:
            self._compute_score_()
            self._changed_ = False
            
        return score
        
def utest():
    '''Run a series of tests and checks on PolyTransform instances
    '''
    #
    #
    #
    print("\nTesting basic evaluation.")
    nt = PolyTransform([[0,1,0],[2,0,0],[1,1,1],[0.5,3,1]])
    print("\tPolyTransform\t:\n%s"%nt)
    print("\tVars\t:\t%d\n\tTerms\t:\t%d"%(nt.num_vars,nt.num_terms))
    X = np.array([[1,2,1],[2,2,2]])
    print("\tData\t:\n%s"%X)
    print("\tTransform\t:\n%s"%nt(X, extend = True))
    #
    #
    #
    print("\nTesting classification.")
    print("\n\tBoston House Prices\n")
    #
    d = datasets.load_boston()
    X = preprocessing.normalize(d.data)
    y = (d.target - d.target.mean())/d.target.std()
    #
    num_samples, num_features = X.shape
    X_train, X_test, y_train, y_test = cv.train_test_split(X,y)
    print('\tData set:\n\t\tnum_samples\t:\t%s\n\t\tnum_features\t:\t%s'%(num_samples, num_features))
    print('\tTrain set:\n\t\tnum_samples\t:\t%s\n\t\tnum_features\t:\t%s'%X_train.shape)
    print('\tTest set:\n\t\tnum_samples\t:\t%s\n\t\tnum_features\t:\t%s'%X_test.shape)
    # 
    p = PolyTransform(np.eye(num_features))
    Z_train = p(X_train)
    Z_test = p(X_test)
    #
    #
    #
    estimator = svm.SVR()
    scores = cv.cross_val_score(estimator, X_train, y_train, cv = 5)
    print("\tScore:\n\t\t%5f ±%5f"%(scores.mean(), scores.std()))
    #
    #
    #
    print("\nTesting genetic algorithms.")
    #
    creator.create("PolyTransformFitness", base.Fitness, weights = (1.0,))
    creator.create("PolyTransformIndividual", PolyTransform, fitness = creator.PolyTransformFitness)
    toolbox = base.Toolbox()
    toolbox.register("crossover", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutGaussian, mu = 0.0, std = 1.0)
    #
    #
    #
    print("\nDone.")
    #
    
if __name__ == '__main__':
    utest()        