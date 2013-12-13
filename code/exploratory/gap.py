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
from sklearn import linear_model
from deap import algorithms, base, creator, tools
#
#   Supress deprecated warnings
#
import warnings
warnings.filterwarnings('ignore')
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
    
class PolyTerms:
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
        
        and the PolyTerms [Z1 = [1,0], Z2 = [2,2], Z3 = [2,0]] we obtain
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
            an instance of PolyTerms
            
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

class AbstractPolyScorer:
    def __init__(self):
        self.poly = None
        self.scores = None
        
    def compute_score(self):
        return 0.0
        
    def score(self, poly):
        self.poly = poly
        self.scores = None
        return self.compute_score()
        
class Regression_PolyScorer(AbstractPolyScorer):
    '''Compute the score of SVR after polynomial
    '''
    def __init__(self, dataset, estimator):
        AbstractPolyScorer.__init__(self)
        self.X = preprocessing.normalize(dataset.data)
        self.y = (dataset.target - dataset.target.mean())/dataset.target.std()
        #
        self.num_samples, self.num_features = self.X.shape
        self.estimator = estimator
        self.scores = None
        
    def compute_score(self):
        Z = self.poly(self.X)
        self.scores = cv.cross_val_score(
            self.estimator,
            Z,
            self.y,
            cv = 5)
        return self.scores.mean()
        
def utest():
    '''Run a series of tests and checks on PolyTerms instances
    '''
    #
    #
    #
    print('\nTesting basic evaluation.')
    poly = PolyTerms([
        [0,1,0],
        [2,0,0],
        [1,1,1],
        [0.5,3,1] ])
    print('\tPolyTerms\t:\n%s'%poly)
    print('\tVars\t:\t%d\n\tTerms\t:\t%d'%(poly.num_vars,poly.num_terms))
    X = np.array([
        [1,2,1],
        [2,2,2 ]])
    print('\tData\t:\n%s'%X)
    print('\tTransform\t:\n%s'%poly(X))
    #
    #
    #
    print('\nTesting regression.')
    print('\n\tBoston House Prices\n')
    #
    ps = Regression_PolyScorer(
        datasets.load_boston(),
        linear_model.LinearRegression()
        )
    #
    poly = PolyTerms(np.eye(ps.num_features))
    print('\tPolyTerms\t:\n%s'%poly)
    #
    ps.score(poly)
    print('\tScore:\t%5f ±%5f'%(ps.scores.mean(), ps.scores.std()))
    #
    #
    #
    print('\nTesting polynomial search.')
    #
    creator.create('PolyTermsFitness', base.Fitness, weights = (1.0,))
    creator.create('PolyTermsIndividual', PolyTerms, fitness = creator.PolyTermsFitness)
    toolbox = base.Toolbox()
    toolbox.register('crossover', tools.cxOnePoint)
    toolbox.register('mutate', tools.mutGaussian, mu = 0.0, std = 1.0)
    #
    toolbox.register('attr_polyterm',
        np.random.choice,
        a = [0, 1, 2, 3, 0.5, 1.5, 2.5],
        size = ps.num_features )
    #
    
    #
    #
    #
    print('\nDone.')
    #
    
if __name__ == '__main__':
    utest()        