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
from sklearn import metrics
from sklearn import linear_model
from deap import algorithms, base, creator, tools
#
#   Supress deprecated warnings
#
import warnings
warnings.filterwarnings('ignore')
#
#   Save a prediction in the Kaggle submission format
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
#   Load a dataset
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
#
#   Represent a set of monomial transformations
#    
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
    def __init__(self, degrees, dtype = None):
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
        if dtype is None:
            self.terms = np.array( degrees, dtype = np.int8 )
        else:
            self.terms = np.array( degrees, dtype = dtype)
        self.num_terms, self.num_vars = self.terms.shape
        
        self.coef_ = None
        self.intercept_ = None
                
    def simplify(self, epsilon = 1E-5):

        if self.coef_ is None:
            return self, None
        #
        #   remove terms with |coef| < epsilon
        #
        c = np.copy( self.coef_ )            
        t = np.copy( self.terms )
        big = np.abs(c) > epsilon
        t = t[big,:]
        c = c[big]
        #
        #   join common terms, summing the coefficients
        #
        n,_ = t.shape
        cs = list()
        open_terms = list(range(n))
        closed_terms = list()
        t_ = list()
        c_ = list()
        while open_terms:
            i = open_terms.pop()
            si = c[i]
            closed_terms.append(i)
            ti = t[i]
            for j in open_terms:
                if (t[j] == ti).all():
                    open_terms.remove(j)
                    closed_terms.append(j)
                    si += c[j]
            t_.append(list(ti))
            c_.append(si)
        p = PolyTerms(np.array(t_))
        p.coef_ = c_
        p.intercept_ = self.intercept_
        return p
        
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
        if self.coef_ is None:
            return self.terms.__repr__()
        else:
            s = " ".join( ['%+5.4f%s'%p for p in zip(self.coef_, self.terms)]) 
            if not(self.intercept_ is None):
                s = s + ' %+5.4f'%self.intercept_
            return s
#
def random_matrix(num_terms = 2, num_vars = 2, valid_degrees = [0,1,2]):
    z = np.zeros( (num_terms, num_vars) )
    for r in range(num_terms):
        for c in range(num_vars):
            z[r,c] = np.random.choice(a = valid_degrees)    
    return z
#
class AbstractPolyScorer:
    '''Define the interface for a scorer.
    '''
    def __init__(self):
        self.poly = None
        self.scores = None
        
    def compute_score(self):
        return 0.0
        
    def score(self, poly):
        self.poly = poly
        self.scores = None
        return self.compute_score()
#        
class Regression_PolyScorer(AbstractPolyScorer):
    '''Compute the score of SVR after polynomial transformation.
    The score results from cross-validation of a fitted estimator.
    '''
    def __init__(self, dataset, estimator):
        AbstractPolyScorer.__init__(self)
        self.X = np.copy(dataset.data)
        self.y = np.copy(dataset.target)
        #
        #
        self.num_samples, self.num_features = self.X.shape
        self.estimator = estimator
        
    def get_loss(self):
        def loss(x,y):
            return np.dot(x - y, x - y)
        return loss
        
    def compute_score(self):
        Z = self.poly(self.X)
        Z_train, Z_test, y_train, y_test = cv.train_test_split(Z,self.y, train_size = 0.05)
        self.estimator.fit(Z_train, y_train)
        self.poly.coef_ = self.estimator.coef_
        self.poly.intercept_ = self.estimator.intercept_
        self.scores = cv.cross_val_score(
            self.estimator,
            Z_test,
            y_test,
            cv = 3,
            scoring = metrics.make_scorer(
                self.get_loss(),
                greater_is_better = False,
                needs_threshold = False
            )
        )
        return self.scores
#
class PolyDataset:
    def __init__(self, data_features, target_feature):
        num_samples = len(data_features[0])
        num_features = len(data_features)
        self.data = np.zeros( (num_samples, num_features) )
        for i,f in enumerate(data_features):
            self.data[:,i] = f
        self.target = np.array(target_feature)
#        
def get_dataset_fc():
    num_samples = 100
    x1 = np.random.uniform(0,1, num_samples)
    x2 = np.random.uniform(0,1, num_samples)
    e = np.random.uniform(-4,0.0001, num_samples)    
    y = (1.5 * x1 * x2) + (3 * x1 ** 2) + (2 * x2 ** 2) + e
    pd = PolyDataset([x1, x2], y)
    
    return pd    

#
def get_dataset_jpn():    
    num_samples = 100
    x1 = np.random.poisson(1, num_samples)
    x2 = np.random.poisson(1, num_samples)
    x3 = np.random.uniform(0,1, num_samples)
    x4 = np.random.uniform(0,1, num_samples)
    y = x2 * (x4 ** 2) + (x1 ** 2) * x3 + 5
    pd = PolyDataset( [x1, x2, x3, x4], y)
    
    return pd       
#
class PolySearcher:
    DEFAULT_PARAMS = {
        'max_num_terms' : 4,
        'valid_degrees' : [0, 1, 2],
        'pop_size': 400,
        'num_generations': 50,
        'hof_size': 1,
        'num_mutations': 1,
        'tournment_size': 6,
        'mu': 30,
        'lambda_': 70,
        'cxpb': 0.25,
        'mutpb': 0.05,
        'verbose': False,
    }
    
    def __init__(self,dataset, params = None):
        self.params = PolySearcher.DEFAULT_PARAMS
        if not(params is None):
            self.params = dict(self.params, **params)

        self.dataset = dataset
        self._toolbox_ = None
    
    def get_estimator(self):
        return linear_model.LinearRegression()
        
    def get_scorer(self):
        return Regression_PolyScorer(self.dataset, self.estimator)
          
    def get_operators(self, **kw):
        def mate(x,y):

            x_ = x.terms.copy()
            y_ = y.terms.copy()
            n,m = x_.shape
            nm = n*m
        
            a = x_.reshape( (1, nm) ).tolist()[0]
            b = y_.reshape( (1, nm) ).tolist()[0]
            c = np.array( tools.cxOnePoint(a,b) )
            a = creator.PolyTerms_Individual( c[0,:].reshape( (n,m) ) )
            b = creator.PolyTerms_Individual( c[1,:].reshape( (n,m) ) )
            return (a,b)
            
        def mutate(p):
            z = np.copy(p.terms)
        
            rows, cols = z.shape
            for _ in range(self.params['num_mutations']):
                r = np.random.randint(rows)
                c = np.random.randint(cols)
                z[r,c] = np.random.choice(a = self.params['valid_degrees'])
            
            return (creator.PolyTerms_Individual(z), )    
        return {'mate': mate, 'mutate': mutate}
        
    def get_stats(self):
        stats = tools.Statistics(lambda x: x.fitness.values)
        stats.register('mean', tools.mean)
        return stats
       
    def get_toolbox(self):
        #    
        creator.create('PolyTerms_Fitness', base.Fitness, weights = (1.0,))
        creator.create('PolyTerms_Individual', PolyTerms, fitness = creator.PolyTerms_Fitness)
        #
        toolbox = base.Toolbox()
        #
        #   Individual and Population creation
        #
        toolbox.register('individual', creator.PolyTerms_Individual,
            random_matrix(
                num_terms = self.params['max_num_terms'],
                num_vars = self.scorer.num_features,
                valid_degrees = self.params['valid_degrees']
            )
        )
        #
        toolbox.register('population', tools.initRepeat, list, toolbox.individual)
        #
        #   Genetic operators
        #
        operators = self.get_operators()
        for (k,op) in operators.items():
            toolbox.register(k, op )
        #
        #   Fitness
        #
        toolbox.register('evaluate', self.scorer.score)
        toolbox.register('select', tools.selTournament, tournsize = self.params['tournment_size'])
        
        return toolbox
        
    def get_halloffame(self):
        pop = self.toolbox.population(n = self.params['pop_size'])
        hof = tools.HallOfFame(self.params['hof_size'])
        algorithms.eaMuPlusLambda( pop, self.toolbox,
            mu = self.params['mu'],
            lambda_ = self.params['lambda_'],
            cxpb = self.params['cxpb'],
            mutpb = self.params['mutpb'],
            ngen = self.params['num_generations'],
            stats = self.stats,
            halloffame = hof,
            verbose = self.params['verbose'] )
        return hof
        
    def __call__(self):
        self.estimator = self.get_estimator()
        self.scorer = self.get_scorer()
        self.toolbox = self.get_toolbox()
        self.stats = self.get_stats()
        self.halloffame = self.get_halloffame()
        return self.halloffame
        
