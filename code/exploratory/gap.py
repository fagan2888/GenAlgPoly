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
            self.terms = np.array( degrees )
        else:
            self.terms = np.array( degrees, dtype = dtype)
        self.num_terms, self.num_vars = self.terms.shape
        
        self.coef_ = None
        self.intercept_ = None
                
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
    def __termrepr__(self, t):
        return "".join(["X_{%d}^{%d}"%(i+1,d) for (i,d) in enumerate(t) if not(d == 0)])
        
    def __repr__(self):
        '''Computes a string representation of the instance.
            
            Returns:
                a string with the exponents matrix.
        '''
        if self.coef_ is None:
            return self.terms.__repr__()
        else:
            terms_str = [self.__termrepr__(t) for t in self.terms]
            return "".join( ['%+5.2f%s'%p for p in zip(self.coef_, terms_str)]) + '%+5.2f'%self.intercept_
                    
def mutate_polyterm(p, num_mutations = 1, valid_degrees = [0,1,2]):
    z = np.copy(p.terms)
    rows, cols = z.shape
    for _ in range(num_mutations):
        r = np.random.randint(rows)
        c = np.random.randint(cols)
        z[r,c] = np.random.choice(a = valid_degrees)
    return PolyTerms(z)
        

    
def random_matrix(num_terms = 2, num_vars = 2, valid_degrees = [0,1,2]):
    z = np.zeros( (num_terms, num_vars) )
    for r in range(num_terms):
        for c in range(num_vars):
            z[r,c] = np.random.choice(a = valid_degrees)
    
    return z
    
def random_polyterm(num_terms = 2, num_vars = 2, valid_degrees = [0,1,2]):        
    return PolyTerms(random_matrix(num_terms, num_vars, valid_degrees))
        
class RandomPolyTerms(PolyTerms):
    def __init__(self,num_terms, num_vars, valid_degrees):
        PolyTerms.__init__(self, np.zeros( (num_terms, num_vars)))
        self.terms = random_polyterm(num_terms, num_vars, valid_degrees).terms
        
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
        
def test_loss(x,y):
    return np.dot(x - y, x - y)
    
rms_loss = metrics.make_scorer( test_loss,
    greater_is_better = False,
    needs_threshold = False
)

class PolyRegression_Scorer(AbstractPolyScorer):
    '''Compute the score of SVR after polynomial
    '''
    def __init__(self, dataset, estimator):
        AbstractPolyScorer.__init__(self)
        self.X = np.copy(dataset.data)
        self.y = np.copy(dataset.target)
        #
        #
        self.num_samples, self.num_features = self.X.shape
        self.estimator = estimator
        self.scores = None
        
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
            scoring = rms_loss)
        return self.scores
        

class PolyDataset:
    def __init__(self, data_features, target_feature):
        num_samples = len(data_features[0])
        num_features = len(data_features)
        self.data = np.zeros( (num_samples, num_features) )
        for i,f in enumerate(data_features):
            self.data[:,i] = f
        self.target = np.array(target_feature)
#
#######################################################################################
#
#######################################################################################
#
#######################################################################################
#
#######################################################################################
#
        
def test(tests = ['basic', 'regression', 'search']):
    '''Run a series of tests and checks on PolyTerms instances
    '''
    #
    #######################################################################################
    #
    #######################################################################################
    #
    if 'basic' in tests:
        print('\nTesting basic evaluation.')
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
        print('\nTesting random polyterm generator.')
        print('\tRandom PolyTerm:\n%s'%random_polyterm(4,2,[0,1,2,0.5]))
    #
    #######################################################################################
    #
    #######################################################################################
    #
    if 'regression' in tests:
        print('\nTesting regression.')
        print('\n\tBoston House Prices\n')
        #
        scorer = PolyRegression_Scorer(
            datasets.load_boston(),
            linear_model.LinearRegression()
            )
        #
        poly = random_polyterm(4, scorer.num_features, [0,1,2,0.5])
        print('\tPolyTerms:\n%s'%poly)
        #
        scorer.score(poly)
        print('\tEstimator coefs:\t%s'%(scorer.estimator.coef_))
        print('\tEstimator intercept:\t%s'%(scorer.estimator.intercept_))
        print('\tExpected score (error):\t%5f ±%5f'%(scorer.scores.mean(), scorer.scores.std()))
    
    if 'deap' in tests:
        import array
        import random

        from deap import algorithms
        from deap import base
        from deap import creator
        from deap import tools

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        # Attribute generator
        toolbox.register("attr_bool", random.randint, 0, 1)

        # Structure initializers
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 100)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        def evalOneMax(individual):
            return sum(individual),

        toolbox.register("evaluate", evalOneMax)
        toolbox.register("mate", tools.cxTwoPoints)
        toolbox.register("mutate", tools.mutFlipBit, indpb = 0.05)
        toolbox.register("select", tools.selTournament, tournsize = 3)
    
        pop = toolbox.population(n=100)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", tools.mean)
        stats.register("std", tools.std)
        stats.register("min", min)
        stats.register("max", max)

        algorithms.eaSimple(pop, toolbox,
            cxpb = 0.5,
            mutpb = 0.2,
            ngen = 10,
            stats = stats,
            halloffame = hof,
            verbose = True )
    #
    #######################################################################################
    #
    #######################################################################################
    #
    if 'search' in tests:        
        print('\nTesting polynomial search.')
        #
        from deap import algorithms
        from deap import base
        from deap import creator
        from deap import tools
        #
        max_num_terms = 4
        valid_degrees = [0, 1, 2]
        #        
        num_samples = 100
        x1 = np.random.uniform(0,1, num_samples)
        x2 = np.random.uniform(0,1, num_samples)
        e = np.random.uniform(0,0.0001, num_samples)    
        y = 1 * x1 * x2 + 3 * x1 ** 2 + x2 ** 2 + e
        pd = PolyDataset([x1, x2], y)
        
        #num_samples = 100
        #x1 = np.random.poisson(size = num_samples)
        #x2 = np.random.poisson(size = num_samples)
        #x3 = np.random.uniform(0,1, num_samples)
        #x4 = np.random.uniform(0,1, num_samples)
        #y = x2 * (x4 ** 2) + (x1 ** 2) * x3 + 5
        #pd = PolyDataset( [x1, x2, x3, x4], y)
        
        
        scorer = PolyRegression_Scorer(
            pd,
            linear_model.LinearRegression() )
        #    
        creator.create('PolyTerms_Fitness', base.Fitness, weights = (1.0,))
        creator.create('PolyTerms_Individual', PolyTerms, fitness = creator.PolyTerms_Fitness)
        #
        
        toolbox = base.Toolbox()
        #
        #   Individual and Population creation
        #
        #
        toolbox.register('individual', creator.PolyTerms_Individual,
            random_matrix(
                num_terms = max_num_terms,
                num_vars = scorer.num_features,
                valid_degrees = valid_degrees
            )
        )
        #
        toolbox.register('population', tools.initRepeat, 
            list,
            toolbox.individual
        )
        #
        #   Genetic operators
        #
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

        def mutate(p, num_mutations = 1, valid_degrees = [0,1,2]):
            z = np.copy(p.terms)
            
            rows, cols = z.shape
            for _ in range(num_mutations):
                r = np.random.randint(rows)
                c = np.random.randint(cols)
                z[r,c] = np.random.choice(a = valid_degrees)
                
            return (creator.PolyTerms_Individual(z), )
                        
        toolbox.register('mate', mate )
        toolbox.register('mutate', mutate,
            num_mutations = 1,
            valid_degrees = valid_degrees )
        #
        #   Fitness
        #
        toolbox.register('evaluate', scorer.score)
        toolbox.register('select', tools.selTournament, tournsize = 6)
        #
        #   Evolution
        #
        pop = toolbox.population(n = 400)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda x: x.fitness.values)
        stats.register('min', min)
        stats.register('mean', tools.mean)
        stats.register('max', max)
        stats.register('std', tools.std)        
        algorithms.eaMuPlusLambda( pop, toolbox,
            mu = 30,
            lambda_ = 70,
            cxpb = 0.25,
            mutpb = 0.05,
            ngen = 50,
            stats = stats, halloffame = hof, verbose = True )
            
        print(hof[0])
        print(hof[0].coef_)
        print(hof[0].intercept_)
        print(hof[0].fitness)    
    #    
    #
    #
    print('\nDone.')
    #
    
if __name__ == '__main__':
    test(['search'])        