#
#   TODO add regularisation to error
#
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
        '''Reduces the polynomial expression by cutting terms with small coefficients and
        joining terms with common monomial.

        Args:
            epsilon (double): the threshold to cut to zero

        Returns:
            an instance of PolyTerms
        '''

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
        if len(c_) > 0:
            p = PolyTerms(np.array(t_))
            p.coef_ = c_
        else:
            p = PolyTerms(np.zeros( (1,self.num_vars) ) )
            p.coef_ = [0.0]
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
            return np.c_[Z,X]
        else:
            return Z

    def score(self, X, y):
        Z = self(X)
        y_hat = np.dot(Z, self.coef_) + self.intercept_
        y_mean = np.mean(y)
        u = np.dot(y - y_hat, y - y_hat)
        v = np.dot(y - y_mean, y - y_mean )
        return (1. - u/v)

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

class EPRR:
    def __init__(self,
        regularization_penalty = 1.0,
        epsilon = 1E-7,
        valid_degrees = (0,1,2),
        maxnum_terms = 4,
        train_size = 0.05,
        cross_validations = 3,
        include_dataset = False,
        pop_size = 400,
        num_generations = 50,
        num_mutations = 1,
        tournment_size = 6,
        mu= 30,
        lambda_ = 70,
        cxpb = 0.25,
        mutpb = 0.05,
        hof_size = 1,
        verbose = False ):

        #
        #   Parameters
        #
        self.regularization_penalty = regularization_penalty
        self.epsilon = epsilon
        self.valid_degrees = valid_degrees
        self.maxnum_terms = maxnum_terms
        self.train_size = train_size
        self.cross_validations = cross_validations
        self.include_dataset = include_dataset
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.num_mutations = num_mutations
        self.tournment_size = tournment_size
        self.mu= mu
        self.lambda_ = lambda_
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.hof_size = hof_size
        self.verbose = verbose

        #
        #   Attributes
        #
        self.ga_toolbox_ = None
        self.poly_ = None
        self.ga_estimator_ = None
        self.ga_scorer_ = None
        self.ga_operators_ = None
        self.ga_stats_ = None
        self.ga_hall_of_fame_ = None

        #
        #   "Hidden"
        #
        self.__num_train_features = None
        self.__num_train_samples = None
        self.__trained__ = False

    def show(self, t):
        if self.verbose:
            print(t)

    def __random_matrix(self):
        '''Initializes a matrix with random degrees, compatible with the dataset.

        Args:
            N/A

        Returns:
            an np.array
        '''
        z = np.zeros( (self.maxnum_terms, self.__num_train_features) )
        for r in range(self.maxnum_terms):
            for c in range(self.__num_train_features):
                z[r,c] = np.random.choice(a = self.valid_degrees)
        return z

    def create_ga_estimator(self):
        '''Creates an estimator to compute the coefficients of a polynomial on a dataset.

        Args:
            N/A

        Returns:
            an instance of a sklearn.REGRESSOR
        '''
        return linear_model.LinearRegression()

    def create_ga_scorer(self, X, y):
        '''Creates a scorer based in a dataset. The scorer stores the dataset and an estimator; Given a polynomial, the scorer determines the error of the polynomial prediction.
        '''
        if self.ga_estimator_ is None:
            return None

        def poly_scorer(p):
            Z = p(X, self.include_dataset)
            Z_train, Z_test, y_train, y_test = cv.train_test_split( Z, y,
                    train_size = self.train_size )
            self.ga_estimator_.fit(Z_train, y_train)
            p.coef_ = self.ga_estimator_.coef_[:p.num_terms]
            p.intercept_ = self.ga_estimator_.intercept_
            #
            #   Compute number of significative terms
            #
            ps = p.simplify(epsilon = self.epsilon)
            k = ps.num_terms
            penalty = self.regularization_penalty ** k
            #
            #   Evaluate (regularized) score by cross validation
            #
            scores = cv.cross_val_score(
                    self.ga_estimator_,
                    Z_test,
                    y_test,
                    cv = self.cross_validations,
                    scoring = metrics.make_scorer(
                        lambda x,y: penalty * np.dot(x-y,x-y),
                        greater_is_better = False,
                        needs_threshold = False ) )
            return scores

        self.__num_train_samples, self.__num_train_features = X.shape
        return poly_scorer

    def create_ga_operators(self, **kw):
        def mate(x,y):

            x_ = x.terms.copy()
            y_ = y.terms.copy()
            n,m = x_.shape
            nm = n*m

            a = x_.reshape( (1, nm) ).tolist()[0]
            b = y_.reshape( (1, nm) ).tolist()[0]
            c = np.array( tools.cxOnePoint(a,b) )
            a = creator.pt_individual( c[0,:].reshape( (n,m) ) )
            b = creator.pt_individual( c[1,:].reshape( (n,m) ) )
            return (a,b)

        def mutate(p):
            z = np.copy(p.terms)

            rows, cols = z.shape
            for _ in range(self.num_mutations):
                r = np.random.randint(rows)
                c = np.random.randint(cols)
                z[r,c] = np.random.choice(a = self.valid_degrees)

            return ( creator.pt_individual(z), )

        return {'mate': mate, 'mutate': mutate}

    def create_ga_toolbox(self):
        if self.ga_scorer_ is None:
            return None

        creator.create('pt_fitness',
                base.Fitness,
                weights = (1.0,))
        creator.create('pt_individual',
                PolyTerms,
                fitness = creator.pt_fitness)

        tb = base.Toolbox()
        tb.register('individual',
                creator.pt_individual,
                self.__random_matrix())
        tb.register('population',
                tools.initRepeat,
                list,
                tb.individual)
        self.ga_operators_ = self.create_ga_operators()
        for (k,op) in self.ga_operators_.items():
            tb.register(k, op)

        tb.register('evaluate', self.ga_scorer_)
        tb.register('select',
                tools.selTournament,
                tournsize = self.tournment_size)

        return tb

    def create_ga_stats(self):
        stats = tools.Statistics(lambda x: x.fitness.values)
        stats.register('mean', tools.mean)

        return stats

    def create_ga_hall_of_fame(self):
        self.show('Create population')
        pop = self.ga_toolbox_.population(n = self.pop_size)
        self.show('\tpopulation: OK')
        self.show('Create hof')
        hof = tools.HallOfFame(self.hof_size)
        self.show('\thof: OK')
        self.show('\nSTART GA SEARCH')
        algorithms.eaMuCommaLambda( pop, self.ga_toolbox_,
            mu = self.mu,
            lambda_ = self.lambda_,
            cxpb = self.cxpb,
            mutpb = self.mutpb,
            ngen = self.num_generations,
            stats = self.ga_stats_,
            halloffame = hof,
            verbose = self.verbose )
        return hof

    def score(self, X, y):
        if self.__trained__:
            return self.poly_.score(X,y)
        else:
            return None

    def predict(self, X):
        if self.__trained__:
            return self.poly_(X)
        else:
            return None

    def fit(self, X, y):
        self.show('Create GA estimator')
        self.ga_estimator_ = self.create_ga_estimator()
        self.show('\tGA estimator: OK')
        self.show('Create GA scorer')
        self.ga_scorer_ = self.create_ga_scorer(X,y)
        self.show('\tGA scorer: OK')
        self.show('Create GA toolbox')
        self.ga_toolbox_ = self.create_ga_toolbox()
        self.show('\tGA toolbox: OK')
        self.show('Create GA stats')
        self.ga_stats_ = self.create_ga_stats()
        self.show('\tGA stats: OK')
        self.show('Create GA hall of fame')
        self.ga_hall_of_fame_ = self.create_ga_hall_of_fame()
        self.show('\tGA hall of fame: OK')
        self.poly_ = self.ga_hall_of_fame_[0]
        self.__trained__ = True
        return self
