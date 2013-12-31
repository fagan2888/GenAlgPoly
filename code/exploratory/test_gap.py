#!/usr/bin/env python
    
from gap import *
                   
                   
def test_basic():
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
    print('\nTesting polyterm simplification.')
    p0 = PolyTerms([[0,1], [1,1], [2,1], [0,1]])
    print('\tPolyTerms:\n%s'%p0)
    p0.coef_ = [2, 3, 1E-6, 2]
    p1 = p0.simplify()
    print('\tReduced PolyTerms:\n%s'%p1)
    

def test_regression():
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

def test_deap():
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
    toolbox.register("individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_bool, 
        100 )
    toolbox.register("population",
        tools.initRepeat,
        list,
        toolbox.individual )

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

def test_search():
    s = PolySearcher(get_dataset_fc(),
        {   'verbose': True,
            'pop_size': 750,
            'num_generations': 50,
            'mutpb': 0.1
        } )
    hof = s()
    fittest = hof[0]
    fitness = fittest.fitness
    fittest = fittest.simplify(1e-2)
    print('\tFound:\t\t%s\n\twith error:\t%s'%(fittest,fitness))
    
    
def test(tests = ['basic', 'regression', 'search']):
    '''Run a series of tests and checks on PolyTerms instances
    '''
    if 'basic' in tests:
        test_basic()
    if 'regression' in tests:
        test_regression()        
    if 'deap' in tests:
        test_deap()
    if 'search' in tests:        
        test_search()
        
if __name__ == '__main__':
    test(["search"])
