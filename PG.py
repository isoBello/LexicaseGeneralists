# -*- coding: <UTF-8> -*-

import math
import sys
import numpy as np
import pandas as pd
import operator
import random
import Main

from deap import algorithms, base, tools, creator, gp

toolbox = base.Toolbox()
history = tools.History()
df = pd.DataFrame()

hof = None
log = None
mstats = None

data = {}
pop = []
args = 0
index = 0


def datasets():
    iteration = 0
    indexes = [1, 2, 3, 4, 5, 6]

    global data, args, index, pop, log, hof, df
    routine()
    # The algorithm has the following idea:
    # 1. We get the entire train archive of our dataset to get our final population with best fitness.
    # 2. After that, we do cross-validation with this population to find the best individuals.
    # 3. Then, we generate the LOGS and DF.
    # 4. Finally, we do the things in KNN.

    # while iteration < 5:
    #     Main.read(indexes[iteration])
    #     data = Main.get_data()
    #     args = Main.get_args()
    #     index = indexes[iteration]
    #
    #     iteration += 1

    df = df.drop_duplicates()
    Main.statistics(pop, log, hof, df)


# --------------------------------- PG OPERATIONS ---------------------------------
# Safe Division
def analytic_quotient(c, b):
    return c / (math.sqrt(1 + math.pow(b, 2)))


# Transform the expression in our tree in a callable function
def eval_symb(individual):
    func = toolbox.compile(expr=individual)
    # Root Mean Square Error - Ela é a raiz do erro médio quadrático da diferença entre a predição e o valor real.
    global data, index, df
    errors = []
    for k, v in data.items():
        errors.append((func(*v) - k) ** 2)

    inserts = {'Individual': str(individual), sys.argv[index]: math.sqrt(np.mean(errors))}
    df = df.append(inserts, ignore_index=True)

    return math.sqrt(np.mean(errors)),


# This is just to set the max number of args in our primitive set. This don't have any influence on fitness.
Main.read(1)
data = Main.get_data()
args = Main.get_args()

pset = gp.PrimitiveSet("MAIN", args)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(analytic_quotient, 2)
pset.addEphemeralConstant("rand", lambda: random.randint(-1, 1))
a = {}
for x in range(0, args + 1):
    a['ARG' + str(x)] = 'x' + str(x)

pset.renameArguments(**a)

# Meta-Factory allowing to create that will fulfill the needs of the evolutionary algorithms
creator.create("FitnessMin", base.Fitness, weights=[-1.0])
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

# Contains the evolutionary operators
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", eval_symb)
toolbox.register("select", tools.selTournament, tournsize=7)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=6))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=6))
toolbox.decorate("mate", history.decorator)
toolbox.decorate("mutate", history.decorator)


def routine():
    random.seed(318)

    global data, pop, log, hof, mstats
    pop = toolbox.population(n=10)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)

    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.9, 0.1, 10, stats=mstats,
                                   halloffame=hof, verbose=True)
