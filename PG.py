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

hof = None
log = None
mstats = None

data = {}
pop = []
args = 0

# Files
tests = sys.argv[2].split(" ")
trains = sys.argv[3].split(" ")
outs = sys.argv[4].split(" ")
logs_trains = sys.argv[5].split(" ")
logs_tests = sys.argv[6].split(" ")

# Dataframes
df_train = pd.DataFrame(columns=['Individual', 'Arquivo', 'Erro'])
df_test = pd.DataFrame(columns=['Individual', 'Arquivo', 'Erro'])


def datasets():
    iteration = 0

    global data, args, pop, log, hof, fits, df_train, df_test
    routine()

    # The algorithm has the following idea:
    # 1. We get the entire train archive of our dataset to get our final population with best fitness.
    # 2. After that, we do cross-validation with this population to find the best individuals.
    # 3. Then, we generate the LOGS and DF.
    # 4. Finally, we do the things in KNN.

    while iteration < 5:
        files = cross_validation(iteration)
        ofile = outs[iteration]
        ltrain = logs_trains[iteration]
        ltest = logs_tests[iteration]

        for file in files:
            Main.read(file)
            data = Main.get_data()
            args = Main.get_args()

            for individual in pop:
                evaluate(individual, file)
        Main.statistics(pop, log, hof, df_train, df_test, ofile, ltrain, ltest)
        iteration += 1


def cross_validation(iteration):
    global tests, trains

    if iteration == 0:
        return [tests[0], tests[1], tests[2], tests[3], trains[4]]
    elif iteration == 1:
        return [tests[0], tests[1], tests[2], trains[3], tests[4]]
    elif iteration == 2:
        return [tests[0], tests[1], trains[2], tests[3], tests[4]]
    elif iteration == 3:
        return [tests[0], trains[1], tests[2], tests[3], tests[4]]
    elif iteration == 4:
        return [trains[0], tests[1], tests[2], tests[3], tests[4]]
    else:
        print("Nothing to be done here...moving on...")


# --------------------------------- PG OPERATIONS ---------------------------------
# Safe Division
def analytic_quotient(c, b):
    return c / (math.sqrt(1 + math.pow(b, 2)))


# Transform the expression in our tree in a callable function
def eval_symb(individual):
    func = toolbox.compile(expr=individual)
    # Root Mean Square Error - Ela é a raiz do erro médio quadrático da diferença entre a predição e o valor real.
    global data
    errors = []
    for k, v in data.items():
        errors.append((func(*v) - k) ** 2)

    return math.sqrt(np.mean(errors)),


def evaluate(individual, file):
    func = toolbox.compile(expr=individual)
    global data, df_train, df_test

    for k, v in data.items():
        erro = (func(*v) - k)
        if "train" in file:
            row = {'Individual': individual, 'Arquivo': file, 'Erro': erro}
            df_train = df_train.append(row, ignore_index=True)
        else:
            row = {'Individual': individual, 'Arquivo': file, 'Erro': erro}
            df_test = df_test.append(row, ignore_index=True)


# This is just to set the max number of args in our primitive set. This don't have any influence on fitness.
Main.read(sys.argv[1])
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
    # atual = len(pop)
    # inserts = {'Individual': pop, files[0]: fits[0:atual], files[1]: fits[atual:atual * 2],
    #            files[2]: fits[atual * 2:atual * 3],
    #            files[3]: fits[atual * 3:atual * 4], files[4]: fits[atual * 4:atual * 5]}
    # df = pd.DataFrame(inserts)
    # Main.statistics(pop, log, hof, df, out, lfile)
    # inserts = {'Individual': pop, sys.argv[2]: fits[0:atual], sys.argv[3]: fits[atual:atual*2],
    #            sys.argv[4]: fits[atual*2:atual*3],
    #            sys.argv[5]: fits[atual*3:atual*4], sys.argv[6]: fits[atual*4:atual*5]}
