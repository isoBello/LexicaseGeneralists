# -*- coding: <UTF-8> -*-
# !/usr/bin/env pypy
import math
import sys
import pyarrow
import os
import numpy as np
import pandas as pd
import operator
import random
import multiprocessing
import Main

from deap import algorithms, base, tools, creator, gp
from fastparquet import write

toolbox = base.Toolbox()
history = tools.History()

hof = None
log = None
mstats = None

data = {}
pop = []
args = 0

# Folders and Files
folders = sys.argv[3]
folds = []
filename = sys.argv[4]


def datasets():
    iteration = 0

    global folders, folds, filename
    # The algorithm has the following idea:
    # 1. We get the entire train archive of our dataset to get our final population with best fitness.
    # 2. After that, we do cross-validation with this population to find the best individuals.
    # 3. Then, we generate the LOGS and DF.
    # 4. Finally, we do the things in KNN.

    while iteration < len(os.listdir(filename)):
        files = cross_validation(iteration)

        for file in files:
            filename = filename + file
            df = pd.read_csv(filename, sep=",")

            with open(filename, 'r') as archive:
                line = archive.readline()

            get_args(len(line) - 1)

            filename = filename.replace(file, "")

            df = df.apply(lambda col: col.astype(float))

            name = folders + "fold" + str(iteration) + "/"
            folds.append(name)

            path = os.path.join(folders, "fold" + str(iteration) + "/")
            try:
                os.mkdir(path)
            except OSError:
                pass

            names = name + file.replace("original", "").replace(".dat", ".pq")
            write(names, df)

        iteration += 10


def cross_validation(iteration):
    files = os.listdir("original")
    return files[0:iteration] + files[iteration+1:]


def get_args(argument):
    global args
    args = argument


# --------------------------------- PG OPERATIONS ---------------------------------
# Safe Division
def analytic_quotient(c, b):
    return c / (math.sqrt(1 + math.pow(b, 2)))


# Transform the expression in our tree in a callable function
def eval_symb(individual):
    func = toolbox.compile(expr=individual)
    # Root Mean Square Error - Ela é a raiz do erro médio quadrático da diferença entre a predição e o valor real.
    global data

    errors = data.apply(lambda linha: func(*linha[:-1]) - linha[-1], axis=1)
    result = math.sqrt(errors.pow(2).mean())

    return result,


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
    pop = toolbox.population(n=1000)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)

    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)


def main():
    global data, pop, log, hof, folds

    for folder in os.listdir(sys.argv[3]):
        folds.append(sys.argv[3] + folder + "/")

    for fold in folds:
        data = pd.read_parquet(fold, engine="pyarrow")
        pool = multiprocessing.Pool()
        toolbox.register("map", pool.map)
        _pop, _log = algorithms.eaSimple(pop, toolbox, 0.9, 0.1, 250, stats=mstats,
                                         halloffame=hof, verbose=True)
        pool.close()
        pop = _pop
        log = _log
    Main.statistics(pop, log, hof)
