# -*- coding: <UTF-8> -*-

import math
import joypy as joypy
import numpy as np
import operator
import matplotlib.pylab as plt
import random
import statsmodels as sm
import pandas as pd
import seaborn as sn
import csv

from deap import algorithms, base, tools, creator, gp

toolbox = base.Toolbox()
history = tools.History()

data = {}

FILE = []
TESTS = []
NAME = []
pop = []
famous = []
potencial = []




# --------------------------------- FILE AND DATASETS OPERATIONS ---------------------------------
def read_data(round, tround=False):
    global data, args, FILE, TEST, TESTS
    temp = []
    data = {}

    if round != -1 and not tround:
        str = FILE[round]
    elif round == -1 and not tround:
        str = TEST
    else:
        str = TESTS[round]

    with open(str, "r") as archive:
        lines = archive.readlines()
        for line in lines:
            values = line.split(",")
            for v in values[:-1]:
                v = float(v)
                temp.append(v)
            args = len(temp)
            data[float(values[-1])] = temp
            temp = []
    archive.close()


def clean():
    global FILE, NAME, DATASET, TEST
    FILE.clear()
    NAME.clear()
    DATASET = ""
    TEST = ""


def files(i, j, tst=False):
    global FILE, TESTS

    for tr in range(i, j):
        FILE.append("original/towerData-train-" + str(tr) + ".dat")
        NAME.append("Tests/towerData-train-" + str(tr) + ".dat")

        if tst:
            TESTS.append("original/towerData-test-" + str(tr) + ".dat")

    FILE = list(dict.fromkeys(FILE))
    TESTS = list(dict.fromkeys(TESTS))


def datasets():
    iteration = 0
    global TEST, JF, DATASET

    ms = routine()

    while iteration < 5:
        clean()

        TEST = "original/towerData-test-" + str(iteration) + ".dat"
        JF = "tower-test.json"
        DATASET = "towerData"

        if iteration == 0:
            files(1, 5)
        elif iteration == 1:
            files(0, 1)
            files(2, 5)
        elif iteration == 2:
            files(0, 2)
            files(3, 5)
        elif iteration == 3:
            files(4, 5)
            files(0, 3)
        else:
            files(0, 5)
        train(ms)
        test(ms)
        iteration += 1

    validation()


def out(population, log, hof, i):
    global NAME
    if "original" not in NAME:
        with open(NAME[i], "w") as archive:
            archive.write("Population: " + '\n')
            for p in population:
                archive.write(str(p) + '\n')
            archive.write('\n')
            archive.write(str(log) + '\n')
            archive.write('\n')
            for h in hof:
                archive.write("Best individual: " + str(h[0]) + '\n')
                archive.write("Fitness: " + str(h[0].fitness) + '\n')
    archive.close()


def write_json():
    global JF
    with open(JF, "w") as archive:
        archive.write("[" + '\n')
        for f in famous:
            archive.write("\t [" + str(f.expr) + "]")
        archive.write("]" + '\n')
    archive.close()


# --------------------------------- PG OPERATIONS ---------------------------------
# Safe Division
def analytic_quotient(c, b):
    return c / (math.sqrt(1 + math.pow(b, 2)))


# Transform the expression in our tree in a callable functionacorde
def eval_symb(individual):
    func = toolbox.compile(expr=individual)
    # Root Mean Square Error - Ela é a raiz do erro médio quadrático da diferença entre a predição e o valor real.
    global data
    errors = []
    for k, v in data.items():
        errors.append((func(*v) - k) ** 2)
    return math.sqrt(np.mean(errors)),


files(0, 1)
read_data(0)
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
    global data, pop

    pop = []

    toolbox.register("deme", tools.initRepeat, list, toolbox.individual)

    sizes = 200, 200, 200, 200, 200
    pop = [toolbox.deme(n=s) for s in sizes]

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)

    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    return mstats


def train(ms):
    global data, pop, famous

    for i in range(0, 4):
        data.clear()
        read_data(i)

        h = tools.HallOfFame(1)
        algorithms.eaSimple(pop[i], toolbox, 0.9, 0.1, 250, stats=ms, halloffame=h, verbose=True)
        famous.append(h)
        # print log
        out(pop, ms, famous, i)


def test(ms):
    global data, famous

    data.clear()
    read_data(-1)

    pop_specialist = []

    for i in range(len(famous)):
        pop_specialist.append(famous[i][0])

    h = tools.HallOfFame(1)
    algorithms.eaSimple(pop_specialist, toolbox, 0.9, 0.1, 250, stats=ms, halloffame=h, verbose=True)
    potencial.append(h)


def validation():
    global data, potencial
    files(0, 5, tst=True)

    individuals = {}
    pop_specialist = []

    for i in range(len(potencial)):
        pop_specialist.append(potencial[i][0])

    for specialist in pop_specialist:
        fits = []
        for i in range(0, 5):
            data.clear()
            read_data(i, tround=False)
            fit_train = eval_symb(specialist)

            data.clear()
            read_data(i, tround=True)
            fit_test = eval_symb(specialist)

            fits.append((fit_train, fit_test))
        individuals[str(specialist)] = fits
    statistics(individuals)
    plot_analytics()


def plot_analytics():
    csv_name = "statistics-" + str(DATASET) + ".csv"
    fig_name = "statistics-" + str(DATASET)

    df = pd.read_csv(csv_name)

    # Draw Plot - JoyPlot
    plt.figure(figsize=(16, 10), dpi=80)
    fig, axes = joypy.joyplot(df, column=['Fitness'], by="Classe", ylim='own',
                              grid='both', legend=False, xlabels=True, ylabels=True,
                              alpha=0.6, linewidth=.5, linecolor='w', fade=True)

    plt.title('Fitness dos Especialistas - ' + DATASET.upper(), fontsize=22)
    plt.rc("font", size=12)
    plt.xlabel('Fitness', fontsize=14, color='grey', alpha=1)
    plt.ylabel('Classe de Dados', fontsize=8, color='grey', alpha=1)
    plt.savefig(fig_name + "Joy", bbox_inches='tight')

    # Plot - MultiGrid
    plt.figure(figsize=(25, 20), dpi=80)
    dt = df[['Individuo', 'Fitness', 'Classe']]
    sn.catplot(x="Individuo", y="Fitness", hue="Classe", kind="swarm", data=dt)

    plt.ylabel('$Fitness$')
    plt.yticks(fontsize=10, alpha=.7)
    plt.title('Fitness dos Especialistas - ' + DATASET.upper(), fontsize=15)
    plt.grid(axis='both', alpha=.3)
    plt.savefig(fig_name + "MultiGrid", bbox_inches='tight')

    # Plot - Lines
    plt.figure(figsize=(25, 20), dpi=80)
    dt = df[['Individuo', 'Fitness', 'Classe']]
    g = sn.FacetGrid(dt, col="Classe", margin_titles=True, height=4)
    g.map(plt.scatter, "Fitness", "Classe", color="#338844", edgecolor="white", s=50, lw=1)
    for ax in g.axes.flat:
        ax.axline((0, 0), slope=.2, c=".2", ls="--", zorder=0)
    # g.set(xlim=(0, 60), ylim=(0, 14))

    plt.ylabel('$Fitness$')
    plt.yticks(fontsize=10, alpha=.7)
    plt.title('Fitness dos Especialistas - ' + DATASET.upper(), fontsize=15)
    plt.grid(axis='both', alpha=.3)
    plt.savefig(fig_name + "Lines", bbox_inches='tight')


def statistics(individuals):
    global FILE, TESTS, DATASET

    fields = ['Individuo', 'Fitness', 'Classe']
    f = []
    j = 0

    with open("statistics-" + str(DATASET) + ".csv", 'w', newline='') as archive:
        writer = csv.DictWriter(archive, fieldnames=fields)
        writer.writeheader()

        for k, v in individuals.items():
            for val in v:
                b, c = val
                b = float('.'.join(str(ele) for ele in b))
                c = float('.'.join(str(ele) for ele in c))
                f.append(b)
                f.append(c)

            writer.writerow({'Individuo': j, 'Fitness': f[0], 'Classe': 'Treino'})
            writer.writerow({'Individuo': j, 'Fitness': f[1], 'Classe': 'Teste'})
            writer.writerow({'Individuo': j, 'Fitness': f[2], 'Classe': 'Treino'})
            writer.writerow({'Individuo': j, 'Fitness': f[3], 'Classe': 'Teste'})
            writer.writerow({'Individuo': j, 'Fitness': f[4], 'Classe': 'Treino'})
            writer.writerow({'Individuo': j, 'Fitness': f[5], 'Classe': 'Teste'})
            writer.writerow({'Individuo': j, 'Fitness': f[6], 'Classe': 'Treino'})
            writer.writerow({'Individuo': j, 'Fitness': f[7], 'Classe': 'Teste'})
            writer.writerow({'Individuo': j, 'Fitness': f[8], 'Classe': 'Treino'})
            writer.writerow({'Individuo': j, 'Fitness': f[9], 'Classe': 'Teste'})
            f.clear()
            j += 1
    archive.close()


if __name__ == "__main__":
    datasets()
