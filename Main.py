# -*- coding: <UTF-8> -*-
# !/usr/bin/env pypy

import sys
import PG
import Predictions


def main():
    # The first time you run, you need to create the data folders. So, we call PG.datasets
    # After that, we can do our stuff, so we can comment the function and run again with the other two PG functions
    # After finished the individuals evolution, we can start the predictions
    # The first predictions function evaluates individuals for each parquet file, for each line.
    # Is important to the kmeans algorithm.
    # The second predictions function find the archives to clustering - this is just made to save time
    # PG.datasets()

    # PG.routine()
    # PG.main()

    # Starting post precessing
    # Predictions.main()
    # Predictions.archives()
    Predictions.predictions()


def statistics(pop, log, hof):
    # Write statistics
    with open(sys.argv[1], "w") as archive:
        archive.write('\n')
        archive.write(str(log) + '\n')
        archive.write('\n')
        archive.write("Hall of Fame (Best Individuals): " + '\n')
        for h in hof:
            archive.write("Best individual: " + str(h) + '\n')
            archive.write("Fitness: " + str(h.fitness).replace(",", "") + '\n')

    # Write population
    with open(sys.argv[2], "w") as archive:
        archive.write("Population: " + '\n')
        for p in pop:
            archive.write(str(p) + '\n')
    archive.close()


if __name__ == "__main__":
    main()
