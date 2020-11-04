# -*- coding: <UTF-8> -*-
# !/usr/bin/env pypy
import sys
import PG

args = 0
data = {}


def read(file):
    global args, data
    temp = []
    try:
        with open(file, "r") as archive:
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
    except ValueError:
        print("Nothing to be done here! Moving on...")


def main():
    PG.datasets()


def get_data():
    global data
    return data


def get_args():
    global args
    return args


def statistics(pop, log, hof, df_train, df_test, ofile, ltrain, ltest):

    # Write LOG for trains and tests
    df_train.to_csv(ltrain, index=True)  # Write the csv file for train
    df_test.to_csv(ltest, index=True)  # Write the csv file for tests

    # Write statistics
    with open(ofile, "w") as archive:
        archive.write("Population: " + '\n')
        for p in pop:
            archive.write(str(p) + '\n')
        archive.write('\n')
        archive.write(str(log) + '\n')
        archive.write('\n')
        archive.write("Hall of Fame (Best Individuals): " + '\n')
        for h in hof:
            archive.write("Best individual: " + str(h) + '\n')
            archive.write("Fitness: " + str(h.fitness).replace(",", "") + '\n')
    archive.close()


if __name__ == "__main__":
    main()
