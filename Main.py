# -*- coding: <UTF-8> -*-
# !/usr/bin/env pypy
import sys
import PG

args = 0
data = {}


def read(index):
    global args, data
    temp = []
    try:
        with open(sys.argv[index], "r") as archive:
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


def statistics(pop, log, hof, df):
    # Do stuff

    df.to_csv(sys.argv[7], index=True)  # Write the csv file

    # Write LOG
    with open(sys.argv[8], "w") as archive:
        archive.write("Population: " + '\n')
        for p in pop:
            archive.write(str(p) + '\n')
        archive.write('\n')
        archive.write(str(log) + '\n')
        archive.write('\n')
        for h in hof:
            archive.write("Best individual: " + str(h) + '\n')
            archive.write("Fitness: " + str(h.fitness).replace(",", "") + '\n')
    archive.close()


if __name__ == "__main__":
    main()
