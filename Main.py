# -*- coding: <UTF-8> -*-
# !/usr/bin/env pypy
import sys

data = {}
FILES = []
args = 0


def read(index):
    global args, data
    with open(FILES[index], "r") as archive:
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


def main():
    for i in range(0, 5):
        FILES.append(sys.argv[i])


def data():
    global data
    return data


def args():
    global args
    return args


if __name__ == "__main__":
    main()
