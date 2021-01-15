# -*- coding: <UTF-8> -*-
# !/usr/bin/env pypy
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from datetime import datetime

import PG

finals = []


def main():
    folders = []
    for folder in os.listdir(sys.argv[3]):
        folders.append(sys.argv[3] + folder + "/")

    df = pd.DataFrame(columns=['Arquivo', 'Individual', 'Erro'])

    with open(sys.argv[2], 'r') as archive:
        for individuo in archive.readlines():
            try:
                func = PG.toolbox.compile(expr=individuo)
                for fold in folders:
                    try:
                        for file in os.listdir(fold):
                            try:
                                filepath = fold + "/" + file
                                data = pd.read_parquet(filepath, engine="pyarrow")

                                df['Arquivo'] = file
                                df['Individual'] = individuo

                                try:
                                    df['Erro'] = data.apply(lambda linha: func(*linha[:-1]) - linha[-1], axis=1)

                                except TypeError:
                                    pass

                                times = datetime.now().strftime('%H:%M:%S')
                                df.to_parquet("Results/" + times + "-" + file)

                            except FileNotFoundError:
                                print(filepath + " not found")
                                pass
                    except NotADirectoryError:
                        pass
            except (SyntaxError, TypeError) as e:
                pass
    archive.close()


def predictions():
    errors = {}
    for file in os.listdir("Results/"):
        filepath = "Results/" + file
        data = pd.read_parquet(filepath, engine="pyarrow")
        individuo = data.iloc[0]['Individual']
        errors[individuo] = data['Erro'].copy()

    results = pd.DataFrame(errors)
    results = results.fillna(999999)
    kmeans = KMeans(n_clusters=5, random_state=0).fit(results)

    # Get the cluster centroids
    print(kmeans.cluster_centers_)

    df = pd.DataFrame(kmeans.cluster_centers_)
    df.to_csv(sys.argv[6])

    # Get the cluster labels
    print(kmeans.labels_)

    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x')

    plt.title('Data points and cluster centroids')

    plt.savefig(sys.argv[5], bbox_inches='tight')
