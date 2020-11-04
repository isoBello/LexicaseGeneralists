#!/usr/bin/env bash

. venv/bin/activate

testes="original/airfoil-test-0.dat original/airfoil-test-1.dat original/airfoil-test-2.dat original/airfoil-test-3.dat original/airfoil-test-4.dat"
treinos="original/airfoil-train-0.dat original/airfoil-train-1.dat original/airfoil-train-2.dat original/airfoil-train-3.dat original/airfoil-train-4.dat"
original="original/originals/airfoil-test.dat"
output="Tests/statistics0-airfoil.dat Tests/statistics1-airfoil.dat Tests/statistics2-airfoil.dat Tests/statistics3-airfoil.dat Tests/statistics4-airfoil.dat"
LOGTRAIN="Tests/LOG0-train-statistics.csv Tests/LOG1-train-statistics.csv Tests/LOG2-train-statistics.csv Tests/LOG3-train-statistics.csv Tests/LOG4-train-statistics.csv"
LOGTEST="Tests/LOG0-test-statistics.csv Tests/LOG1-test-statistics.csv Tests/LOG2-test-statistics.csv Tests/LOG3-test-statistics.csv Tests/LOG4-test-statistics.csv"
artestes=($testes)
artreinos=($treinos)
arout=($output)
arltrains=($LOGTRAIN)
arltests=($LOGTEST)

python3 Main.py "$original" "${artestes[*]}" "${artreinos[*]}" "${arout[*]}" "${arltrains[*]}" "${arltests[*]}"
