#!/usr/bin/env bash

. venv/bin/activate

declare -i count
testes="original/airfoil-test-0.dat original/airfoil-test-1.dat original/airfoil-test-2.dat original/airfoil-test-3.dat original/airfoil-test-4.dat"
artestes=($testes)
treinos="original/airfoil-train-0.dat original/airfoil-train-1.dat original/airfoil-train-2.dat original/airfoil-train-3.dat original/airfoil-train-4.dat"
artreinos=($treinos)
output="Tests/statistics-airfoil.csv"
count=0

while [ $count -le 4 ]
do
  if [ $count == 0 ]
  then
    python3 Main.py "${artestes[0]}" "${artestes[1]}" "${artestes[2]}" "${artestes[3]}" "${artreinos[4]}" "$output"
    count=$(( count + 1 ))
  elif [ $count == 1 ]
  then
    python3 Main.py "${artestes[0]}" "${artestes[1]}" "${artestes[2]}" "${artreinos[3]}" "${artestes[4]}" "$output"
    count=$(( count + 1 ))
  elif [ $count == 2 ]
  then
    python3 Main.py "${artestes[0]}" "${artestes[1]}" "${artreinos[2]}" "${artestes[3]}" "${artestes[4]}" "$output"
    count=$(( count + 1 ))
  elif [ $count == 3 ]
  then
    python3 Main.py "${artestes[0]}" "${artreinos[1]}" "${artestes[2]}" "${artestes[3]}" "${artestes[4]}" "$output"
    count=$(( count + 1 ))
  elif [ $count == 4 ]
  then
    python3 Main.py "${artreinos[0]}" "${artestes[1]}" "${artestes[2]}" "${artestes[3]}" "${artestes[4]}" "$output"
    count=$(( count + 1 ))
  else
    echo "Nothing to be done here!"
    count=$(( count + 1 ))
  fi
done