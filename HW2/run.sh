#!/bin/bash

START_INDEX=1
END_INDEX=15

echo "--- Data Science Homework Executor ---"

echo "--------------------------------------------------"

for i in $(seq $START_INDEX $END_INDEX); do
    FILE_NAME="Problem_${i}.py"
    echo "Executing $FILE_NAME"
    python3 "$FILE_NAME"
done

