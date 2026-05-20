#!/usr/bin/env bash

NAME="shakespeare"

cd ../leaf_utils

python3 stats.py --name $NAME

cd ../$NAME