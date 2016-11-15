#!/usr/bin/env bash

csvgrep -c 5 -r "[^NA]" data/train_ver2.csv > data/training_no_na.csv
csvgrep -c 2 -f data/id_samples.csv data/training_no_na.csv >> data/training_sample.csv
