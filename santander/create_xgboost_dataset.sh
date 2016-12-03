#!/usr/bin/env bash

csvgrep -c 1 -r "2015-05|2015-06|2016-05"  data/training_no_na.csv > data/training_xgboost.csv
