#!/usr/bin/env bash

version=$2
description=$3

if [ "$1" = "lgbm" ]; then
  python lgbm.py -v 16 -d ''
elif [ "$1" = "catboost" ]; then
  python catbst.py -v 10 -d ''
else
  echo "Sorry, $YES_OR_NO not recognized. Enter yes or no."
  exit 1
fi

sudo poweroff