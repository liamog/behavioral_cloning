#!/bin/bash
pushd .
cd "$(dirname "$0")"

gsutil rsync -d ../data/proccessed_and_pickled/ gs://liamog_udacity/bh_cloning/
popd
