#!/bin/bash
pushd .
cd "$(dirname "$0")"

gsutil cp gs://liamog_udacity/bh_cloning/output/bh_clone.hdf5 output/model.h5
popd
