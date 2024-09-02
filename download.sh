#!/bin/bash

if [ ! -d "bmodel_files" ]; then
    if [ ! -f "bmodel_files.tar.gz" ]; then
        wget https://github.com/ZillaRU/roop_face/releases/download/v0.1/bmodel_files.tar.gz
    fi
    tar xzf bmodel_files.tar.gz
    rm -rf bmodel_files.tar.gz
fi

if [ ! -d "onnx_weights/buffalo_l" ]; then
    mkdir onnx_weights
    cd onnx_weights
    wget https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
    unzip buffalo_l.zip
    rm buffalo_l.zip
    cd -
fi