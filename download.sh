#!/bin/bash

if [ ! -d "bmodel_files" ]; then
    wget https://github.com/ZillaRU/roop_face/releases/download/v0.1/bmodel_files.tar.gz
    tar -xzf bmodel_files.tar.gz
    rm -rf bmodel_files.tar.gz
fi