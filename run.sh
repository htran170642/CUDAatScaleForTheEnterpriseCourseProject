#!/usr/bin/env bash
make clean build

./bin/nppFilters ./data/lady.pgm canny ./output/lady_canny.bmp >> output/output.log
./bin/nppFilters ./data/lady.pgm sobel ./output/lady_sobel.bmp >> output/output.log