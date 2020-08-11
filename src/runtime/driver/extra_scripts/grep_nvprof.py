#!/usr/bin/env python3
import sys

with open(sys.argv[1]) as f:
    inside = False
    for line in f:
        if "NVPROF" in line:
            inside = True
        if inside:
            print(line.strip())
        if '' == line.strip():
            inside = False
