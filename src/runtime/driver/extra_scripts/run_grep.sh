#!/bin/bash

./grep_nvprof.py ../results/p100_medium_16jobs_4.single-assignment.2.workloader-log &> medium-sa.txt
./grep_nvprof.py ../results/p100_medium_16jobs_4.cg.4.workloader-log &> medium-cg.txt
./grep_nvprof.py ../results/p100_medium_16jobs_4.mgb.16.workloader-log &> medium-mgb.txt


./grep_nvprof.py ../results/p100_large_16jobs_4.single-assignment.2.workloader-log &> large-sa.txt
./grep_nvprof.py ../results/p100_large_16jobs_4.cg.3.workloader-log &> large-cg.txt
./grep_nvprof.py ../results/p100_large_16jobs_4.mgb.8.workloader-log &> large-mgb.txt

