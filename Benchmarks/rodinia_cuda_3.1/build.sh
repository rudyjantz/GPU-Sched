#!/bin/bash

benchmarks=(
    b+tree                                                                      
    backprop                                                                    
    bfs                                                                         
    hotspot3D                                                                   
    particlefilter                                                              
    srad/srad_v1                                                                
    srad/srad_v2               
)

for b in ${benchmarks[@]}; do
    pushd cuda/${b}
    make
    popd
done
