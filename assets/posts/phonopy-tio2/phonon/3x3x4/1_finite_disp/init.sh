#!/bin/bash

rm POSCAR-* 2>/dev/null
phonopy -d --dim="3 3 4" -c ../../optimized_unitcell.vasp

for pos in POSCAR-*
do
    mkdir -p d_${pos/POSCAR-}
    cd d_${pos/POSCAR-}
    ln -sf ../INCAR
    ln -sf ../POTCAR
    ln -sf ../KPOINTS
    cp ../${pos} POSCAR
    cd ..
done
