#!/bin/bash

if [ -f OUTCAR.gz ]; then
    gzip -d OUTCAR.gz
fi

rm -f m_??/traj*.vasp 2>/dev/null

for mode in {01..03}    # select the modes
do
    mkdir -p m_${mode}
    cd m_${mode}
    (( m = mode - 1 ))
    ./phonon_traj.py -i ../OUTCAR -p ../POSCAR -m ${m} -t 300 -msd classical --linear_traj -nsw 15
    for ii in {01..15}
    do
        mkdir -p ${ii}
        cd ${ii}
        ln -sf ../../INCAR
        ln -sf ../../POTCAR
        ln -sf ../../KPOINTS
        ln -sf ../traj_${ii}.vasp POSCAR
        cd ../
    done
    cd ..
done

gzip OUTCAR
