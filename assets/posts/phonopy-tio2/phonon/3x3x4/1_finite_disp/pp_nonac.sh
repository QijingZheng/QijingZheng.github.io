if [ ! -f FORCE_SETS ]; then
    phonopy -f d_00?/vasprun.xml
fi

phonopy band.conf
phonopy dos.conf
python ph_band-dos.py
