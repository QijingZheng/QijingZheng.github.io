if [ ! -f FORCE_SETS ]; then
    phonopy -f d_00?/vasprun.xml
fi

phonopy --nac band.conf
phonopy --nac dos.conf
python ph_band-dos.py
