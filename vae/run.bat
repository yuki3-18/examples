@echo off

set anaconda=C:/ProgramData/Anaconda3/envs/tl/python.exe
set py=E:/git/pytorch/vae/main.py

set indir=E:/git/pytorch/vae/input/s100/filename.txt

call %anaconda% %py% --input %indir% --mode 2 --latent_dim 24 --topo True --ramda 1 --beta 0

pause