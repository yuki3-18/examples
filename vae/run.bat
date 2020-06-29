@echo off

set anaconda=C:/ProgramData/Anaconda3/envs/tl/python.exe
set py=E:/git/pytorch/vae/main.py

set indir=hole/

call %anaconda% %py% --input %indir% --latent_dim 6 --lam 10000 --beta 0.1 --topo --epochs 500

pause