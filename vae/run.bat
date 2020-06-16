@echo off

set anaconda=C:/ProgramData/Anaconda3/envs/tl/python.exe
set py=E:/git/pytorch/vae/main.py

set indir=hole/

call %anaconda% %py% --input %indir% --latent_dim 3 --lam 0 --beta 1 --batch-size 128

pause