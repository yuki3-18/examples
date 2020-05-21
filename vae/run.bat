@echo off

set anaconda=C:/ProgramData/Anaconda3/envs/tl/python.exe
set py=E:/git/pytorch/vae/main.py

set indir=E:/git/TFRecord_example/input/CT/th_150/filename.txt

call %anaconda% %py% --input %indir% --mode 1 --latent_dim 24 --lam 0 --beta 0.1 --batch-size 128

pause