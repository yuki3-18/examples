@echo off

set anaconda=C:/ProgramData/Anaconda3/envs/toplayer/python.exe
set py=E:/git/pytorch/vae/predict_gen.py

set input="E:/git/pytorch/vae/input/s100/filename.txt"
set path="E:/git/pytorch/vae/results/artificial/z_3/B_0.1/"
set model=%path%model.pkl
set outdir=%path%gen/

call %anaconda% %py% --input %input% --model %model% --outdir %outdir% --mode 0

pause