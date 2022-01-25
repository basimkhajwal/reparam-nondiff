#!/bin/bash

lr=${1:-0.001}
sample_n_grad=${2:-1}
sample_n_var=${3:-8}
iters=${4:-100}
plot_n=${5:-10}
plot_step=${6:-5}

python2 main_run.py res/ bm-sns.py   $iters $lr $sample_n_grad $sample_n_var $plot_n $plot_step run,plot
python2 main_run.py res/ bm-tcl.py   $iters $lr $sample_n_grad $sample_n_var $plot_n $plot_step run,plot
python2 main_run.py res/ bm-time.py  $iters $lr $sample_n_grad $sample_n_var $plot_n $plot_step run,plot
