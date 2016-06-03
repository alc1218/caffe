#!/usr/bin/env sh

TOOLS=../build/tools

#OUTPUFLIE  _  cifashion_finetunning_MaxIters_solver

$TOOLS/caffe train \
  --solver=solver.prototxt 2>&1 | tee test_no_0_SGD.log

# reduce learning rate by factor of 10 after 8 epochs
#$TOOLS/caffe train \
#  --solver=wide_eyes_tech/quick_solver.prototxt
  #--solver=examples/cifar10/cifar10_quick_solver_lr1.prototxt \
  #--snapshot=wide_eyes_tech/quick_iter_4000.solverstate.h5
