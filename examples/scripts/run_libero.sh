#!/bin/bash
proj_name=CoupleNR_LIBERO
device_id=2

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl  
export MUJOCO_EGL_DEVICE_ID=$device_id

export OPENPI_DATA_HOME=./openpi
export EXP=./logs/$proj_name; 
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.14

log_dir="logs/ablation_any"
mkdir -p "$log_dir"
log_file="${log_dir}/libero.log"
# 如果文件已存在则清空
: > "$log_file"

JAX_TRACEBACK_FILTERING=off python3 examples/launch_train_sim.py \
--algorithm pixel_sac \
--env libero \
--seed 42 \
--label dsrl \
--wandb_project ${proj_name} \
--batch_size 256 \
--discount 0.999 \
--max_steps 500000 \
--eval_interval 10000 \
--log_interval 500 \
--eval_episodes 10 \
--multi_grad_step 20 \
--start_online_updates 500 \
--query_freq 20 \
--task_id 0 \
--task_suite libero_10 \
--pi0_model /data0/zh1/.cache/openpi/pi0_libero40_10-30shot/pi0_libero40_10-30shot/20000 \
--pi0_config pi0_libero40_10-30shot \
--eval_at_begin 1 \
--qwarmup 0 \
--kl_coeff 0.0 \
--res_coeff 0.1 \
--max_timesteps 500 \
--res_H 100000 \
>>"$log_file" 2>&1
