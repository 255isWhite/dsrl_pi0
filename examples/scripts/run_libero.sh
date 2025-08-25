#!/bin/bash
proj_name=DSRL_pi0_Libero
device_id=3

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl  
export MUJOCO_EGL_DEVICE_ID=$device_id

export OPENPI_DATA_HOME=./openpi
export EXP=./logs/$proj_name; 
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.16

log_dir="logs/ablation_any"
mkdir -p "$log_dir"
log_file="${log_dir}/libero.log"
# 如果文件已存在则清空
: > "$log_file"

JAX_TRACEBACK_FILTERING=off python3 examples/launch_train_sim.py \
--algorithm pixel_sac_residual_2td \
--env libero \
--seed 42 \
--label gas_ball \
--wandb_project ${proj_name} \
--batch_size 256 \
--discount 0.999 \
--max_steps 500000 \
--eval_interval 10000 \
--log_interval 500 \
--eval_episodes 10 \
--multi_grad_step 20 \
--start_online_updates 500 \
--resize_image 64 \
--action_magnitude 1.0 \
--query_freq 20 \
--hidden_dims 128 \
--task_id 9 \
--task_suite libero_90 \
--pi0_model /mnt/ssd1/data/zh1/pi0/checkpoints/pi0_libero130_1shot/libero130_1shot/20000 \
--pi0_config pi0_libero130_1shot \
--eval_at_begin 1 \
--qwarmup 1 \
--kl_coeff 1.0 \
--res_coeff 0.1 \
--max_timesteps 400 \
--res_H 20000 \
--decay_kl 1 \
>>"$log_file" 2>&1
