#!/bin/bash
proj_name=DSRL_test
device_id=0

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl  
export MUJOCO_EGL_DEVICE_ID=$device_id

export OPENPI_DATA_HOME=./openpi
export EXP=./logs/test/$proj_name
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.15

JAX_TRACEBACK_FILTERING=off python3 examples/launch_train_sim.py \
--algorithm pixel_sac_residual_2td \
--env libero \
--wandb_project ${proj_name} \
--batch_size 256 \
--discount 0.999 \
--seed 0 \
--max_steps 500000  \
--eval_interval 200 \
--log_interval 500 \
--eval_episodes 1 \
--multi_grad_step 20 \
--start_online_updates 2 \
--resize_image 64 \
--action_magnitude 1.0 \
--query_freq 20 \
--hidden_dims 128 \
--task_id 57 \
--task_suite libero_90 \
--pi0_model /data0/pi0/libero_130_1shot_2w  \
--pi0_config pi0_libero130_1shot \
--eval_at_begin 1 \
--kl_coeff 1.0 \
--qwarmup 1 \
--max_timesteps 40 \
--use_res 1 \