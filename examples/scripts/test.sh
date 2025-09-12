#!/bin/bash
proj_name=DSRL_test
device_id=2

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl  
export MUJOCO_EGL_DEVICE_ID=$device_id

export OPENPI_DATA_HOME=./openpi
export EXP=./logs/test/$proj_name
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.2

JAX_TRACEBACK_FILTERING=off python3 examples/launch_train_sim.py \
--algorithm pixel_sac \
--env libero \
--wandb_project ${proj_name} \
--batch_size 256 \
--discount 0.999 \
--seed 0 \
--max_steps 500000  \
--eval_interval 10000 \
--log_interval 50 \
--media_log_fold 100 \
--eval_episodes 1 \
--multi_grad_step 20 \
--start_online_updates 200 \
--resize_image 64 \
--query_freq 20 \
--task_id 0 \
--task_suite libero_goal \
--pi0_model /data/soft/wangzh/.cache/openpi/checkpoints/pi0_libero40_10-30shot/20000 \
--pi0_config pi0_libero40_10-30shot \
--eval_at_begin 1 \
--kl_coeff 0.0 \
--qwarmup 1 \
--max_timesteps 200 \
--use_res 0 \
--label 48notanh \
--action_magnitude 1.0 \
--bc_coeff 1.0 \