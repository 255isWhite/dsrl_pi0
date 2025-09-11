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
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.15

JAX_TRACEBACK_FILTERING=off python3 examples/launch_train_sim.py \
--algorithm pixel_sac \
--env libero \
--wandb_project ${proj_name} \
--batch_size 256 \
--discount 0.999 \
--seed 0 \
--max_steps 500000  \
--eval_interval 10000 \
--log_interval 500 \
--eval_episodes 1 \
--multi_grad_step 20 \
--start_online_updates 200 \
--start_warmup_size 4 \
--num_online_gradsteps_batch 40000 \
--resize_image 64 \
--action_magnitude 1.0 \
--query_freq 20 \
--task_id 0 \
--task_suite libero_object \
--pi0_model /data0/zh1/.cache/openpi/pi0_libero40_10-30shot/pi0_libero40_10-30shot/20000 \
--pi0_config pi0_libero40_10-30shot \
--eval_at_begin 1 \
--kl_coeff 1.0 \
--qwarmup 1 \
--max_timesteps 300 \
--use_res 0 \
--label test_distill \
--action_magnitude 3.0 \