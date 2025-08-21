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
export XLA_PYTHON_CLIENT_PREALLOCATE=false

JAX_TRACEBACK_FILTERING=off python3 examples/launch_train_sim.py \
--algorithm pixel_sac_residual_2td \
--env robomimic \
--wandb_project ${proj_name} \
--batch_size 256 \
--discount 0.999 \
--seed 0 \
--max_steps 500000  \
--eval_interval 100 \
--log_interval 50 \
--eval_episodes 10 \
--multi_grad_step 20 \
--start_online_updates 50 \
--resize_image 64 \
--action_magnitude 1.0 \
--query_freq 20 \
--hidden_dims 128 \
--task_id 57 \
--task_suite lift \
--pi0_model /data/.zh1/pi0/robomimic_5shot/29999 \
--pi0_config pi0_robomimic \
--eval_at_begin 1 \
--kl_coeff 1.0 \
--qwarmup 1 \
--max_timesteps 200 \
--use_res 1 \
--dataset_root /data/.zh1/robomimic/env_hdf5 \
# --label test \
# --prefix test \