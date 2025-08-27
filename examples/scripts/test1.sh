#!/bin/bash
proj_name=DSRL_test
device_id=1

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl  
export MUJOCO_EGL_DEVICE_ID=$device_id

export OPENPI_DATA_HOME=./openpi
export EXP=./logs/test/$proj_name
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# export XLA_PYTHON_CLIENT_MEM_FRACTION=0.15
export XLA_FLAGS="--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_as_text"

JAX_TRACEBACK_FILTERING=off python3 examples/launch_train_sim.py \
--algorithm pixel_sac \
--env libero \
--wandb_project ${proj_name} \
--batch_size 8 \
--discount 0.999 \
--seed 0 \
--max_steps 500000  \
--eval_interval 10000 \
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
--pi0_model /data/.zh1/pi0/libero130_1shot/20000/20000 \
--pi0_config pi0_libero130_1shot \
--eval_at_begin 0 \
--kl_coeff 1.0 \
--qwarmup 1 \
--max_timesteps 200 \
--use_res 1 \
--denoise_steps 2 \