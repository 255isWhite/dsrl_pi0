#!/bin/bash
proj_name=DSRL_test
device_id=3
client_gpu_id=4
host=localhost
port=11452
save_dir=/home/wangzh/dsrl_pi0/robotwin_log
task_name=lift_pot

# generate current time
current_time=$(date +%Y%m%d_%H%M%S)
# log file
log_file=./envx_log/robotwin_client_${task_name}_${current_time}.log

# make log directory
mkdir -p ./envx_log

source /home/wangzh/miniconda3/etc/profile.d/conda.sh
conda activate robotwin

python3 -u robotwin_client/client.py \
  --host $host \
  --port $port \
  --gpu_id $client_gpu_id \
  --task_name $task_name \
  --save_dir $save_dir \
  --seed 42 \
  > $log_file 2>&1 &
pid=$!

# 定义退出时清理函数
cleanup() {
    echo "Killing server process (PID=$pid)..."
    kill $pid 2>/dev/null
}
trap cleanup EXIT

echo "Start robotwin client at $host with GPU $client_gpu_id, Waiting for several seconds..."
sleep 1

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl  
export MUJOCO_EGL_DEVICE_ID=$device_id

export OPENPI_DATA_HOME=./openpi
export EXP=./logs/test/$proj_name
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false

conda activate dsrl_pi0
JAX_TRACEBACK_FILTERING=off python3 examples/launch_train_sim.py \
--algorithm pixel_sac_residual_2td \
--env robotwin \
--wandb_project ${proj_name} \
--batch_size 256 \
--discount 0.999 \
--seed 0 \
--max_steps 500000  \
--eval_interval 20 \
--log_interval 5 \
--eval_episodes 1 \
--multi_grad_step 20 \
--start_online_updates 2 \
--query_freq 20 \
--task_suite lift_pot \
--pi0_model /data/soft/wangzh/.cache/openpi/checkpoints/pi0_robotwin_clean/30000 \
--pi0_config pi0_robotwin_clean \
--eval_at_begin 1 \
--kl_coeff 1.0 \
--qwarmup 1 \
--max_timesteps 60 \
--save_dir /home/wangzh/dsrl_pi0/robotwin_log \
--client_addr $host:$port \