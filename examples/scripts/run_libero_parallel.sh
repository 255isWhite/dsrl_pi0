#!/bin/bash
set -euo pipefail

# ===== 可配置 =====
proj_name="DSRL_pi0_Libero"
gpu_list=(2 3)                          # 物理 GPU ID
kl_list=(1.0 0.0)

per_proc_cap_gb=12
max_concurrency_per_gpu=2
safety_gb=1
sleep_between_launch=5
check_interval=10

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OPENPI_DATA_HOME=./openpi
export EXP=./logs/$proj_name;

# ====== 新增：进程 PID 记录与 Ctrl+C 杀停 ======
pids=()
cleanup() {
    echo ""
    echo "⚠️ 捕获到中断信号，正在杀掉所有子进程..."
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "  → kill $pid"
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    wait
    echo "✅ 已全部杀掉"
    exit 1
}
trap cleanup SIGINT SIGTERM

# ===== 工具函数 =====
mb() { python3 - <<PY
print(int(round($1*1024)))
PY
}

get_gpu_mem_mb() {
  nvidia-smi --query-gpu=memory.total,memory.used,memory.free \
    --format=csv,noheader,nounits -i "$1" | awk -F',' '{print $1" "$2" "$3}'
}

max_slots_raw() {
  local gpu_id=$1
  read -r total used free <<<"$(get_gpu_mem_mb "$gpu_id")"
  local cap_mb=$(mb $per_proc_cap_gb)
  echo $(( total / cap_mb ))
}

max_slots() {
  local gpu_id=$1
  local raw; raw=$(max_slots_raw "$gpu_id")
  (( raw < 1 )) && raw=1
  (( raw > max_concurrency_per_gpu )) && raw=$max_concurrency_per_gpu
  echo "$raw"
}

calc_fraction_for_gpu() {
  local gpu_id=$1
  read -r total used free <<<"$(get_gpu_mem_mb "$gpu_id")"
  local cap_mb=$(mb $per_proc_cap_gb)
  python3 - <<PY
total=$total; cap=$cap_mb
frac = min(0.95, max(0.05, cap/total))
print(f"{frac:.4f}")
PY
}

slot_lock_path() { echo "/tmp/gpu_${1}_slot${2}.lock"; }

slot_is_free() {
  local lock_file; lock_file=$(slot_lock_path "$1" "$2")
  if flock -n "$lock_file" true; then
    rm -f "$lock_file"
    return 0
  else
    return 1
  fi
}

occupied_slots() {
  local gpu_id=$1
  local max; max=$(max_slots "$gpu_id")
  local occ=0
  for slot in $(seq 0 $((max-1))); do
    local lock_file; lock_file=$(slot_lock_path "$gpu_id" "$slot")
    if ! flock -n "$lock_file" true; then
      ((occ+=1))
    else
      rm -f "$lock_file" 2>/dev/null || true
    fi
  done
  echo "$occ"
}

print_gpu_status() {
  echo "------ GPU 显存状态 ------"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
             --format=csv,noheader,nounits \
      | awk -F',' '{printf "GPU %d (%s): Total=%dMB, Used=%dMB, Free=%dMB\n", $1,$2,$3,$4,$5}'
  echo "-------------------------"
}

# 持锁启动到指定槽
start_task_on_slot() {
  local gpu_id=$1
  local slot=$2
  local kl=$3
  local mem_fraction=$4
  local lock_file; lock_file=$(slot_lock_path "$gpu_id" "$slot")
  local log_file="logs/kl_coeff_ablation/gpu${gpu_id}_slot${slot}.log"

  local t; t=$(date "+%Y-%m-%d %H:%M:%S")
  echo "[$t] 启动 GPU $gpu_id / slot $slot → kl=$kl, mem_fraction=$mem_fraction" | tee -a "$log_file"
  print_gpu_status | tee -a "$log_file"

  # 子进程生命周期内持有锁
  flock -n "$lock_file" bash -lc "
    CUDA_VISIBLE_DEVICES=$gpu_id \
    MUJOCO_EGL_DEVICE_ID=$gpu_id \
    XLA_PYTHON_CLIENT_PREALLOCATE=true \
    XLA_PYTHON_CLIENT_MEM_FRACTION=$mem_fraction \
    python3 examples/launch_train_sim.py \
      --algorithm pixel_sac \
      --env libero \
      --prefix Qwarmup_${kl}coeff_klMu \
      --wandb_project ${proj_name} \
      --batch_size 256 \
      --discount 0.999 \
      --seed 0 \
      --max_steps 500000 \
      --eval_interval 10000 \
      --log_interval 500 \
      --eval_episodes 20 \
      --multi_grad_step 20 \
      --start_online_updates 500 \
      --resize_image 64 \
      --action_magnitude 1.0 \
      --query_freq 20 \
      --hidden_dims 128 \
      --task_id 57 \
      --task_suite libero_90 \
      --pi0_model pi0_libero \
      --pi0_config pi0_libero \
      --eval_at_begin 1 \
      --kl_coeff $kl \
    > \"$log_file\" 2>&1
  " &
  pid=$!
  pids+=("$pid")  # 记录 PID
}

mkdir -p logs/kl_coeff_ablation

# ===== 主调度 =====
for kl in "${kl_list[@]}"; do
  while true; do
    min_occ=999
    declare -A occ_map
    for gpu_id in "${gpu_list[@]}"; do
      occ=$(occupied_slots "$gpu_id")
      occ_map[$gpu_id]=$occ
      (( occ < min_occ )) && min_occ=$occ
    done

    launched=0
    for gpu_id in "${gpu_list[@]}"; do
      occ=${occ_map[$gpu_id]}
      (( occ != min_occ )) && continue
      max=$(max_slots "$gpu_id")
      (( occ >= max )) && continue

      read -r total used free <<<"$(get_gpu_mem_mb "$gpu_id")"
      cap_mb=$(mb $per_proc_cap_gb)
      safety_mb=$(mb $safety_gb)
      (( free < cap_mb + safety_mb )) && continue

      for slot in $(seq 0 $((max-1))); do
        if slot_is_free "$gpu_id" "$slot"; then
          mem_fraction=$(calc_fraction_for_gpu "$gpu_id")
          start_task_on_slot "$gpu_id" "$slot" "$kl" "$mem_fraction"
          sleep "$sleep_between_launch"
          launched=1
          break
        fi
      done

      (( launched == 1 )) && break
    done

    (( launched == 1 )) && break
    sleep "$check_interval"
  done
done

wait
echo "所有任务已完成 ✅"
