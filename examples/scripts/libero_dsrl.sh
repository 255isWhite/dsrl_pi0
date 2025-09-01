#!/bin/bash
set -euo pipefail
unset CUDA_VISIBLE_DEVICES   

# ===== 可配置 =====
proj_name="DSRL_pi0_Libero"
# gpu_list=(2 3)                          # 物理 GPU ID

# ablations=(
#   "qwarmup=1,seed=42"
#   "qwarmup=1,seed=43"
# )

gpu_list=(0 1 2 3 4 5 6 7)                          # 物理 GPU ID
ablations=(
  "label=dsrl,task_id=7,task_suite=libero_10"
  "label=dsrl,task_id=9,task_suite=libero_10"
  "label=dsrl,task_id=2,task_suite=libero_spatial"
  "label=dsrl,task_id=4,task_suite=libero_spatial"
  "label=dsrl,task_id=6,task_suite=libero_object"
  "label=dsrl,task_id=8,task_suite=libero_object"
  "label=dsrl,task_id=4,task_suite=libero_goal"
  "label=dsrl,task_id=8,task_suite=libero_goal"
)


per_proc_cap_gb=11
max_concurrency_per_gpu=6
safety_gb=1
sleep_between_launch=1
check_interval=1

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

  # 第一次温柔终止
  for id in "${pids[@]}"; do
    pid="${id%%:*}"; pgid="${id##*:}"
    if kill -0 "$pid" 2>/dev/null; then
      echo "  → TERM 进程组 -$pgid"
      kill -TERM "-$pgid" 2>/dev/null || true
    fi
  done

  # 给点时间做清理
  sleep 2

  # 兜底强杀
  for id in "${pids[@]}"; do
    pid="${id%%:*}"; pgid="${id##*:}"
    if kill -0 "$pid" 2>/dev/null; then
      echo "  → KILL 进程组 -$pgid"
      kill -KILL "-$pgid" 2>/dev/null || true
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

# ✅ 新：把 "k=v,k2=v2" 转成安全的命令行片段：--k v --k2 v2
dict_to_args() {
  local kvs="$1"
  local args=()
  IFS=',' read -ra pairs <<< "$kvs"
  for pair in "${pairs[@]}"; do
    IFS='=' read -r k v <<< "$pair"
    k="$(echo "$k" | xargs)"
    v="$(echo "$v" | xargs)"
    if [[ -z "${k}" ]]; then continue; fi
    if [[ -z "${v}" ]]; then
      # 空值时按 flag 处理：--flag
      args+=("--${k}")
    else
      args+=("--${k}" "${v}")
    fi
  done
  # 逐个做 Shell 安全转义并打印为一行
  printf '%q ' "${args[@]}"
}

# ✅ 极稳的 tag 生成：逗号->下划线，等号->短横，剥掉不安全字符
kv_to_tag() {
  local kvs="$1"
  python3 - "$kvs" <<'PY'
import re, sys
s = sys.argv[1].strip()
# 规范化：去空白、逗号->_、等号->-
s = s.replace(',', '_').replace('=', '-')
# 仅保留: 字母数字、下划线、点、冒号、短横
s = re.sub(r'[^A-Za-z0-9_.:-]', '-', s)
print(s)
PY
}

run_ts=$(date "+%Y%m%d_%H%M%S")
start_task_on_slot() {
  local gpu_id=$1
  local slot=$2
  local kvs="$3"
  local mem_fraction=$4

  local lock_file; lock_file=$(slot_lock_path "$gpu_id" "$slot")
  local norm_kvs="$kvs"
  local ablation_args; ablation_args="$(dict_to_args "$norm_kvs")"
  local tag; tag="$(kv_to_tag "$norm_kvs")"

  local log_dir="logs/ablation_any/${run_ts}"
  mkdir -p "$log_dir"
  local log_file="${log_dir}/${tag}.log"
  : > "$log_file"

  local t; t=$(date "+%Y-%m-%d %H:%M:%S")
  echo "[$t] 启动 GPU $gpu_id / slot $slot → ablation={$kvs}, mem_fraction=$mem_fraction" | tee -a "$log_file"
  print_gpu_status | tee -a "$log_file"

  (
    exec 200>"$lock_file"
    flock -n 200 || exit 1

    export CUDA_VISIBLE_DEVICES=$gpu_id
    export MUJOCO_EGL_DEVICE_ID=$gpu_id
    export XLA_PYTHON_CLIENT_PREALLOCATE=true
    export XLA_PYTHON_CLIENT_MEM_FRACTION=$mem_fraction

    exec python3 examples/launch_train_sim.py \
      --algorithm pixel_sac \
      --env libero \
      --seed 42 \
      --prefix "${tag}_G${gpu_id}" \
      --wandb_project ${proj_name} \
      --batch_size 256 \
      --max_steps 500000 \
      --eval_interval 10000 \
      --log_interval 500 \
      --eval_episodes 10 \
      --multi_grad_step 20 \
      --start_online_updates 500 \
      --query_freq 20 \
      --task_id 21 \
      --task_suite libero_90 \
      --pi0_model /data0/pi0/libero_130_1shot_2w \
      --pi0_config pi0_libero130_1shot \
      --eval_at_begin 1 \
      --qwarmup 0 \
      --kl_coeff 0.0 \
      --res_coeff 0.1 \
      --max_timesteps 400 \
      $(echo $ablation_args) \
      >>"$log_file" 2>&1
    status=$?
    if (( status != 0 )); then
      echo "❌ [$(date '+%Y-%m-%d %H:%M:%S')] 任务崩溃: GPU=$gpu_id, ablation={$kvs}, exit_code=$status" | tee -a "$log_file"
    fi
  ) &
  pid=$!
  pgid="$(ps -o pgid= "$pid" | tr -d ' ')"
  pids+=("$pid:$pgid")
}


# ===== 主调度 =====
for kvs in "${ablations[@]}"; do
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
          start_task_on_slot "$gpu_id" "$slot" "$kvs" "$mem_fraction"   # ✅ 传入本次 ablation
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
