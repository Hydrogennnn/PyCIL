#!/bin/bash
trap 'exit 0' EXIT    # 以错误码0退出，防止集群会重复执行错误脚本
set -e                # 脚本出错立刻退出

# -- 进入目录：你也可以手写路径，进入你想要的目录，运行对应的脚本 -- 
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)   # 提取该脚本所在目录的绝对路径
cd "$SCRIPT_DIR"                            # 进入该目录

# --- 日志设置：将终端输出同步到本地文件run.log中 ---

save_name="avcil_baseline_ks"
log_file="logs/${save_name}.log"
: > $log_file
exec > >(stdbuf -oL tee -a "$log_file") 2>&1 # 确保实时刷新


wandb login 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



# torchrun --nproc_per_node=2 main.py \
python main.py \
  --prefix reproduce \
  --dataset kinetics \
  --memory_size 340 \
  --memory_per_class 20 \
  --no-fixed_memory \
  --shuffle \
  --init_cls 6 \
  --increment 6 \
  --model_name avcil \
  --device 0 \
  --seed 42 \
  --project nips26 \
  --save_name $save_name \
  --init_epoch 1 \
  --epochs 1 \



# torchrun --nproc_per_node=2 main.py \
# python main.py \
#   --prefix reproduce \
#   --dataset ave \
#   --memory_size 340 \
#   --memory_per_class 20 \
#   --no-fixed_memory \
#   --shuffle \
#   --init_cls 7 \
#   --increment 7 \
#   --model_name avcil \
#   --device 0 \
#   --seed 42 \
#   --project nips26 \
#   --save_name $save_name \
