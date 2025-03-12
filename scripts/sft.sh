#!/bin/bash

# Read configuration file
config_file="./configs/sft_cfg.json"

# Default number of GPUs
default_num_gpus=1

# Variables for specifying GPUs and number of GPUs to use
specified_gpus=""
num_gpus=$default_num_gpus

# Check if the file exists  -f检查文件是否存在
if [ ! -f "$config_file" ]; then
    echo "Error: Configuration file $config_file does not exist"
    exit 1 
fi

# Parse JSON file and extract parameters  python -c允许在终端执行python代码 -m 执行模块 -i 执行代码后进入交互模型  -u 不使用缓冲，使用实时日志
use_deepspeed=$(python -c "import json; print(json.load(open('$config_file'))['EnvironmentArguments']['use_deepspeed'])")

# Check for command line arguments  $#表示传递参数的个数 -gt表示great then
while [ $# -gt 0 ]; do
    case $1 in
        -g|--gpus)
            specified_gpus="$2"
            shift 2
            ;;
        -n|--num_gpus)
            num_gpus="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Generate GPU list based on specified GPUs or num_gpus  
if [ -n "$specified_gpus" ]; then
    gpu_list=$specified_gpus
    num_gpus=$(echo $specified_gpus | tr ',' '\n' | wc -l)  # tr ',' '\n'将逗号替换为换行符，再使用wc -l统计行数
    echo "Using specified GPUs: $gpu_list"
else
    gpu_list=$(seq -s, 0 $((num_gpus-1))) # seq -s, 0 $((num_gpus-1))生成0到num_gpus-1的数字序列, -s 数字分隔符为逗号。
    echo "No GPUs specified, using first $num_gpus GPUs: $gpu_list"
fi

echo "Number of GPUs: $num_gpus"
echo "GPU list: $gpu_list"

# Set the running command based on parameters
if [ "$use_deepspeed" = "True" ]; then
    # deepspeed命令行 deepspeed main.py --num_nodes 节点数 --num_gpus 每个节点GPU数 --master_port 主节点进程端口 --include 指定特定GPU -exclude 排除的GPU --deepspeed deepspeed_config.json>
    run_command="deepspeed --include localhost:$gpu_list --master_port 15555 src/Main.py $config_file"
    echo "Running with DeepSpeed"
elif [ "$num_gpus" -gt 1 ]; then
    # torchrun命令行 torchrun --nproc_per_node 每个节点启动进程数，也就是每个节点GPU数量 --nnodes 节点总数 --node_rank= 指定当前节点号码 --master_addr=<ADDR> 主节点IP地址 --master_port=<PORT> 主节点端口 --rdzv_id=<ID> 此次训练ID --rdzv_backend=<BACKEND> 指定训练后端，有c10d, etcd, consul, rendezvous, static等。 --rdzv_endpoint=<ENDPOINT> rendezvous后端的连接地址 main.py --config_file config.json
    run_command="CUDA_VISIBLE_DEVICES=$gpu_list torchrun --nproc_per_node=$num_gpus --master_port=15200 src/Main.py $config_file"
    echo "Running in multi-GPU distributed mode"
else
    run_command="CUDA_VISIBLE_DEVICES=$gpu_list python src/Main.py $config_file"
    echo "Running in single GPU mode"
fi

# Print the command that will be executed
echo "-------------------------------------"    
echo "$run_command"
echo "-------------------------------------"
# Execute the command
eval $run_command