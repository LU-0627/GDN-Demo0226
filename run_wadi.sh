#!/bin/bash

# 显示帮助信息
show_help() {
    echo "Usage: ./run_wadi.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -d, --device DEVICE    Set device type (cpu or gpu, default: cpu)"
    echo "  -g, --gpu-id ID        Set GPU ID (default: 0)"
    echo "  -e, --epoch EPOCH      Set number of epochs (default: 30)"
    echo "  -m, --moe-num NUM      Set MOE number (default: 8)"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Example:"
    echo "  ./run_wadi.sh --device gpu --gpu-id 0"
    echo "  ./run_wadi.sh -d cpu"
}

# 默认参数
DEVICE="cpu"
GPU_ID=0
EPOCH=30
MOE_NUM=8

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--device)
            DEVICE="$2"
            shift 2
            ;;
        -g|--gpu-id)
            GPU_ID="$2"
            shift 2
            ;;
        -e|--epoch)
            EPOCH="$2"
            shift 2
            ;;
        -m|--moe-num)
            MOE_NUM="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            show_help
            exit 1
            ;;
    esac

done

# 验证设备类型
if [[ "$DEVICE" != "cpu" && "$DEVICE" != "gpu" ]]; then
    echo "Error: Device must be either 'cpu' or 'gpu'"
    show_help
    exit 1
fi

# 数据集配置
dataset="wadi"
seed=5
batchSize=32
slideWin=5
dim=64
outLayerNum=1
slideStride=1
topk=5
outLayerInterDim=128
valRatio=0.2
decay=0
pathPattern="$dataset"
comment="wadi-default"
report="best"

# 运行命令
if [[ "$DEVICE" == "cpu" ]]; then
    python main.py \
        -dataset "$dataset" \
        -save_path_pattern "$pathPattern" \
        -slide_stride "$slideStride" \
        -slide_win "$slideWin" \
        -batch "$batchSize" \
        -epoch "$Epoch" \
        -comment "$comment" \
        -random_seed "$seed" \
        -decay "$decay" \
        -dim "$dim" \
        -out_layer_num "$outLayerNum" \
        -out_layer_inter_dim "$outLayerInterDim" \
        -val_ratio "$valRatio" \
        -report "$report" \
        -topk "$topk" \
        -moe_num "$MOE_NUM" \
        -device cpu
else
    CUDA_VISIBLE_DEVICES="$GPU_ID" python main.py \
        -dataset "$dataset" \
        -save_path_pattern "$pathPattern" \
        -slide_stride "$slideStride" \
        -slide_win "$slideWin" \
        -batch "$batchSize" \
        -epoch "$Epoch" \
        -comment "$comment" \
        -random_seed "$seed" \
        -decay "$decay" \
        -dim "$dim" \
        -out_layer_num "$outLayerNum" \
        -out_layer_inter_dim "$outLayerInterDim" \
        -val_ratio "$valRatio" \
        -report "$report" \
        -topk "$topk" \
        -moe_num "$MOE_NUM"
fi
