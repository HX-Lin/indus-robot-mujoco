#!/bin/bash
#
# 自动化流水线：数据集生成完成 → 离线提取 → 双蒸馏训练
#
# 用法:
#     bash run_pipeline.sh
#
# 会自动：
# 1. 等待数据集生成完成（检测 datasets_lerobot/meta/info.json）
# 2. 运行 GDINO 离线提取（热图 + Swin-T 特征）
# 3. 启动双蒸馏训练
#

set -e

cd /home/lin/workspace/indus-robot-mujoco/cs-platform
source ../cs_env/bin/activate

DATASET_DIR="datasets_lerobot"
HEATMAP_DIR="gdino_heatmaps"
OUTPUT_DIR="outputs/vita_v4_gdino_distill"

echo "============================================================"
echo "  自动化流水线: 数据集 → 离线提取 → 双蒸馏训练"
echo "============================================================"

# ========================== Step 1: 等待数据集生成完成 ==========================
echo ""
echo "[Step 1/3] 等待数据集生成完成..."

while true; do
    # 检查 summary.json 是否存在（generate_dataset.py 最后才写入）
    if [ -f "${DATASET_DIR}/summary.json" ]; then
        # 再检查 info.json 中的 total_episodes 是否 >= 200
        if [ -f "${DATASET_DIR}/meta/info.json" ]; then
            TOTAL_EPS=$(python -c "import json; print(json.load(open('${DATASET_DIR}/meta/info.json'))['total_episodes'])" 2>/dev/null || echo "0")
            if [ "$TOTAL_EPS" -ge 100 ]; then
                echo "  数据集生成完成! 共 ${TOTAL_EPS} 个 episodes"
                break
            fi
        fi
    fi

    # 显示当前进度
    if [ -d "${DATASET_DIR}/videos/observation.images.overhead/chunk-000" ]; then
        N_VIDEOS=$(ls ${DATASET_DIR}/videos/observation.images.overhead/chunk-000/*.mp4 2>/dev/null | wc -l)
        echo "  等待中... 当前已生成 ${N_VIDEOS} 个视频文件"
    else
        echo "  等待中... 数据集目录尚未创建"
    fi
    sleep 30
done

# 显示数据集信息
echo ""
echo "  数据集信息:"
python -c "
import json
info = json.load(open('${DATASET_DIR}/meta/info.json'))
print(f\"    Episodes: {info['total_episodes']}\")
print(f\"    Frames:   {info['total_frames']}\")
print(f\"    FPS:      {info['fps']}\")
"

# ========================== Step 2: 离线提取热图 + 语义特征 ==========================
echo ""
echo "============================================================"
echo "[Step 2/3] GDINO 离线提取 (热图 + Swin-T 语义特征)"
echo "============================================================"

# 清理旧的提取结果
if [ -d "${HEATMAP_DIR}" ]; then
    echo "  删除旧的提取结果: ${HEATMAP_DIR}"
    rm -rf "${HEATMAP_DIR}"
fi

python extract_gdino_heatmaps.py \
    --batch_size 8 \
    --num_workers 2 \
    --output_dir "${HEATMAP_DIR}" \
    --prompt "object." \
    --data_dir "${DATASET_DIR}" \
    --repo_id "local/mr2010-welding"

echo ""
echo "  提取完成! 文件列表:"
find "${HEATMAP_DIR}" -name "*.pt" -exec du -h {} \;

# ========================== Step 3: 双蒸馏训练 ==========================
echo ""
echo "============================================================"
echo "[Step 3/3] 启动双蒸馏训练 (heatmap + feature distillation)"
echo "============================================================"

python train.py \
    --policy vita \
    --use_gdino_distill \
    --heatmap_dir "${HEATMAP_DIR}" \
    --distill_weight 0.5 \
    --feat_distill_weight 0.5 \
    --steps 100000 \
    --batch_size 32 \
    --output_dir "${OUTPUT_DIR}" \
    --save_freq 5000 \
    --log_freq 100

echo ""
echo "============================================================"
echo "  流水线完成!"
echo "  模型保存在: ${OUTPUT_DIR}"
echo "============================================================"
