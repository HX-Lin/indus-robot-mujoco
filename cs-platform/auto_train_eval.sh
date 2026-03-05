#!/bin/bash
# 自动化脚本: 等待 Exp19/Exp20 训练完成后自动评估
# 用法: bash auto_train_eval.sh

PYTHON="/home/lin/workspace/indus-robot-mujoco/cs_env/bin/python"
CD_DIR="/home/lin/workspace/indus-robot-mujoco/cs-platform"
cd "$CD_DIR"

echo "============================================================"
echo "  Auto Train + Eval Pipeline"
echo "  Exp19: joint abs | Exp20: EE delta"
echo "============================================================"

# 等待训练进程结束
echo ""
echo "[1/4] Waiting for training processes to finish..."
while true; do
    count=$(pgrep -f "train.py.*exp19\|train.py.*exp20" | wc -l)
    if [ "$count" -eq 0 ]; then
        echo "  All training processes completed!"
        break
    fi
    # 显示进度
    for exp in exp19_joint_abs exp20_ee_delta; do
        last_dir=$(ls -d outputs/$exp/checkpoint_* 2>/dev/null | sort | tail -1)
        if [ -n "$last_dir" ]; then
            step=$(basename "$last_dir" | sed 's/checkpoint_0*//')
            echo "  $(date '+%H:%M:%S') | $exp: step $step/100000 | procs=$count"
        fi
    done
    sleep 60
done

# 评估 Exp19 (joint abs)
echo ""
echo "[2/4] Evaluating Exp19 (joint abs)..."
$PYTHON evaluate.py \
    --checkpoint outputs/exp19_joint_abs/best \
    --num_episodes 20 \
    --max_steps 1000 \
    --video_dir outputs/eval_exp19 \
    --action_type action \
    2>&1 | tee outputs/eval_exp19.log

# 评估 Exp20 (EE delta)
echo ""
echo "[3/4] Evaluating Exp20 (EE delta)..."
$PYTHON evaluate.py \
    --checkpoint outputs/exp20_ee_delta/best \
    --num_episodes 20 \
    --max_steps 1000 \
    --video_dir outputs/eval_exp20 \
    --action_type action.ee_delta \
    2>&1 | tee outputs/eval_exp20.log

# 汇总结果
echo ""
echo "[4/4] ============================================================"
echo "  RESULTS SUMMARY"
echo "============================================================"
echo ""
echo "--- Exp19 (joint abs) ---"
grep -E "approach|avg_reward|Cov|avg_dist|SUMMARY" outputs/eval_exp19.log 2>/dev/null | tail -10
echo ""
echo "--- Exp20 (EE delta) ---"
grep -E "approach|avg_reward|Cov|avg_dist|SUMMARY" outputs/eval_exp20.log 2>/dev/null | tail -10
echo ""
echo "============================================================"
echo "  All done! $(date)"
echo "============================================================"
