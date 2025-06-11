#!/bin/bash
#SBATCH --job-name=nanogpt_gpu
#SBATCH --partition=a100
#SBATCH --nodes=1                 # ✅ 指定 1 个节点（系统要求）
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=20:00:00
#SBATCH --output=/scratch/ch5r24/slurm_logs/nanogpt_%j.out
#SBATCH --error=/scratch/ch5r24/slurm_logs/nanogpt_%j.err

echo "===== JOB STARTED at $(date) on $(hostname) ====="

# 设置路径
HOME_DIR=/home/ch5r24/Research/nano_gpt_noise_train
SCRATCH_BASE=/scratch/ch5r24/Temp
SCRATCH_DIR=$SCRATCH_BASE/nano_gpt_noise_train

# 1. 清空并复制项目目录到 /scratch
echo "Cleaning existing $SCRATCH_DIR if any..."
rm -rf $SCRATCH_DIR
mkdir -p $SCRATCH_BASE
echo "Copying project to scratch..."
cp -r $HOME_DIR $SCRATCH_DIR

# 2. 激活环境并进入项目目录
source /scratch/ch5r24/anaconda3/bin/activate nanogpt
cd $SCRATCH_DIR

# 3. 执行训练任务
echo "Running training..."
python run_repeated_exps.py

# 4. 拷贝结果回 home（仅需要的部分）
echo "Copying back results..."

# 创建目标路径以确保存在
mkdir -p $HOME_DIR/out-shakespeare-char
mkdir -p $HOME_DIR/summaries

# 拷贝所有以 'both' 开头的子目录
rsync -av --include 'both*/' --include 'both*/**' --exclude '*' \
  $SCRATCH_DIR/out-shakespeare-char/ $HOME_DIR/out-shakespeare-char/

# 拷贝 summary_GPU 文件夹
rsync -av $SCRATCH_DIR/summaries/summary_GPU.csv $HOME_DIR/summaries/

echo "===== JOB FINISHED at $(date) ====="