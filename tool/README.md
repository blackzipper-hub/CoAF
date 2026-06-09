# Casual_CoAF tools

## clean_finished_job_logs.py

扫描 `training/cog_video_training/logs/` 下所有 SLURM 的 `.out` / `.err` 日志，**仅处理已结束任务**（通过 `squeue` / `sacct` 判断），并压缩 tqdm 进度条等冗余输出。

### 清理内容

- 去除 ANSI 转义序列
- 同一 training step 的多条 `Steps: ...` 进度行合并为一条
- `Loading weights: ...` 中间进度合并，保留最终状态行

### 用法

在集群登录节点执行（需要 `squeue` / `sacct`）：

```bash
cd /project/llmsvgen/sunkai/robomaster_3d/Casual_CoAF

# 预览将清理哪些文件
python tool/clean_finished_job_logs.py --dry-run

# 默认：生成 *.out.cleaned / *.err.cleaned，不改动原文件
python tool/clean_finished_job_logs.py

# 原地覆盖（会先备份为 *.bak）
python tool/clean_finished_job_logs.py --in-place

# 指定日志根目录
python tool/clean_finished_job_logs.py --log-root /path/to/logs --dry-run
```

### 说明

- 正在 `squeue` 中运行的任务会被跳过
- 日志文件名需符合 `jobname-<jobid>.out` 或 `jobname-<jobid>_<array>.err`（与 sbatch `%x-%j` / `%x-%A_%a` 一致）
- `sacct` 中已无记录且不在队列里的任务，按已结束处理
