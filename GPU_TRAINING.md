# GPU 训练手册：回声定位条件的世界模型

本文档是在有 NVIDIA GPU 的机器上复现 audio-conditioned fine-tuning 实验的完整指南。

---

## 1. 环境准备

```bash
git clone https://github.com/lyk555/mini-world-model
cd mini-world-model

# Python 3.10+，安装依赖
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install numpy scipy pillow imageio imageio-ffmpeg moviepy

# 验证 GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 2. 下载 baseline checkpoint

```bash
mkdir -p runs/latent128_cache
# 从 release 下载（70 MB）
wget https://github.com/lyk555/mini-world-model/releases/download/v1.0/mini-world-model-128-cache.pt \
     -O runs/latent128_cache/ema.pt
```

这是 128px latent-space + 3D cache 的正式模型（33M 参数，训练 15000 步）。

## 3. 快速验证 baseline（可选但推荐）

```bash
# 确认预训练模型在你的 GPU 上工作（~2 分钟）
python rollout.py --ckpt runs/latent128_cache/ema.pt --seed 3 --inputs 1 \
    --out outputs/baseline_check
# 预期输出末尾：spin/walk/return 三阶段的 PSNR（参考值约 11.9/12.6/13.1 dB）
```

## 4. Audio fine-tune（主实验）

```bash
python train.py \
    --ae runs/latent128_cache/ema.pt \
    --init runs/latent128_cache/ema.pt \
    --audio --cache_extra 4 \
    --bs 16 --steps 10000 --lr 1e-4 --warmup 200 \
    --out runs/audio_finetune
```

| 参数 | 说明 |
|------|------|
| `--init` | 从 baseline 初始化（embed_audio 零初始化 → 起点 = baseline） |
| `--audio` | 启用回声条件（64 tokens × 96 维，25% dropout） |
| `--cache_extra 4` | 与 baseline 架构一致（3D cache） |
| `--bs 16` | 按显存调整（128px latent，16 GB 可跑；显存不足就 8） |

**参考时间**：RTX 4090 约 0.3 s/步 → 10000 步 ≈ 1 小时。

**检查点**：每 `--every`（默认 1000）步保存 `runs/audio_finetune/ema.pt`（自包含，捆绑 AE）。

## 5. 评估：baseline vs +audio

相同 seed 分别 rollout：

```bash
python rollout.py --ckpt runs/latent128_cache/ema.pt --seed 3 --inputs 1 \
    --out outputs/eval_baseline_s3
python rollout.py --ckpt runs/audio_finetune/ema.pt --seed 3 --inputs 1 \
    --out outputs/eval_audio_s3
```

**主指标**：`return` 阶段平均 PSNR（look-back）。每个输出目录的 `metrics.txt` 有分阶段结果，`psnr.json` 有逐帧数据。

**统计显著性**：至少跑 4 个 seed（3, 7, 11, 42），对比 return 阶段 PSNR 的均值差和逐帧配对差（paired）。

## 6. 消融实验（可选但推荐）

### 6a. 假音频对照（验证模型真的在读音频）

给模型**错误房间**的回声（几何不匹配），若 return PSNR 下降 → 模型确实在使用音频：

```bash
# 在 rollout.py 中临时打乱 scn（不同 seed 的房间）
# 或在评估脚本里换一个 scn 调用 synth_echo
```

### 6b. 紧凑特征 baseline（消融 AudioEncoder）

`miniatlas/audio.py` 里有个更简单的 `echo_features()`（7 路径 × 2 耳的解析延迟/幅度向量，14 维），可以替代 mel 谱图 encoder，验证"学习到的音频表征 vs 手工特征"。

## 7. 架构速查

```
Scene（程序化房间参数）
  ├─► render()        → RGB-D views（视觉条件，原有）
  └─► synth_echo()    → 双耳回声波形 [B,2,T]
        └─► AudioEncoder → log-mel → patchify → 64 tokens
              └─► MiniAtlas.forward(audio=tokens, audio_mask=...)
                    tokens 拼入序列（target 之后、context 之前），full attention
```

关键文件：
- `miniatlas/audio.py` — 解析回声合成（ISM + 遮挡 + Sabine T60）
- `miniatlas/audioenc.py` — 波形 → tokens
- `miniatlas/model.py` — `embed_audio`（零初始化）+ `audio_mask`
- `AB_EXPERIMENT.md` — CPU 概念验证报告
- `REVIEW_AND_AUDIO_PLAN.md` — 完整设计文档

## 8. 故障排除

| 问题 | 解决 |
|------|------|
| OOM | `--bs 8` 或 `--bs 4` |
| `KeyError: 'model'` on `--init` | 已修复（commit 39c1cec），确认拉取最新 main |
| triton/torch.compile 报错 | 加 `--no-compile` |
| 脚步样本缺失 | 确认 `assets/footsteps/footstep00.wav` 存在（已提交到仓库） |
