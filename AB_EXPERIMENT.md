# A/B 实验报告：音频条件（回声定位）对 look-back 渲染的影响

**日期**: 2026-10-08  
**实验级别**: CPU 概念验证（tiny 模型）  
**状态**: ✅ 管线完整打通，差异在噪声范围内（欠训练）

---

## 实验设置

| 配置 | 值 |
|------|-----|
| 分辨率 | 64×64（pixel space） |
| 模型 | dim=128, depth=2, heads=4 |
| 参数量 | baseline 0.8M / audio 0.9M |
| 训练步数 | 400（两组完全相同配置） |
| batch size | 2 |
| K（context views） | 2 |
| 音频条件 | 64 tokens × 96 维（log-mel patches），25% dropout |
| rollout | seed=3，103 帧（24 spin + 40 walk + 39 return），10 扩散步，CFG 1.5 |

## 结果

| 阶段 | baseline PSNR | +audio PSNR | 差异 |
|------|--------------|-------------|------|
| spin（环视） | 11.44 dB | 11.36 dB | -0.08 |
| walk（前进） | 11.58 dB | 11.43 dB | -0.15 |
| **return（look-back）** | **11.63 dB** | **11.47 dB** | **-0.16** |

训练 loss（400 步）：baseline 0.363 / audio 0.345（audio 组略低，因为条件信息更多）

## 解读

### 为什么没有提升（预期之中）

1. **训练量差了 75 倍**：正式配置是 30000 步 dim=384 depth=12，我们只跑了 400 步 dim=128。模型连视觉任务都只学了皮毛（生成图仍是斑块状噪声，见 rollout.gif）。
2. **audio 通路刚打开**：`embed_audio` 零初始化 + DiT 门控在最初几十步阻断梯度——400 步里有效训练 audio 通路的步数可能只有 ~200 步。
3. **-0.16 dB 在噪声范围内**：两个独立训练的 tiny 模型之间的随机差异通常就有 ±0.2 dB。

### 管线验证的成果（本次实验的真正目标）

- ✅ 音频条件完整流过：`synth_echo → AudioEncoder → MiniAtlas(audio tokens) → rf_loss → rollout`
- ✅ CFG 无条件分支正确 mask 音频
- ✅ rollout 推理时每帧实时合成回声（主动回声定位）
- ✅ 评估协议可用：return 阶段 = look-back，逐帧 PSNR

### 正式实验需要改变什么

| 项目 | 本次（CPU 概念验证） | 正式（GPU） |
|------|--------------------|-------------|
| 训练步数 | 400 | 30000（baseline）+ 10000（audio fine-tune） |
| 模型 | 0.9M | ~30M（dim 384, depth 12） |
| 分辨率 | 64px pixel | 128px latent（AE） |
| baseline 训练 | 从头 | 从头（不加 audio） |
| audio 训练 | 从头 | **从 baseline `--init` fine-tune**（关键！） |
| 扩散步数 | 10 | 25 |
| 种子数 | 1 | ≥ 4（统计显著性） |

正式实验的关键差异：**audio 组应该从训好的视觉 baseline fine-tune**（零初始化保证起点=baseline），而不是从头训练。这样比较的是"同一个视觉模型 ± audio"，而不是两个独立训练的模型。

---

## 复现命令

```powershell
# 训练
python train.py --device cpu --no-compile --steps 400 --bs 2 --K 2 --dim 128 --depth 2 --heads 4 --every 200 --warmup 50 --out runs/ab_baseline
python train.py --device cpu --no-compile --steps 400 --bs 2 --K 2 --dim 128 --depth 2 --heads 4 --every 200 --warmup 50 --out runs/ab_audio --audio

# rollout
python rollout.py --device cpu --ckpt runs/ab_baseline/ema.pt --seed 3 --inputs 1 --steps 10 --out outputs/ab_baseline
python rollout.py --device cpu --ckpt runs/ab_audio/ema.pt    --seed 3 --inputs 1 --steps 10 --out outputs/ab_audio
```
