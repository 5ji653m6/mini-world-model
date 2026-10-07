# 视频质量 Review & 音频融入训练方案

## 1. 问题诊断：为什么视频"不像人走在空间中"

### 并排对比（左：我们的 PIL walker / 右：miniatlas 原生渲染器）

见 `compare_t1.png`、`compare_t2.png`、对比视频 `compare_walk.mp4`。

### 我们的 walker 视频的 6 个硬伤

| # | 问题 | 根因（quick_improved_walker.py） |
|---|------|--------------------------------|
| 1 | **画面不随视角转动** | 渲染公式完全忽略 yaw/pitch！上半屏永远画"墙渐变"，下半屏永远画地板，行走=贴图平移 |
| 2 | **假透视** | 地板用 `world_x = px*5/py` 的粗糙近似，格子大小不随距离正确收缩，地平线固定在画面正中 |
| 3 | **没有几何体** | 物体是 2D 贴图（`screen_x = half + (obj_x-cam_x)*50`），不随相机旋转，没有遮挡关系 |
| 4 | **没有光照** | 无点光源、无漫反射、无阴影，所有像素亮度是常数渐变 |
| 5 | **没有房间结构** | 看不到墙角、墙-天花板交界、门/窗，空间感为零 |
| 6 | **debug 文字遮挡** | X/Z/Y 坐标覆盖了约 40% 画面 |

### miniatlas 原生渲染器已具备一切（我们没用它！）

`miniatlas/scene.py` 是一个 **GPU/CPU 批量 ray caster**，每像素真实投射光线：

- ✅ 真透视投影（`cam_dirs` + c2w 矩阵，FOV 75°）
- ✅ 点光源 + 硬阴影 + 环境光（`shade()` 中的 Lambert + shadow ray）
- ✅ 每房间 2-8 个物体（球体/yaw 旋转盒子，带条纹/棋盘纹理）
- ✅ 墙面图案 + 贴花（墙上的画、地板地毯）
- ✅ RGB + **精确深度图** + 精确相机位姿（训练必需）
- ✅ 解析几何，渲染 128×128 一帧仅 ~0.15s（CPU）

**结论：不需要"加强"我们的渲染器——直接丢弃 PIL 假渲染，用 miniatlas 渲染器重新生成行走数据。**

---

## 2. 核心目标：回声定位 + audio latents 融入训练

### 研究假设

> 脚步声回声携带**视线外**的空间几何信息（背后墙的距离、房间体积、反射面材质）。
> 把 audio latents 作为条件融入 world model，模型在 **look-back（回头看）** 时——
> 即视觉记忆稀疏或完全缺失的方向——渲染质量应显著提升。

### 为什么这个项目天然适合做这个实验

1. **数据 on-the-fly 生成**（`data.py:make_batch`）：场景参数完全已知 → 音频可以**解析计算**，无需真实录音
2. **rollout.py 已有完美评估协议**：轨迹 = spin → walk → **return（沿原路走回，朝向相反）**，逐帧 PSNR，按阶段分组。`return` 阶段就是 look-back！
3. **模型已有条件融合范式**：`embed_cache`（3D cache token）和 `ctx_mask`（随机 dropout）——audio 条件可以完全照搬这个模式

### 架构设计

```
Scene (参数化几何: 房间尺寸/6面墙/物体位置)
   │
   ├─► render() → RGB-D 视图（现有）
   │
   └─► audio_sim() → 双耳 IR → 脚步声卷积 → 波形 [B,2,T]   ← 新增 miniatlas/audio.py
            │
            ▼
       AudioEncoder (1D-CNN / patch) → audio tokens [B,Na,Da]  ← 新增
            │
            ▼
       MiniAtlas.forward(..., audio=audio_tokens)              ← 修改 model.py
       （拼入 token 序列，full attention，随机 dropout）
```

### 音频合成（回声定位的关键）

从 `Scene` 参数**解析计算**脉冲响应（Image Source Method，一阶反射）：

- **声源**：脚步声，位于相机下方 1.6m（脚踝），每步触发
- **接收器**：双耳，位于相机位置，随 yaw 旋转（ITD/ILD 由此产生）
- **6 面墙一阶镜像声源** → 6 个回声的**到达时间差编码了各面墙的距离（包括背后的墙！）**
- **物体遮挡**：声源-接收器路径被球/盒挡住时衰减（射线求交，复用 `_hit_objects`）
- **混响尾巴**：Schroeder 指数衰减，T60 由 Sabine 公式从房间体积/吸声系数计算

这一切在 PyTorch 里 batch 化，和 `render()` 一样快。

### 训练策略（两阶段 + 消融）

| 阶段 | 内容 | 命令要点 |
|------|------|---------|
| 0 | 复现 baseline | `train.py` 原样（无 audio） |
| 1 | +audio fine-tune | 从 baseline checkpoint `--init` 恢复，audio encoder 零初始化（不破坏已有能力） |
| 2 | 消融评估 | `rollout.py` 同一 seed 跑两次：有/无 audio 条件，对比 `return` 阶段 PSNR |

**关键细节——audio dropout**：训练时以一定概率（如 25%）丢弃 audio 条件（参照 `P_NCTX` 对 context views 的 dropout），保证模型不依赖 audio，推理时有无均可。这也是 CFG 所需的。

### 评估指标

1. **主指标**：rollout `return` 阶段 PSNR（+audio vs baseline）
2. **辅指标**：return 阶段深度图 MSE（几何准确性）
3. **诊断**：audio-ablation 曲线——只给"背后墙距离"不同的音频，看生成深度是否跟随

---

## 3. 实施计划（按依赖排序）

| Phase | 任务 | 产出 |
|-------|------|------|
| **P0** | 用 miniatlas 渲染器重新生成行走数据集（含深度、位姿、轨迹元数据）替代 PIL walker | 新数据集（真 3D） |
| **P1** | `miniatlas/audio.py`：ISM 回声合成，从 Scene 解析计算，torch 批量化 | 单元测试：回声到达时间 vs 墙距离 |
| **P2** | Audio encoder + `model.py` 融合（`embed_audio`，照搬 `embed_cache` 模式） | 前向 shape 测试 |
| **P3** | `data.py:make_batch` 集成 audio；`train.py` 加 `--audio` flag | 可训练管线 |
| **P4** | 小规模训练验证（CPU 可行：64px、dim 256、depth 6、几千步） | loss 曲线 + preview |
| **P5** | rollout 对比评估（return 阶段 PSNR） | metrics.txt 对比表 |

### 硬件现实的说明

本机只有 Intel 核显（无 CUDA）。渲染和数据生成没问题（已验证 0.15s/帧）。
训练需要**大幅缩小规模**：64px pixel-space、dim≤256、depth≤6、bs≤4，或者把训练放到有 GPU 的机器上跑（代码完全兼容，不用改）。

---

## 4. 立即可做的第一步

P0：写一个 `walker_v2`（基于 miniatlas），输出：
- `frames/`（真 3D 渲染，无 debug 文字）
- `depth/`（深度图）
- `poses.json`（每帧精确 6DoF 位姿）
- `audio.wav`（与位姿同步的脚步声+回声）
- 轨迹含 spin/walk/return 三阶段（直接对应 look-back 评估）

这份数据既是 demo 视频，也是 P4/P5 的训练/评估格式预演。
