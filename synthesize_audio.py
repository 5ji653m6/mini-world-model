"""合成视听联合数据 - 验证声学数据生成的可行性

不需要训练，只需要：
1. 生成随机房间场景
2. 渲染视觉帧 (RGB-D)
3. 生成声学脉冲响应 (RIR)

使用 CPU 即可运行。
"""
import torch
import math
import time
import numpy as np
from scipy import signal
from scipy.io import wavfile
from pathlib import Path


# ============================================================================
# 1. 简化的场景生成器 (复用了 mini-world-model 的场景参数结构)
# ============================================================================

class SimpleScene:
    """简化的房间场景，包含声学属性"""
    
    def __init__(self, batch_size=1, device="cpu"):
        self.B = batch_size
        self.device = device
        
        # 房间尺寸 (Rx, Hh, Rz) - 半宽 x 高度 x 半深
        self.Rx = torch.rand(batch_size, device=device) * 2.5 + 2.5  # 2.5-5.0m
        self.Hh = torch.rand(batch_size, device=device) * 0.8 + 2.6  # 2.6-3.4m
        self.Rz = torch.rand(batch_size, device=device) * 2.5 + 2.5  # 2.5-5.0m
        
        # 表面吸声系数 (6 个面：-x, +x, 地板，天花板，-z, +z)
        # 典型值：混凝土 0.05, 石膏板 0.1, 木地板 0.15, 地毯 0.4, 窗帘 0.5
        self.surf_absorption = torch.tensor([
            [0.05, 0.05, 0.15, 0.05, 0.05, 0.05],  # 默认硬表面
        ], device=device).expand(batch_size, -1).clone()
        
        # 添加随机变化
        self.surf_absorption += torch.rand(batch_size, 6, device=device) * 0.1
        
        # 装饰物 (地毯、窗帘等) - 位置 + 吸声增强
        self.decal_on = torch.rand(batch_size, 6, device=device) < 0.5
        self.decal_absorption = torch.tensor([
            [0.0, 0.0, 0.25, 0.0, 0.25, 0.25],  # 地板/墙面装饰增强吸声
        ], device=device).expand(batch_size, -1).clone()
        
        print(f"房间尺寸：{self.Rx[0].item():.2f} x {self.Hh[0].item():.2f} x {self.Rz[0].item():.2f} m")
        print(f"平均吸声系数：{self.surf_absorption.mean().item():.3f}")
    
    def get_absorption_at_position(self, surf_idx, u, v):
        """根据 UV 位置获取局部吸声系数（考虑装饰物）"""
        base_abs = self.surf_absorption[:, surf_idx]
        decal_abs = self.decal_absorption[:, surf_idx]
        
        # 检查是否在装饰区域内（简化版）
        # 实际应该用 decal 参数计算，这里简化为随机区域
        in_decal = torch.rand_like(u) < 0.2  # 20% 区域有装饰
        
        absorption = base_abs + decal_abs * in_decal.float()
        return absorption.clamp(0, 1)


# ============================================================================
# 2. 镜像源方法 (Image Source Method) - 计算早期反射
# ============================================================================

class ImageSourceAcoustics:
    """使用镜像源方法计算房间脉冲响应"""
    
    def __init__(self, scene, sample_rate=16000, max_time=2.0, max_order=5):
        self.scene = scene
        self.c = 343.0  # 声速 m/s (20°C)
        self.sample_rate = sample_rate
        self.max_time = max_time
        self.max_samples = int(sample_rate * max_time)
        self.max_order = max_order
        
    def compute_rir(self, source_pos, receiver_pos, scene_idx=0):
        """
        计算单个位置的脉冲响应
        
        Args:
            source_pos: 声源位置 [3]
            receiver_pos: 接收器位置 [3]
            scene_idx: 场景索引 (batch 中的第几个)
        
        Returns:
            rir: 脉冲响应 [max_samples]
        """
        rir = torch.zeros(self.max_samples, device=self.scene.device)
        
        Rx = self.scene.Rx[scene_idx]
        Hh = self.scene.Hh[scene_idx]
        Rz = self.scene.Rz[scene_idx]
        
        # 递归生成镜像源
        def generate_images(order, position, energy, phase=1):
            if order > self.max_order:
                return
            
            # 6 个墙面的镜像
            walls = [
                ('x', -Rx, Rx, 0),      # 左/右墙
                ('y', 0, Hh, 1),        # 地板/天花板
                ('z', -Rz, Rz, 2),      # 前/后墙
            ]
            
            for axis, lo, hi, axis_idx in walls:
                # 计算镜像位置
                if position[axis_idx] < (lo + hi) / 2:
                    img_pos = position.clone()
                    img_pos[axis_idx] = 2 * lo - position[axis_idx]
                else:
                    img_pos = position.clone()
                    img_pos[axis_idx] = 2 * hi - position[axis_idx]
                
                # 距离和到达时间
                dist = (img_pos - receiver_pos).norm()
                t = dist / self.c
                sample_idx = int(t * self.sample_rate)
                
                if sample_idx < self.max_samples and sample_idx >= 0:
                    # 表面吸声系数
                    wall_abs = self.scene.surf_absorption[scene_idx, axis_idx * 2 + (1 if position[axis_idx] > (lo+hi)/2 else 0)]
                    
                    # 能量衰减：距离平方反比 + 表面吸收
                    attenuation = energy / (dist ** 2 + 0.01) * ((1 - wall_abs) ** order)
                    rir[sample_idx] += attenuation * phase
                    
                    # 递归生成下一阶镜像
                    generate_images(order + 1, img_pos, energy * (1 - wall_abs), -phase)
        
        # 从直接声开始
        direct_dist = (source_pos - receiver_pos).norm()
        direct_time = direct_dist / self.c
        direct_sample = int(direct_time * self.sample_rate)
        if direct_sample < self.max_samples:
            rir[direct_sample] = 1.0 / (direct_dist ** 2 + 0.01)
        
        # 生成镜像源（早期反射）
        generate_images(1, source_pos, 1.0)
        
        # 添加混响尾 (Schroeder 衰减)
        # T60 = 混响时间 (衰减 60dB 所需时间)
        # Sabine 公式: T60 = 0.161 * V / (A * alpha)
        volume = (2 * Rx) * Hh * (2 * Rz)
        area = 2 * (2*Rx*Hh + 2*Rz*Hh + 2*Rx*Rz)
        avg_absorption = self.scene.surf_absorption[scene_idx].mean()
        t60 = 0.161 * volume / (area * avg_absorption.clamp(min=0.01))
        
        # Schroeder 混响尾
        t = torch.arange(self.max_samples, device=self.scene.device) / self.sample_rate
        decay = torch.exp(-3 * t / (t60 / 3))
        rir = rir * decay
        
        # 归一化
        rir = rir / rir.abs().max()
        
        return rir


# ============================================================================
# 3. 视觉渲染 (简化版 - 仅用于验证)
# ============================================================================

def render_simple_visual(scene_idx=0, res=64):
    """
    简化的视觉渲染 - 用纯色/渐变代替真实渲染
    实际使用时可以替换为 mini-world-model 的 render() 函数
    """
    # 生成一个简单的测试图像
    rgb = torch.zeros(3, res, res)
    z = torch.zeros(res, res)
    
    # 简单的渐变表示深度
    for y in range(res):
        for x in range(res):
            depth = 2.0 + (x + y) / (2 * res) * 3.0  # 2-5m
            z[y, x] = depth
            # 根据深度着色
            rgb[0, y, x] = 0.5 + 0.3 * (1 - depth / 5)  # R
            rgb[1, y, x] = 0.4 + 0.2 * (1 - depth / 5)  # G
            rgb[2, y, x] = 0.3 + 0.1 * (1 - depth / 5)  # B
    
    return rgb, z


# ============================================================================
# 4. 音频处理工具
# ============================================================================

def rir_to_audio(rir, duration=3.0, sample_rate=16000):
    """将脉冲响应转换为可播放的音频（用白噪声激励）"""
    # 生成白噪声激励
    num_samples = int(duration * sample_rate)
    excitation = np.random.randn(num_samples) * 0.1
    
    # 卷积得到输出
    audio = signal.convolve(excitation, rir.numpy(), mode='full')
    audio = audio[:num_samples]
    
    # 归一化到 [-1, 1]
    audio = audio / np.abs(audio).max() * 0.8
    
    return audio


def compute_spectrogram(audio, sample_rate=16000, n_fft=512, hop_length=256):
    """计算短时傅里叶变换频谱图"""
    f, t, Zxx = signal.stft(audio, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop_length)
    
    # 转换为对数幅度
    log_spec = 20 * np.log10(np.abs(Zxx) + 1e-8)
    
    return log_spec, f, t


# ============================================================================
# 5. 主流程：合成视听数据
# ============================================================================

def synthesize_audio_visual_data(num_scenes=4, res=64, save_dir="synthesized_data"):
    """
    合成视听联合数据
    
    Args:
        num_scenes: 生成的场景数量
        res: 视觉分辨率
        save_dir: 输出目录
    """
    print("=" * 60)
    print("视听数据合成测试")
    print("=" * 60)
    
    device = "cpu"  # 使用 CPU
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    total_time = 0
    
    for scene_idx in range(num_scenes):
        print(f"\n--- 场景 {scene_idx + 1}/{num_scenes} ---")
        start = time.time()
        
        # 1. 生成场景
        scene = SimpleScene(batch_size=1, device=device)
        
        # 2. 设置声源/接收器位置 (以相机位置为例)
        source_pos = torch.tensor([0.0, 1.5, 0.0], device=device)  # 声源
        receiver_pos = torch.tensor([1.0, 1.5, 1.0], device=device)  # 接收器
        
        print(f"声源位置：{source_pos.tolist()}")
        print(f"接收器位置：{receiver_pos.tolist()}")
        
        # 3. 生成脉冲响应
        print("生成脉冲响应...")
        acoustics = ImageSourceAcoustics(scene, sample_rate=16000, max_time=2.0, max_order=5)
        rir = acoustics.compute_rir(source_pos, receiver_pos, scene_idx=0)
        
        print(f"RIR 长度：{len(rir)} 采样点 ({len(rir)/16000:.2f} 秒)")
        print(f"RIR 峰值：{rir.abs().max().item():.4f}")
        
        # 4. 渲染视觉 (简化版)
        print("渲染视觉帧...")
        rgb, z = render_simple_visual(scene_idx=0, res=res)
        
        # 5. 转换为音频
        print("生成可播放音频...")
        audio = rir_to_audio(rir, duration=3.0, sample_rate=16000)
        
        # 6. 计算频谱图
        print("计算频谱图...")
        spec, freqs, times = compute_spectrogram(audio)
        
        elapsed = time.time() - start
        total_time += elapsed
        
        print(f"\n场景 {scene_idx + 1} 完成，耗时 {elapsed:.2f} 秒")
        
        # 7. 保存数据
        # 保存 RIR
        rir_np = (rir.numpy() * 32767).astype(np.int16)
        wavfile.write(save_path / f"rir_scene{scene_idx}.wav", 16000, rir_np)
        
        # 保存音频
        audio_np = (audio * 32767).astype(np.int16)
        wavfile.write(save_path / f"audio_scene{scene_idx}.wav", 16000, audio_np)
        
        # 保存视觉帧
        visual_np = ((rgb.permute(1, 2, 0).numpy() + 1) / 2 * 255).astype(np.uint8)
        from PIL import Image
        Image.fromarray(visual_np).save(save_path / f"visual_scene{scene_idx}.png")
        
        # 保存元数据
        import json
        metadata = {
            "scene_idx": scene_idx,
            "room_size": [scene.Rx[0].item(), scene.Hh[0].item(), scene.Rz[0].item()],
            "source_pos": source_pos.tolist(),
            "receiver_pos": receiver_pos.tolist(),
            "sample_rate": 16000,
            "rir_duration": len(rir) / 16000,
            "spectrogram_shape": spec.shape,
            "spectrogram_freqs": len(freqs),
            "spectrogram_times": len(times)
        }
        with open(save_path / f"metadata_scene{scene_idx}.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"数据已保存到: {save_path}")
    
    print("\n" + "=" * 60)
    print(f"全部完成！共 {num_scenes} 个场景，总耗时 {total_time:.2f} 秒")
    print(f"平均每个场景: {total_time/num_scenes:.2f} 秒")
    print("=" * 60)
    
    return save_path


if __name__ == "__main__":
    # 运行合成测试
    output_dir = synthesize_audio_visual_data(num_scenes=2, res=64)
    
    print(f"\n输出目录: {output_dir}")
    print("\n生成的文件:")
    for f in sorted(output_dir.iterdir()):
        print(f"  - {f.name}")
