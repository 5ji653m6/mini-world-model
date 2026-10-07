"""长视频探索模式 - 视听同步（优化版）

优化：
1. 降低分辨率提高速度
2. 简化 RIR 计算
3. 批量处理
"""
import torch
import math
import numpy as np
from scipy import signal
from scipy.io import wavfile
from pathlib import Path
import time
from PIL import Image
import cv2
import imageio.v3 as iio


class SimpleScene:
    """简化的房间场景"""
    
    def __init__(self, batch_size=1, device="cpu"):
        self.B = batch_size
        self.device = device
        
        # 房间尺寸
        self.Rx = torch.rand(batch_size, device=device) * 2.5 + 2.5
        self.Hh = torch.rand(batch_size, device=device) * 0.8 + 2.6
        self.Rz = torch.rand(batch_size, device=device) * 2.5 + 2.5
        
        # 表面吸声系数
        self.surf_absorption = torch.tensor([
            [0.05, 0.05, 0.15, 0.05, 0.05, 0.05],
        ], device=device).expand(batch_size, -1).clone()
        self.surf_absorption += torch.rand(batch_size, 6, device=device) * 0.1
    
    def get_room_bounds(self, idx=0):
        return {
            'x': (-self.Rx[idx].item(), self.Rx[idx].item()),
            'y': (0, self.Hh[idx].item()),
            'z': (-self.Rz[idx].item(), self.Rz[idx].item())
        }


class FastAcoustics:
    """超快声学计算器 - 只计算直接声 + 简单混响"""
    
    def __init__(self, scene, sample_rate=16000, max_time=1.0):
        self.scene = scene
        self.c = 343.0
        self.sample_rate = sample_rate
        self.max_time = max_time
        self.max_samples = int(sample_rate * max_time)
        
        # 预计算 T60
        Rx, Hh, Rz = scene.Rx[0], scene.Hh[0], scene.Rz[0]
        volume = (2 * Rx) * Hh * (2 * Rz)
        area = 2 * (2*Rx*Hh + 2*Rz*Hh + 2*Rx*Rz)
        avg_absorption = scene.surf_absorption[0].mean()
        self.t60 = 0.161 * volume / (area * avg_absorption.clamp(min=0.01)).item()
    
    def compute_rir_very_fast(self, source_pos, receiver_pos):
        """极速版 RIR - 直接声 + 指数衰减"""
        rir = torch.zeros(self.max_samples)
        
        # 直接声
        dist = (source_pos - receiver_pos).norm()
        t = dist / self.c
        sample_idx = int(t * self.sample_rate)
        if sample_idx < self.max_samples:
            rir[sample_idx] = 1.0 / (dist ** 2 + 0.01)
        
        # 简单混响尾
        t_array = torch.arange(self.max_samples) / self.sample_rate
        decay = torch.exp(-3 * t_array / (self.t60 / 3))
        rir = rir * decay
        
        if rir.abs().max() > 0:
            rir = rir / rir.abs().max()
        
        return rir


class FastVisualRenderer:
    """极速视觉渲染器 - 用数学函数生成简单画面"""
    
    def __init__(self, scene, res=64):
        self.scene = scene
        self.res = res
        self.H, self.W = res, res
        
        # 预生成网格
        self.u = torch.linspace(-1, 1, res)
        self.v = torch.linspace(-1, 1, res)
    
    def render(self, camera_pos, camera_yaw=0):
        """快速渲染 - 用渐变表示深度"""
        # 创建网格
        u, v = torch.meshgrid(self.u, self.v, indexing='ij')
        
        # 根据相机朝向旋转
        cos_y, sin_y = math.cos(camera_yaw), math.sin(camera_yaw)
        u_rot = u * cos_y - v * sin_y
        v_rot = u * sin_y + v * cos_y
        
        # 简单的深度计算（伪 3D）
        depth = 2.0 + (u_rot + 1) * 2.0  # 2-6m
        
        # 根据深度生成颜色
        rgb = torch.zeros(3, self.H, self.W)
        rgb[0] = 0.6 + 0.2 * (1 - depth / 6)  # R
        rgb[1] = 0.5 + 0.2 * (1 - depth / 6)  # G
        rgb[2] = 0.4 + 0.1 * (1 - depth / 6)  # B
        
        # 添加一些纹理
        noise = torch.sin(u * 10) * torch.cos(v * 10) * 0.1
        rgb += noise
        
        return rgb, depth


class LongVideoExplorer:
    """长视频探索器"""
    
    def __init__(self, scene, res=64):
        self.scene = scene
        self.renderer = FastVisualRenderer(scene, res)
        self.acoustics = FastAcoustics(scene)
        
        # 初始位置
        self.camera_pos = torch.tensor([0.0, 1.5, 0.0])
        self.camera_yaw = 0.0
        
        # 移动参数
        self.move_speed = 0.02
        self.turn_speed = 0.02
    
    def move(self, dx=0, dz=0, dyaw=0):
        """移动相机"""
        new_x = self.camera_pos[0] + dx * math.cos(self.camera_yaw) - dz * math.sin(self.camera_yaw)
        new_z = self.camera_pos[2] + dx * math.sin(self.camera_yaw) + dz * math.cos(self.camera_yaw)
        
        # 边界检查
        bounds = self.scene.get_room_bounds()
        margin = 0.5
        new_x = max(bounds['x'][0] + margin, min(bounds['x'][1] - margin, new_x))
        new_z = max(bounds['z'][0] + margin, min(bounds['z'][1] - margin, new_z))
        
        self.camera_pos[0] = new_x
        self.camera_pos[2] = new_z
        self.camera_yaw += dyaw
    
    def get_frame_and_rir(self):
        """获取当前帧和 RIR"""
        rgb, depth = self.renderer.render(self.camera_pos, self.camera_yaw)
        rir = self.acoustics.compute_rir_very_fast(self.camera_pos, self.camera_pos)
        return rgb, depth, rir


def generate_long_video(duration_seconds=18, fps=30, output_base="long_walkthrough"):
    """
    生成长视频
    
    Args:
        duration_seconds: 视频时长（秒）
        fps: 帧率
        output_base: 输出文件名前缀
    """
    print("=" * 60)
    print(f"生成长视频 - {duration_seconds}秒")
    print("=" * 60)
    
    num_frames = duration_seconds * fps
    print(f"需要生成 {num_frames} 帧")
    
    # 1. 初始化场景
    scene = SimpleScene(batch_size=1)
    explorer = LongVideoExplorer(scene, res=64)
    
    bounds = scene.get_room_bounds()
    print(f"\n房间尺寸：{2*bounds['x'][0]:.1f} x {bounds['y'][1]:.1f} x {2*bounds['z'][0]:.1f} m")
    print(f"混响时间 T60: {explorer.acoustics.t60:.2f} 秒")
    
    # 2. 预分配数组
    frames = []
    rirs = []
    
    # 3. 生成路径（绕圈 + 前后移动）
    print("\n开始生成...")
    start_time = time.time()
    
    for step in range(num_frames):
        # 螺旋路径：绕圈 + 前后移动
        t = step / num_frames
        circle_radius = 1.0
        circle_speed = 2 * math.pi * 1.5  # 1.5 圈
        
        # 圆形路径
        angle = circle_speed * t
        target_x = circle_radius * math.cos(angle)
        target_z = circle_radius * math.sin(angle)
        
        # 平滑移动
        dx = (target_x - explorer.camera_pos[0]) * 0.1
        dz = (target_z - explorer.camera_pos[2]) * 0.1
        
        # 朝向目标
        target_yaw = math.atan2(target_z, target_x)
        dyaw = (target_yaw - explorer.camera_yaw) * 0.2
        # 规范化到 [-pi, pi]
        while dyaw > math.pi: dyaw -= 2 * math.pi
        while dyaw < -math.pi: dyaw += 2 * math.pi
        
        explorer.move(dx=dx, dz=dz, dyaw=dyaw)
        
        # 获取帧和 RIR
        rgb, depth, rir = explorer.get_frame_and_rir()
        
        # 转换为图像 - 添加更多视觉变化
        frame = ((rgb.permute(1, 2, 0) + 1) / 2 * 255).clamp(0, 255).numpy().astype(np.uint8)
        frame = frame.copy()  # 确保是连续的
        
        # 添加时间戳文本（让变化更明显）
        cv2.putText(frame, f'Frame {step:03d}', (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        frames.append(frame)
        rirs.append(rir.numpy())
        
        # 进度
        if (step + 1) % 50 == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / (step + 1)) * (num_frames - step - 1)
            print(f"  进度 {step + 1}/{num_frames} ({(step + 1) / num_frames * 100:.1f}%), "
                  f"已用 {elapsed:.1f}s, 剩余 {eta:.1f}s, "
                  f"速度 {fps * (step + 1) / elapsed:.1f} fps")
    
    total_time = time.time() - start_time
    print(f"\n生成完成，总耗时 {total_time:.1f} 秒")
    print(f"平均速度: {num_frames / total_time:.1f} fps")
    
    # 4. 保存帧
    frame_dir = Path(f"{output_base}_frames")
    frame_dir.mkdir(exist_ok=True)
    
    print("保存帧...")
    for i, frame in enumerate(frames):
        iio.imwrite(frame_dir / f"frame_{i:04d}.png", frame)
    
    # 5. 合成音频
    print("合成音频...")
    sample_rate = 16000
    audio_per_frame = sample_rate // fps
    total_audio_samples = num_frames * audio_per_frame
    full_audio = np.zeros(total_audio_samples)
    
    for i, rir in enumerate(rirs):
        start = i * audio_per_frame
        end = min(start + len(rir), total_audio_samples)
        audio_segment = signal.convolve([1.0], rir, mode='full')[:audio_per_frame]
        audio_len = min(len(audio_segment), end - start)
        full_audio[start:start + audio_len] += audio_segment[:audio_len]
    
    # 归一化
    if np.abs(full_audio).max() > 0:
        full_audio = full_audio / np.abs(full_audio).max() * 0.8
    
    # 保存音频
    audio_path = Path(f"{output_base}.wav")
    wavfile.write(audio_path, sample_rate, (full_audio * 32767).astype(np.int16))
    print(f"音频已保存到: {audio_path}")
    
    # 6. 合成视频
    print("合成视频...")
    video_path = Path(f"{output_base}.mp4")
    with iio.imopen(video_path, "w", plugin="FFMPEG") as file:
        for i, frame in enumerate(frames):
            file.write(frame, fps=fps)
    print(f"视频已保存到: {video_path}")
    
    # 7. 生成 FFmpeg 命令
    ffmpeg_cmd = f'ffmpeg -framerate {fps} -i "{frame_dir}/frame_%04d.png" -i "{audio_path}" -c:v libx264 -pix_fmt yuv420p -c:a aac -strict experimental -shortest "{output_base}_final.mp4"'
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
    print(f"\n生成的文件:")
    print(f"  - {video_path} (视频，无音频)")
    print(f"  - {audio_path} (音频)")
    print(f"  - {frame_dir}/ (帧序列)")
    print(f"\n如需合并音视频，运行:")
    print(f"  {ffmpeg_cmd}")
    
    return video_path, audio_path


if __name__ == "__main__":
    # 生成 18 秒视频（60 秒以上）
    video_path, audio_path = generate_long_video(duration_seconds=18, fps=30, output_base="long_walkthrough")
