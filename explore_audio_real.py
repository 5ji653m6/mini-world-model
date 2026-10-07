"""使用 mini-world-model 真实渲染器生成视听数据"""
import torch
import math
import numpy as np
from scipy import signal
from scipy.io import wavfile
from pathlib import Path
import time
from PIL import Image
import cv2
import sys

# 导入 mini-world-model 的模块
sys.path.insert(0, str(Path(__file__).parent))
from miniatlas.scene import sample_scenes, render, sample_positions
from miniatlas.camera import look_c2w, yaw_pitch


class RealAcoustics:
    """基于真实场景的声学计算器"""
    
    def __init__(self, scene, sample_rate=16000, max_time=1.0):
        self.scene = scene
        self.c = 343.0
        self.sample_rate = sample_rate
        self.max_time = max_time
        self.max_samples = int(sample_rate * max_time)
        
        # 计算 T60
        Rx, Hh, Rz = scene.room[0]
        volume = (2 * Rx) * Hh * (2 * Rz)
        area = 2 * (2*Rx*Hh + 2*Rz*Hh + 2*Rx*Rz)
        # 使用平均吸声系数
        avg_absorption = scene.surf_absorption[0].mean() if hasattr(scene, 'surf_absorption') else 0.1
        self.t60 = 0.161 * volume / (area * avg_absorption.clamp(min=0.01)).item()
    
    def compute_rir(self, source_pos, receiver_pos):
        """计算脉冲响应"""
        rir = torch.zeros(self.max_samples)
        dist = (source_pos - receiver_pos).norm()
        t = dist / self.c
        sample_idx = int(t * self.sample_rate)
        if sample_idx < self.max_samples:
            rir[sample_idx] = 1.0 / (dist ** 2 + 0.01)
        
        # 混响尾
        t_array = torch.arange(self.max_samples) / self.sample_rate
        decay = torch.exp(-3 * t_array / (self.t60 / 3))
        rir = rir * decay
        
        if rir.abs().max() > 0:
            rir = rir / rir.abs().max()
        
        return rir


def generate_video_real(duration_seconds=18, fps=30, res=64, output_base="walkthrough_real"):
    """使用真实渲染器生成视频"""
    print("=" * 60)
    print(f"生成视听视频（真实渲染器）- {duration_seconds}秒")
    print("=" * 60)
    
    device = "cpu"
    num_frames = duration_seconds * fps
    print(f"需要生成 {num_frames} 帧")
    
    # 1. 生成场景
    print("\n生成场景...")
    scene = sample_scenes(1, device)
    
    # 添加声学属性
    scene.surf_absorption = torch.tensor([
        [0.05, 0.05, 0.15, 0.05, 0.05, 0.05],
    ], device=device)
    
    Rx, Hh, Rz = scene.room[0]
    print(f"房间尺寸：{2*Rx:.1f} x {Hh:.1f} x {2*Rz:.1f} m")
    print(f"物体数量：{scene.obj_on[0].sum().item():.0f}")
    
    # 2. 初始化相机
    acoustics = RealAcoustics(scene, sample_rate=16000, max_time=1.0)
    print(f"混响时间 T60: {acoustics.t60:.2f} 秒")
    
    # 初始位置
    camera_pos = torch.tensor([0.0, 1.5, 0.0], device=device)
    camera_yaw = 0.0
    camera_pitch = 0.0
    
    # 3. 生成帧和 RIR
    frames = []
    rirs = []
    
    print("\n开始渲染...")
    start_time = time.time()
    
    for step in range(num_frames):
        # 螺旋路径
        t = step / num_frames
        circle_radius = 1.5
        circle_speed = 2 * math.pi * 2
        
        angle = circle_speed * t
        target_x = circle_radius * math.cos(angle)
        target_z = circle_radius * math.sin(angle)
        
        # 平滑移动
        dx = (target_x - camera_pos[0]) * 0.15
        dz = (target_z - camera_pos[2]) * 0.15
        
        new_x = camera_pos[0] + dx * math.cos(camera_yaw) - dz * math.sin(camera_yaw)
        new_z = camera_pos[2] + dx * math.sin(camera_yaw) + dz * math.cos(camera_yaw)
        
        # 边界检查
        margin = 0.5
        new_x = max(-Rx.item() + margin, min(Rx.item() - margin, new_x))
        new_z = max(-Rz.item() + margin, min(Rz.item() - margin, new_z))
        
        camera_pos[0] = new_x
        camera_pos[2] = new_z
        
        # 朝向目标
        target_yaw = math.atan2(target_z, target_x)
        dyaw = (target_yaw - camera_yaw) * 0.3
        while dyaw > math.pi: dyaw -= 2 * math.pi
        while dyaw < -math.pi: dyaw += 2 * math.pi
        camera_yaw += dyaw
        
        # 创建相机位姿矩阵
        c2w = look_c2w(
            camera_pos.unsqueeze(0).unsqueeze(0),  # [1, 1, 3]
            torch.tensor([camera_yaw], device=device),
            torch.tensor([camera_pitch], device=device)
        )  # [1, 1, 4, 4]
        
        # 渲染（真实渲染器）
        rgb, depth = render(scene, c2w, H=res, W=res, ss=1)  # [1, 1, 3, res, res], [1, 1, res, res]
        
        # 转换为图像
        rgb_img = rgb[0, 0].permute(1, 2, 0).cpu().numpy()  # [res, res, 3]
        rgb_img = (rgb_img * 255).clip(0, 255).astype(np.uint8)
        
        frames.append(rgb_img)
        
        # 计算 RIR
        rir = acoustics.compute_rir(camera_pos, camera_pos)
        rirs.append(rir.numpy())
        
        # 进度
        if (step + 1) % 100 == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / (step + 1)) * (num_frames - step - 1)
            print(f"  进度 {step + 1}/{num_frames}, 已用 {elapsed:.1f}s, 剩余 {eta:.1f}s")
    
    total_time = time.time() - start_time
    print(f"\n渲染完成，总耗时 {total_time:.1f} 秒")
    print(f"平均速度: {num_frames / total_time:.1f} fps")
    
    # 4. 保存帧
    frame_dir = Path(f"{output_base}_frames")
    frame_dir.mkdir(exist_ok=True)
    
    print("保存帧...")
    for i, frame in enumerate(frames):
        cv2.imwrite(str(frame_dir / f"frame_{i:04d}.png"), frame)
    
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
    
    if np.abs(full_audio).max() > 0:
        full_audio = full_audio / np.abs(full_audio).max() * 0.8
    
    audio_path = Path(f"{output_base}.wav")
    wavfile.write(audio_path, sample_rate, (full_audio * 32767).astype(np.int16))
    print(f"音频已保存到: {audio_path}")
    
    # 6. 合成视频
    print("合成视频...")
    video_path = Path(f"{output_base}.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(video_path, fourcc, fps, (res, res))
    
    for i, frame in enumerate(frames):
        out.write(frame)
        if (i + 1) % 100 == 0:
            print(f"  已写入 {i + 1}/{len(frames)} 帧")
    
    out.release()
    print(f"视频已保存到: {video_path}")
    
    # 7. 验证
    size = video_path.stat().st_size
    print(f"\n文件大小：{size / 1024:.1f} KB")
    
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        actual_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"实际帧数：{actual_frames}, FPS: {actual_fps}, 时长：{actual_frames / actual_fps:.2f} 秒")
        cap.release()
    
    # 8. 检查帧差异
    print("\n帧差异分析:")
    for i in range(1, min(5, len(frames))):
        diff = np.abs(frames[i-1].astype(float) - frames[i].astype(float)).mean()
        print(f"  帧{i-1} -> 帧{i}: 平均差异 = {diff:.2f}")
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
    
    return video_path, audio_path


if __name__ == "__main__":
    video_path, audio_path = generate_video_real(duration_seconds=18, fps=30, res=64, output_base="walkthrough_real")
