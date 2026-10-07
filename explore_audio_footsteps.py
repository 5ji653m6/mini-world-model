"""视听同步探索 - 脚步声版本"""
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

sys.path.insert(0, str(Path(__file__).parent))
from miniatlas.scene import sample_scenes, render
from miniatlas.camera import look_c2w


class FootstepAcoustics:
    """脚步声声学系统"""
    
    def __init__(self, scene, sample_rate=16000):
        self.scene = scene
        self.sample_rate = sample_rate
        
        # 计算混响时间
        Rx, Hh, Rz = scene.room[0]
        volume = (2 * Rx) * Hh * (2 * Rz)
        area = 2 * (2*Rx*Hh + 2*Rz*Hh + 2*Rx*Rz)
        avg_absorption = scene.surf_absorption[0].mean() if hasattr(scene, 'surf_absorption') else 0.1
        self.t60 = 0.161 * volume / (area * avg_absorption.clamp(min=0.01)).item()
        
        # 生成脚步声样本（短促的冲击声 + 混响）
        self.step_sound = self._create_step_sound()
    
    def _create_step_sound(self):
        """创建一个脚步声样本"""
        duration = 0.2  # 0.2 秒
        num_samples = int(self.sample_rate * duration)
        
        # 基础冲击声（类似脚步声）
        t = np.linspace(0, duration, num_samples)
        
        # 低频冲击（鞋底接触地面）
        step = np.exp(-t * 30) * np.sin(2 * np.pi * 80 * t)
        
        # 添加一些高频成分（地面摩擦）
        step += 0.3 * np.exp(-t * 50) * np.sin(2 * np.pi * 200 * t)
        
        # 随机噪声（地面质感）
        noise = np.random.randn(num_samples) * 0.1 * np.exp(-t * 40)
        step += noise
        
        # 添加混响
        step = self._add_reverb(step)
        
        # 归一化
        step = step / np.abs(step).max() * 0.5
        
        return step
    
    def _add_reverb(self, dry_signal):
        """添加房间混响"""
        # 简单混响尾
        t60_val = self.t60 if isinstance(self.t60, (int, float)) else self.t60.item()
        reverb_duration = min(t60_val, 0.5)  # 最多 0.5 秒
        num_samples = int(self.sample_rate * reverb_duration)
        
        t = np.linspace(0, reverb_duration, num_samples)
        decay = np.exp(-3 * t / (t60_val / 3))
        
        # 简单混响（多个延迟叠加）
        reverb = np.zeros(len(dry_signal) + num_samples)
        for delay_ms in [10, 25, 40, 60]:
            delay_samples = int(self.sample_rate * delay_ms / 1000)
            if delay_samples < len(reverb):
                reverb[delay_samples:delay_samples + len(dry_signal)] += dry_signal * 0.3
        
        # 应用衰减（只应用到混响部分）
        reverb_tail = reverb[len(dry_signal):len(dry_signal) + num_samples]
        reverb_tail = reverb_tail * decay
        
        # 混合干声和混响
        return dry_signal + 0.5 * reverb[:len(dry_signal)]
    
    def get_step_sound_for_distance(self, distance):
        """根据距离获取衰减后的脚步声"""
        # 距离衰减（1/d²）
        attenuation = 1.0 / (distance ** 2 + 0.1)
        return self.step_sound * attenuation


def generate_video_with_footsteps(duration_seconds=18, fps=30, res=64, output_base="walkthrough_footsteps"):
    """生成带脚步声的视频"""
    print("=" * 60)
    print(f"生成脚步声视频 - {duration_seconds}秒")
    print("=" * 60)
    
    device = "cpu"
    num_frames = duration_seconds * fps
    print(f"需要生成 {num_frames} 帧")
    
    # 1. 生成场景
    print("\n生成场景...")
    scene = sample_scenes(1, device)
    scene.surf_absorption = torch.tensor([[0.05, 0.05, 0.15, 0.05, 0.05, 0.05]], device=device)
    
    Rx, Hh, Rz = scene.room[0]
    print(f"房间尺寸：{2*Rx:.1f} x {Hh:.1f} x {2*Rz:.1f} m")
    print(f"物体数量：{scene.obj_on[0].sum().item():.0f}")
    
    # 2. 初始化
    acoustics = FootstepAcoustics(scene, sample_rate=16000)
    print(f"混响时间 T60: {acoustics.t60:.2f} 秒")
    
    camera_pos = torch.tensor([0.0, 1.5, 0.0], device=device)
    camera_yaw = 0.0
    camera_pitch = 0.0
    
    # 3. 生成帧和脚步声
    frames = []
    audio_samples = []  # 存储所有音频采样
    
    # 脚步声参数
    step_interval = 18  # 每 18 帧一步（约 1.6 步/秒）
    step_offset = 0
    
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
        
        # 计算移动速度（用于脚步声音量）
        move_distance = math.sqrt(dx**2 + dz**2)
        is_moving = move_distance > 0.001
        
        # 创建相机位姿
        c2w = look_c2w(
            camera_pos.unsqueeze(0).unsqueeze(0),
            torch.tensor([camera_yaw], device=device),
            torch.tensor([camera_pitch], device=device)
        )
        
        # 渲染
        rgb, depth = render(scene, c2w, H=res, W=res, ss=1)
        rgb_img = rgb[0, 0].permute(1, 2, 0).cpu().numpy()
        rgb_img = (rgb_img * 255).clip(0, 255).astype(np.uint8)
        
        frames.append(rgb_img)
        
        # 生成脚步声
        # 每 step_interval 帧放一步
        if is_moving and (step + step_offset) % step_interval == 0:
            # 脚步声音量随速度变化
            step_volume = min(1.0, move_distance * 50)
            step_sound = acoustics.step_sound * step_volume * 0.8
            
            # 添加到音频流
            for i in range(len(step_sound)):
                sample_idx = step * (16000 // 30) + i
                if sample_idx < len(audio_samples):
                    audio_samples[sample_idx] += step_sound[i]
                else:
                    audio_samples.append(step_sound[i])
        
        # 进度
        if (step + 1) % 100 == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / (step + 1)) * (num_frames - step - 1)
            print(f"  进度 {step + 1}/{num_frames}, 已用 {elapsed:.1f}s, 剩余 {eta:.1f}s")
    
    total_time = time.time() - start_time
    print(f"\n渲染完成，总耗时 {total_time:.1f} 秒")
    print(f"平均速度: {num_frames / total_time:.1f} fps")
    
    # 4. 处理音频
    print("处理音频...")
    sample_rate = 16000
    total_samples = num_frames * (sample_rate // fps)
    
    # 补齐音频长度
    while len(audio_samples) < total_samples:
        audio_samples.append(0.0)
    audio_samples = audio_samples[:total_samples]
    
    audio_array = np.array(audio_samples)
    
    # 归一化
    if np.abs(audio_array).max() > 0:
        audio_array = audio_array / np.abs(audio_array).max() * 0.8
    
    # 5. 保存文件
    frame_dir = Path(f"{output_base}_frames")
    frame_dir.mkdir(exist_ok=True)
    
    print("保存帧...")
    for i, frame in enumerate(frames):
        cv2.imwrite(str(frame_dir / f"frame_{i:04d}.png"), frame)
    
    # 保存音频
    audio_path = Path(f"{output_base}.wav")
    wavfile.write(audio_path, sample_rate, (audio_array * 32767).astype(np.int16))
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
    
    # 7. H.264 编码
    print("转换为 H.264...")
    import imageio.v3 as iio
    frames_list = [iio.imread(fp) for fp in sorted(frame_dir.glob("frame_*.png"))]
    frames_array = np.stack(frames_list, axis=0)
    
    h264_path = Path(f"{output_base}_h264.mp4")
    writer = iio.imopen(h264_path, "w", plugin="FFMPEG")
    writer.write(frames_array, fps=fps, codec='libx264')
    writer.close()
    
    size = h264_path.stat().st_size
    print(f"H.264 视频大小：{size / 1024:.1f} KB")
    
    # 8. 验证
    cap = cv2.VideoCapture(h264_path)
    if cap.isOpened():
        actual_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"实际帧数：{actual_frames}, FPS: {actual_fps}, 时长：{actual_frames / actual_fps:.2f} 秒")
        cap.release()
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
    print(f"\n生成的文件:")
    print(f"  - {h264_path} (H.264 视频)")
    print(f"  - {audio_path} (脚步声音频)")
    
    return h264_path, audio_path


if __name__ == "__main__":
    video_path, audio_path = generate_video_with_footsteps(duration_seconds=18, fps=30, res=64, output_base="walkthrough_footsteps")
