"""视听同步探索 - 真实脚步声"""
import torch
import math
import numpy as np
from scipy import signal
from scipy.io import wavfile
from pathlib import Path
import time
import cv2
import sys

sys.path.insert(0, str(Path(__file__).parent))
from miniatlas.scene import sample_scenes, render
from miniatlas.camera import look_c2w


class RealisticFootsteps:
    """真实脚步声合成器"""
    
    def __init__(self, scene, sample_rate=16000):
        self.scene = scene
        self.sample_rate = sample_rate
        
        # 计算混响时间
        Rx, Hh, Rz = scene.room[0]
        volume = (2 * Rx) * Hh * (2 * Rz)
        area = 2 * (2*Rx*Hh + 2*Rz*Hh + 2*Rx*Rz)
        avg_absorption = scene.surf_absorption[0].mean() if hasattr(scene, 'surf_absorption') else 0.1
        self.t60 = 0.161 * volume / (area * avg_absorption.clamp(min=0.01)).item()
        
        print(f"  混响时间 T60: {self.t60:.2f} 秒")
        
        # 生成真实脚步声样本
        self.step_sound = self._create_realistic_step()
        print(f"  脚步声样本长度：{len(self.step_sound)} 采样点 ({len(self.step_sound)/sample_rate:.3f} 秒)")
    
    def _create_realistic_step(self):
        """
        创建真实的脚步声
        
        真实脚步声的特点：
        1. 极短的冲击（<50ms）
        2. 高频成分多（鞋底摩擦）
        3. 低频共鸣（地面振动）
        4. 快速衰减
        """
        duration = 0.15  # 0.15 秒，更短更真实
        num_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, num_samples)
        
        # 1. 主要冲击（高频，模拟鞋底接触）
        # 用白噪声 + 快速衰减
        impulse = np.random.randn(num_samples) * np.exp(-t * 80)
        
        # 2. 低频共鸣（地面振动）
        low_freq = np.sin(2 * np.pi * 50 * t) * np.exp(-t * 40)
        low_freq *= np.random.randn(num_samples) * 0.3  # 加一些随机性
        
        # 3. 摩擦声（中高频）
        friction = np.random.randn(num_samples) * np.exp(-t * 100) * 0.5
        
        # 4. 添加一些咔哒声（硬地面）
        click = np.zeros(num_samples)
        click[5:15] = np.random.randn(10) * 0.8 * np.exp(-np.arange(10) * 0.5)
        
        # 混合所有成分
        step = impulse * 0.6 + low_freq * 0.4 + friction * 0.5 + click * 0.3
        
        # 添加强混响（房间回声）
        step = self._add_room_reverb(step)
        
        # 归一化
        step = step / np.abs(step).max() * 0.7
        
        return step
    
    def _add_room_reverb(self, dry_signal):
        """添加房间混响"""
        t60_val = self.t60 if isinstance(self.t60, (int, float)) else self.t60.item()
        
        # 混响尾长度
        reverb_length = int(self.sample_rate * min(t60_val, 1.0))
        
        # 创建混响
        reverb = np.zeros(len(dry_signal) + reverb_length)
        
        # 早期反射（模拟房间边界反射）
        early_delays = [8, 15, 25, 35, 50]  # 毫秒
        for i, delay_ms in enumerate(early_delays):
            delay_samples = int(self.sample_rate * delay_ms / 1000)
            if delay_samples < len(reverb):
                attenuation = 0.6 ** (i + 1)
                reverb[delay_samples:delay_samples + len(dry_signal)] += dry_signal * attenuation
        
        # 混响尾（指数衰减）
        t = np.arange(reverb_length) / self.sample_rate
        decay = np.exp(-3 * t / (t60_val / 3))
        
        # 多次反射叠加
        for i, delay_ms in enumerate([100, 200, 350, 550, 800]):
            delay_samples = int(self.sample_rate * delay_ms / 1000)
            if delay_samples < len(reverb):
                attenuation = 0.5 ** (i + 1)
                tail = decay[:min(len(dry_signal), reverb_length - delay_samples)]
                if len(tail) > 0:
                    reverb[delay_samples:delay_samples + len(tail)] += dry_signal[:len(tail)] * attenuation * tail
        
        # 混合（混响占 60%）
        wet = reverb[:len(dry_signal)]
        return dry_signal + 0.6 * wet


def generate_video_with_realistic_footsteps(duration_seconds=18, fps=30, res=64, output_base="walkthrough_realistic"):
    """生成带真实脚步声的视频"""
    print("=" * 60)
    print(f"生成真实脚步声视频 - {duration_seconds}秒")
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
    
    # 2. 初始化
    print("\n初始化脚步声系统...")
    footsteps = RealisticFootsteps(scene, sample_rate=16000)
    
    camera_pos = torch.tensor([0.0, 1.5, 0.0], device=device)
    camera_yaw = 0.0
    
    # 脚步声参数
    step_distance_threshold = 0.12  # 每移动 0.12 米放一步
    
    # 3. 生成
    frames = []
    audio_buffer = []
    
    print("\n开始渲染...")
    start_time = time.time()
    step_count = 0
    
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
        
        move_distance = math.sqrt((new_x - camera_pos[0])**2 + (new_z - camera_pos[2])**2)
        
        camera_pos[0] = new_x
        camera_pos[2] = new_z
        
        # 朝向
        target_yaw = math.atan2(target_z, target_x)
        dyaw = (target_yaw - camera_yaw) * 0.3
        while dyaw > math.pi: dyaw -= 2 * math.pi
        while dyaw < -math.pi: dyaw += 2 * math.pi
        camera_yaw += dyaw
        
        # 渲染
        c2w = look_c2w(
            camera_pos.unsqueeze(0).unsqueeze(0),
            torch.tensor([camera_yaw], device=device),
            torch.tensor([0.0], device=device)
        )
        
        rgb, depth = render(scene, c2w, H=res, W=res, ss=1)
        rgb_img = rgb[0, 0].permute(1, 2, 0).cpu().numpy()
        rgb_img = (rgb_img * 255).clip(0, 255).astype(np.uint8)
        
        frames.append(rgb_img)
        
        # 音频缓冲
        samples_per_frame = footsteps.sample_rate // fps
        audio_buffer.extend([0.0] * samples_per_frame)
        
        # 触发脚步声
        if move_distance > step_distance_threshold:
            step_volume = min(1.0, move_distance * 3)
            step_sound = footsteps.step_sound * step_volume * 0.6
            
            frame_audio_start = step * samples_per_frame
            for i in range(min(len(step_sound), samples_per_frame)):
                idx = frame_audio_start + i
                if idx < len(audio_buffer):
                    audio_buffer[idx] += step_sound[i]
            
            step_count += 1
            if step_count <= 3 or step_count % 20 == 0:
                print(f"  步 {step_count}: 帧 {step}, 移动 {move_distance:.3f}m")
        
        if (step + 1) % 100 == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / (step + 1)) * (num_frames - step - 1)
            print(f"  进度 {step + 1}/{num_frames}, 已用 {elapsed:.1f}s, 剩余 {eta:.1f}s")
    
    total_time = time.time() - start_time
    print(f"\n渲染完成，总耗时 {total_time:.1f} 秒")
    print(f"总步数：{step_count} 步（约 {step_count/duration_seconds:.1f} 步/秒）")
    
    # 4. 处理音频
    audio_array = np.array(audio_buffer)
    if np.abs(audio_array).max() > 0:
        audio_array = audio_array / np.abs(audio_array).max() * 0.8
    
    # 5. 保存
    frame_dir = Path(f"{output_base}_frames")
    frame_dir.mkdir(exist_ok=True)
    
    print("保存帧...")
    for i, frame in enumerate(frames):
        cv2.imwrite(str(frame_dir / f"frame_{i:04d}.png"), frame)
    
    audio_path = Path(f"{output_base}.wav")
    wavfile.write(audio_path, 16000, (audio_array * 32767).astype(np.int16))
    print(f"音频已保存到: {audio_path}")
    
    # 6. 合成视频
    print("合成视频...")
    video_path = Path(f"{output_base}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(video_path, fourcc, fps, (res, res))
    for frame in frames:
        out.write(frame)
    out.release()
    
    # 7. H.264 + 合并
    print("转换并合并...")
    import imageio.v3 as iio
    from moviepy import VideoFileClip, AudioFileClip
    
    frames_list = [iio.imread(fp) for fp in sorted(frame_dir.glob("frame_*.png"))]
    frames_array = np.stack(frames_list, axis=0)
    
    h264_path = Path(f"{output_base}_h264.mp4")
    writer = iio.imopen(h264_path, "w", plugin="FFMPEG")
    writer.write(frames_array, fps=fps, codec='libx264')
    writer.close()
    
    final_path = Path(f"{output_base}_final.mp4")
    video = VideoFileClip(str(h264_path))
    audio = AudioFileClip(str(audio_path))
    audio = audio.with_duration(video.duration)
    video = video.with_audio(audio)
    video.write_videofile(final_path, codec='libx264', audio_codec='aac', fps=fps, temp_audiofile='temp-audio.m4a', remove_temp=True)
    video.close()
    
    size = final_path.stat().st_size
    print(f"\n最终视频：{size / 1024:.1f} KB")
    print("\n完成！")
    
    return final_path, audio_path


if __name__ == "__main__":
    video_path, audio_path = generate_video_with_realistic_footsteps(duration_seconds=18, fps=30, res=64)
