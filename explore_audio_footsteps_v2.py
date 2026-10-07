"""视听同步探索 - 脚步声 v2（精确同步 + 强混响）"""
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


class FootstepAcousticsV2:
    """脚步声声学系统 v2 - 精确同步 + 强混响"""
    
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
        
        # 生成脚步声样本（带强混响）
        self.step_sound = self._create_step_sound()
        print(f"  脚步声样本长度：{len(self.step_sound)} 采样点 ({len(self.step_sound)/sample_rate:.2f} 秒)")
    
    def _create_step_sound(self):
        """创建一个带强混响的脚步声样本"""
        duration = 0.3  # 0.3 秒
        num_samples = int(self.sample_rate * duration)
        
        # 基础冲击声
        t = np.linspace(0, duration, num_samples)
        
        # 低频冲击（鞋底接触地面）- 更强
        step = np.exp(-t * 40) * np.sin(2 * np.pi * 60 * t)
        
        # 中频成分（地面质感）
        step += 0.5 * np.exp(-t * 60) * np.sin(2 * np.pi * 120 * t)
        
        # 高频噪声
        noise = np.random.randn(num_samples) * 0.2 * np.exp(-t * 80)
        step += noise
        
        # 添加强混响
        step = self._add_strong_reverb(step)
        
        # 归一化
        step = step / np.abs(step).max() * 0.6
        
        return step
    
    def _add_strong_reverb(self, dry_signal):
        """添加强混响（多个反射 + 长衰减）"""
        t60_val = self.t60 if isinstance(self.t60, (int, float)) else self.t60.item()
        
        # 混响尾长度
        reverb_length = int(self.sample_rate * min(t60_val, 1.5))
        
        # 创建多个延迟反射（模拟房间回声）
        reverb = np.zeros(len(dry_signal) + reverb_length)
        
        # 早期反射（前 50ms 内的主要反射）
        early_delays = [5, 12, 20, 30, 45]  # 毫秒
        for i, delay_ms in enumerate(early_delays):
            delay_samples = int(self.sample_rate * delay_ms / 1000)
            if delay_samples < len(reverb):
                # 每次反射衰减
                attenuation = 0.7 ** (i + 1)
                reverb[delay_samples:delay_samples + len(dry_signal)] += dry_signal * attenuation * 0.8
        
        # 混响尾（指数衰减）
        t = np.arange(reverb_length) / self.sample_rate
        decay = np.exp(-3 * t / (t60_val / 3))
        
        # 多次延迟叠加产生混响感
        for i, delay_ms in enumerate([80, 150, 250, 400, 600, 900]):
            delay_samples = int(self.sample_rate * delay_ms / 1000)
            if delay_samples < len(reverb):
                attenuation = 0.5 ** (i + 1)
                reverb[delay_samples:delay_samples + len(dry_signal)] += dry_signal * attenuation * decay[:len(dry_signal)]
        
        # 混合干声和混响（混响占比更高）
        wet = reverb[:len(dry_signal)]
        return dry_signal + 0.8 * wet


def generate_video_with_footsteps_v2(duration_seconds=18, fps=30, res=64, output_base="walkthrough_footsteps_v2"):
    """生成带精确同步脚步声的视频 v2"""
    print("=" * 60)
    print(f"生成脚步声视频 v2 - {duration_seconds}秒")
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
    
    # 2. 初始化声学系统
    print("\n初始化脚步声系统...")
    acoustics = FootstepAcousticsV2(scene, sample_rate=16000)
    
    camera_pos = torch.tensor([0.0, 1.5, 0.0], device=device)
    camera_yaw = 0.0
    camera_pitch = 0.0
    
    # 脚步声参数
    step_distance_threshold = 0.15  # 每移动 0.15 米放一步
    last_step_position = 0.0  # 上次脚步声的位置
    
    # 3. 生成帧和脚步声
    frames = []
    audio_buffer = []  # 音频采样缓冲区
    
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
        
        # 计算移动距离
        move_distance = math.sqrt((new_x - camera_pos[0])**2 + (new_z - camera_pos[2])**2)
        
        camera_pos[0] = new_x
        camera_pos[2] = new_z
        
        # 朝向目标
        target_yaw = math.atan2(target_z, target_x)
        dyaw = (target_yaw - camera_yaw) * 0.3
        while dyaw > math.pi: dyaw -= 2 * math.pi
        while dyaw < -math.pi: dyaw += 2 * math.pi
        camera_yaw += dyaw
        
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
        
        # 添加当前帧的静音段
        samples_per_frame = acoustics.sample_rate // fps
        audio_buffer.extend([0.0] * samples_per_frame)
        
        # 根据移动距离触发脚步声
        if move_distance > step_distance_threshold:
            # 计算这一步应该放在当前帧的哪个位置
            step_volume = min(1.0, move_distance * 2)  # 音量随移动距离变化
            step_sound = acoustics.step_sound * step_volume * 0.7
            
            # 将脚步声叠加到当前帧的音频段
            frame_audio_start = step * samples_per_frame
            for i in range(min(len(step_sound), samples_per_frame)):
                idx = frame_audio_start + i
                if idx < len(audio_buffer):
                    audio_buffer[idx] += step_sound[i]
            
            last_step_position = move_distance
            
            if step % 100 == 0:
                print(f"  帧 {step}: 移动 {move_distance:.3f}m, 放一步")
        
        # 进度
        if (step + 1) % 100 == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / (step + 1)) * (num_frames - step - 1)
            print(f"  进度 {step + 1}/{num_frames}, 已用 {elapsed:.1f}s, 剩余 {eta:.1f}s")
    
    total_time = time.time() - start_time
    print(f"\n渲染完成，总耗时 {total_time:.1f} 秒")
    
    # 4. 处理音频
    print("处理音频...")
    audio_array = np.array(audio_buffer)
    
    # 归一化
    if np.abs(audio_array).max() > 0:
        audio_array = audio_array / np.abs(audio_array).max() * 0.8
    
    print(f"  音频长度：{len(audio_array)} 采样点 ({len(audio_array)/16000:.2f} 秒)")
    print(f"  音频峰值：{np.abs(audio_array).max():.4f}")
    
    # 5. 保存文件
    frame_dir = Path(f"{output_base}_frames")
    frame_dir.mkdir(exist_ok=True)
    
    print("保存帧...")
    for i, frame in enumerate(frames):
        cv2.imwrite(str(frame_dir / f"frame_{i:04d}.png"), frame)
    
    # 保存音频
    audio_path = Path(f"{output_base}.wav")
    wavfile.write(audio_path, 16000, (audio_array * 32767).astype(np.int16))
    print(f"音频已保存到: {audio_path}")
    
    # 6. 合成视频
    print("合成视频...")
    video_path = Path(f"{output_base}.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(video_path, fourcc, fps, (res, res))
    
    for i, frame in enumerate(frames):
        out.write(frame)
    
    out.release()
    print(f"视频已保存到: {video_path}")
    
    # 7. H.264 编码 + 合并音频
    print("转换为 H.264 并合并音频...")
    import imageio.v3 as iio
    from moviepy import VideoFileClip, AudioFileClip
    
    frames_list = [iio.imread(fp) for fp in sorted(frame_dir.glob("frame_*.png"))]
    frames_array = np.stack(frames_list, axis=0)
    
    h264_path = Path(f"{output_base}_h264.mp4")
    writer = iio.imopen(h264_path, "w", plugin="FFMPEG")
    writer.write(frames_array, fps=fps, codec='libx264')
    writer.close()
    
    # 合并
    final_path = Path(f"{output_base}_final.mp4")
    video = VideoFileClip(str(h264_path))
    audio = AudioFileClip(str(audio_path))
    audio = audio.with_duration(video.duration)
    video = video.with_audio(audio)
    
    video.write_videofile(
        final_path,
        codec='libx264',
        audio_codec='aac',
        fps=fps,
        temp_audiofile='temp-audio.m4a',
        remove_temp=True
    )
    video.close()
    
    size = final_path.stat().st_size
    print(f"\n最终视频大小：{size / 1024:.1f} KB ({size / 1024 / 1024:.2f} MB)")
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
    print(f"\n生成的文件:")
    print(f"  - {final_path} (最终版本，含脚步声)")
    
    return final_path, audio_path


if __name__ == "__main__":
    video_path, audio_path = generate_video_with_footsteps_v2(duration_seconds=18, fps=30, res=64)
