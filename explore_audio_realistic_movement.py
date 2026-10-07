"""视听同步探索 - 真实人体移动（柱状身体 + 碰撞检测 + 上下视角）"""
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


class HumanBody:
    """人体模型 - 柱状身体"""
    
    def __init__(self, radius=0.25, height=1.7):
        self.radius = radius  # 身体半径（米）
        self.height = height  # 眼睛高度（米）
        self.eye_level = height  # 眼睛离地高度


class CollisionDetector:
    """碰撞检测器"""
    
    def __init__(self, scene, body):
        self.scene = scene
        self.body = body
        
        # 房间边界
        self.Rx, self.Hh, self.Rz = scene.room[0].cpu().numpy()
        
        # 物体位置
        self.obj_on = scene.obj_on[0].cpu().numpy()
        self.obj_c = scene.obj_c[0].cpu().numpy()  # 物体中心
        self.obj_h = scene.obj_h[0].cpu().numpy()  # 物体尺寸
        self.obj_type = scene.obj_type[0].cpu().numpy()  # 0=球体，1=盒子
    
    def is_valid_position(self, pos):
        """检查位置是否有效（不穿模）"""
        x, y, z = pos
        
        # 1. 检查房间边界（考虑身体半径）
        if abs(x) > self.Rx - self.body.radius:
            return False
        if abs(z) > self.Rz - self.body.radius:
            return False
        if y < self.body.radius or y > self.Hh - self.body.radius:
            return False
        
        # 2. 检查物体碰撞
        for i in range(len(self.obj_on)):
            if not self.obj_on[i]:
                continue
            
            obj_x, obj_y, obj_z = self.obj_c[i]
            
            if self.obj_type[i] == 0:  # 球体
                # 圆柱体与球体碰撞
                dx = x - obj_x
                dz = z - obj_z
                dist_2d = math.sqrt(dx*dx + dz*dz)
                r = self.obj_h[i, 0]  # 球体半径
                if dist_2d < self.body.radius + r:
                    return False
            else:  # 盒子
                # 圆柱体与 AABB 碰撞
                box_half = self.obj_h[i]
                dx = abs(x - obj_x)
                dz = abs(z - obj_z)
                if dx < self.body.radius + box_half[0] and dz < self.body.radius + box_half[2]:
                    return False
        
        return True
    
    def get_closest_valid_position(self, pos):
        """获取最近的合法位置"""
        x, y, z = pos
        
        # 房间边界
        x = max(-self.Rx + self.body.radius, min(self.Rx - self.body.radius, x))
        z = max(-self.Rz + self.body.radius, min(self.Rz - self.body.radius, z))
        y = max(self.body.height, min(self.Hh - 0.3, y))  # 眼睛高度
        
        return torch.tensor([x, y, z])


class RealisticFootsteps:
    """真实脚步声"""
    
    def __init__(self, scene, sample_rate=16000):
        self.sample_rate = sample_rate
        
        # 计算混响时间
        Rx, Hh, Rz = scene.room[0]
        volume = (2 * Rx) * Hh * (2 * Rz)
        area = 2 * (2*Rx*Hh + 2*Rz*Hh + 2*Rx*Rz)
        avg_absorption = scene.surf_absorption[0].mean() if hasattr(scene, 'surf_absorption') else 0.1
        self.t60 = 0.161 * volume / (area * avg_absorption.clamp(min=0.01)).item()
        
        print(f"  混响时间 T60: {self.t60:.2f} 秒")
        
        self.step_sound = self._create_step_sound()
    
    def _create_step_sound(self):
        duration = 0.2
        num_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, num_samples)
        
        # 低频冲击
        low_freq = np.sin(2 * np.pi * 80 * t) * np.exp(-t * 50)
        low_freq += 0.4 * np.sin(2 * np.pi * 50 * t) * np.exp(-t * 30)
        very_low = np.sin(2 * np.pi * 30 * t) * np.exp(-t * 25)
        high_freq = np.random.randn(num_samples) * np.exp(-t * 150) * 0.1
        
        step = (low_freq * 1.0 + very_low * 0.6 + high_freq * 0.2) * 0.7
        
        # 混响
        step = self._add_reverb(step)
        step = step / np.abs(step).max() * 0.7
        
        return step
    
    def _add_reverb(self, dry_signal):
        t60_val = self.t60 if isinstance(self.t60, (int, float)) else self.t60.item()
        reverb_length = int(self.sample_rate * min(t60_val, 1.0))
        reverb = np.zeros(len(dry_signal) + reverb_length)
        
        for i, delay_ms in enumerate([10, 20, 35, 50]):
            delay_samples = int(self.sample_rate * delay_ms / 1000)
            if delay_samples < len(reverb):
                reverb[delay_samples:delay_samples + len(dry_signal)] += dry_signal * (0.5 ** (i + 1))
        
        t = np.arange(reverb_length) / self.sample_rate
        decay = np.exp(-3 * t / (t60_val / 3))
        
        for i, delay_ms in enumerate([100, 250, 450, 700]):
            delay_samples = int(self.sample_rate * delay_ms / 1000)
            if delay_samples < len(reverb):
                tail = decay[:min(len(dry_signal), reverb_length - delay_samples)]
                if len(tail) > 0:
                    reverb[delay_samples:delay_samples + len(tail)] += dry_signal[:len(tail)] * (0.4 ** (i + 1)) * tail
        
        return dry_signal + 0.5 * reverb[:len(dry_signal)]


def generate_video_realistic(duration_seconds=18, fps=30, res=64, output_base="walkthrough_realistic"):
    """生成真实人体移动的视频"""
    print("=" * 60)
    print(f"生成真实人体移动视频 - {duration_seconds}秒")
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
    
    # 2. 初始化人体和碰撞检测
    print("\n初始化人体模型...")
    body = HumanBody(radius=0.25, height=1.7)
    print(f"  身体半径：{body.radius}m")
    print(f"  眼睛高度：{body.height}m")
    
    collision = CollisionDetector(scene, body)
    footsteps = RealisticFootsteps(scene, sample_rate=16000)
    
    # 初始位置（房间中心，眼睛高度）
    camera_pos = torch.tensor([0.0, body.height, 0.0], device=device)
    camera_yaw = 0.0
    camera_pitch = 0.0  # 初始平视
    
    # 移动参数
    step_distance_threshold = 0.25
    rotation_threshold = 0.05
    
    # 3. 生成
    frames = []
    audio_buffer = []
    
    print("\n开始渲染（带碰撞检测）...")
    start_time = time.time()
    move_step_count = 0
    rotation_step_count = 0
    
    # 目标位置（螺旋路径）
    for step in range(num_frames):
        t = step / num_frames
        circle_radius = 1.0  # 缩小半径，避免太靠近墙壁
        circle_speed = 2 * math.pi * 1.5
        
        angle = circle_speed * t
        target_x = circle_radius * math.cos(angle)
        target_z = circle_radius * math.sin(angle)
        
        # 平滑移动
        dx = (target_x - camera_pos[0]) * 0.1
        dz = (target_z - camera_pos[2]) * 0.1
        
        new_x = camera_pos[0] + dx * math.cos(camera_yaw) - dz * math.sin(camera_yaw)
        new_z = camera_pos[2] + dx * math.sin(camera_yaw) + dz * math.cos(camera_yaw)
        
        # 碰撞检测
        test_pos = torch.tensor([new_x, body.height, new_z], device=device)
        if collision.is_valid_position(test_pos):
            camera_pos[0] = new_x
            camera_pos[2] = new_z
        else:
            # 碰撞，尝试沿着墙壁滑动
            # 简单处理：只移动 X 或 Z 中不碰撞的方向
            test_pos_x = torch.tensor([new_x, body.height, camera_pos[2]], device=device)
            test_pos_z = torch.tensor([camera_pos[0], body.height, new_z], device=device)
            
            if collision.is_valid_position(test_pos_x):
                camera_pos[0] = new_x
            elif collision.is_valid_position(test_pos_z):
                camera_pos[2] = new_z
            # 否则保持原位
        
        # 朝向
        target_yaw = angle
        dyaw = (target_yaw - camera_yaw)
        while dyaw > math.pi: dyaw -= 2 * math.pi
        while dyaw < -math.pi: dyaw += 2 * math.pi
        camera_yaw += dyaw * 0.5
        
        # 上下视角（模拟抬头/低头看）
        # 周期性地看向上方和下方
        pitch_target = 0.2 * math.sin(step * 0.1)  # 缓慢上下摆动
        d_pitch = (pitch_target - camera_pitch) * 0.1
        camera_pitch += d_pitch
        
        # 渲染
        c2w = look_c2w(
            camera_pos.unsqueeze(0).unsqueeze(0),
            torch.tensor([camera_yaw], device=device),
            torch.tensor([camera_pitch], device=device)
        )
        
        rgb, depth = render(scene, c2w, H=res, W=res, ss=1)
        rgb_img = rgb[0, 0].permute(1, 2, 0).cpu().numpy()
        rgb_img = (rgb_img * 255).clip(0, 255).astype(np.uint8)
        
        frames.append(rgb_img)
        
        # 音频缓冲
        samples_per_frame = footsteps.sample_rate // fps
        audio_buffer.extend([0.0] * samples_per_frame)
        frame_audio_start = step * samples_per_frame
        
        # 移动距离
        move_distance = math.sqrt((camera_pos[0].item() - (camera_pos[0].item() - dx))**2 + 
                                  (camera_pos[2].item() - (camera_pos[2].item() - dz))**2)
        
        # 1. 移动时脚步声
        if move_distance > step_distance_threshold:
            step_sound = footsteps.step_sound * 0.7
            for i in range(min(len(step_sound), samples_per_frame)):
                idx = frame_audio_start + i
                if idx < len(audio_buffer):
                    audio_buffer[idx] += step_sound[i]
            move_step_count += 1
        
        # 2. 旋转时原地踏步
        if abs(dyaw) > rotation_threshold and step % 5 == 0:
            stomp_sound = footsteps.step_sound * 0.4
            for i in range(min(len(stomp_sound), samples_per_frame)):
                idx = frame_audio_start + i
                if idx < len(audio_buffer):
                    audio_buffer[idx] += stomp_sound[i]
            rotation_step_count += 1
        
        if (step + 1) % 100 == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / (step + 1)) * (num_frames - step - 1)
            print(f"  进度 {step + 1}/{num_frames}, 已用 {elapsed:.1f}s, 剩余 {eta:.1f}s")
    
    total_time = time.time() - start_time
    print(f"\n渲染完成，总耗时 {total_time:.1f} 秒")
    print(f"移动步数：{move_step_count} 步")
    print(f"旋转步数：{rotation_step_count} 步")
    print(f"总步数：{move_step_count + rotation_step_count} 步")
    
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
    video_path, audio_path = generate_video_realistic(duration_seconds=18, fps=30, res=64)
