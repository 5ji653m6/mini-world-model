"""视听同步探索 - 带详细声学信息输出"""
import torch
import math
import numpy as np
from scipy import signal
from scipy.io import wavfile
from pathlib import Path
import time
import cv2
import sys
import json

sys.path.insert(0, str(Path(__file__).parent))
from miniatlas.scene import sample_scenes, render
from miniatlas.camera import look_c2w


class DetailedAcousticInfo:
    """详细声学信息记录器"""
    
    def __init__(self, scene, sample_rate=16000):
        self.scene = scene
        self.sample_rate = sample_rate
        
        # 房间信息
        self.Rx, self.Hh, self.Rz = scene.room[0].cpu().numpy()
        
        # 表面吸声系数
        self.surf_absorption = scene.surf_absorption[0].cpu().numpy() if hasattr(scene, 'surf_absorption') else np.ones(6) * 0.1
        
        # 物体信息
        self.obj_info = []
        for i in range(len(scene.obj_on[0])):
            if scene.obj_on[0, i].item():
                self.obj_info.append({
                    'type': 'sphere' if scene.obj_type[0, i].item() == 0 else 'box',
                    'position': scene.obj_c[0, i].cpu().numpy().tolist(),
                    'size': scene.obj_h[0, i].cpu().numpy().tolist()
                })
        
        # 计算 T60
        volume = (2 * self.Rx) * self.Hh * (2 * self.Rz)
        area = 2 * (2*self.Rx*self.Hh + 2*self.Rz*self.Hh + 2*self.Rx*self.Rz)
        avg_absorption = self.surf_absorption.mean()
        self.t60 = 0.161 * volume / (area * max(avg_absorption, 0.01))
        
        # 记录所有声学事件
        self.acoustic_events = []
    
    def compute_rir_details(self, source_pos, receiver_pos):
        """
        计算详细的 RIR 信息
        
        返回：
            rir: 脉冲响应
            details: 详细反射信息列表
        """
        rir = np.zeros(int(self.sample_rate * 1.5))
        details = []
        
        # 直接声
        dist = np.linalg.norm(np.array(source_pos) - np.array(receiver_pos))
        t_direct = dist / 343.0
        if t_direct < 1.5:
            sample_idx = int(t_direct * self.sample_rate)
            rir[sample_idx] = 1.0 / (dist ** 2 + 0.01)
            details.append({
                'type': 'direct',
                'distance': dist,
                'time': t_direct,
                'sample': sample_idx,
                'amplitude': 1.0 / (dist ** 2 + 0.01)
            })
        
        # 早期反射（6 个墙面）
        walls = [
            ('left_wall', -self.Rx, 0, 'x'),
            ('right_wall', self.Rx, 1, 'x'),
            ('floor', 0, 2, 'y'),
            ('ceiling', self.Hh, 3, 'y'),
            ('back_wall', -self.Rz, 4, 'z'),
            ('front_wall', self.Rz, 5, 'z')
        ]
        
        for wall_name, wall_pos, surf_idx, axis in walls:
            # 镜像源位置
            src = np.array(source_pos)
            if axis == 'x':
                img_src = np.array([-src[0], src[1], src[2]])
            elif axis == 'y':
                img_src = np.array([src[0], 2*wall_pos - src[1], src[2]])
            else:
                img_src = np.array([src[0], src[1], 2*wall_pos - src[2]])
            
            dist = np.linalg.norm(img_src - np.array(receiver_pos))
            t = dist / 343.0
            if t < 1.5:
                sample_idx = int(t * self.sample_rate)
                absorption = self.surf_absorption[surf_idx]
                attenuation = (1 - absorption) / (dist ** 2 + 0.01)
                rir[sample_idx] += attenuation
                
                details.append({
                    'type': 'early_reflection',
                    'wall': wall_name,
                    'surface_index': surf_idx,
                    'absorption': absorption,
                    'distance': dist,
                    'time': t,
                    'sample': sample_idx,
                    'amplitude': attenuation,
                    'image_source': img_src.tolist()
                })
        
        # 混响尾
        t_array = np.arange(len(rir)) / self.sample_rate
        decay = np.exp(-3 * t_array / (self.t60 / 3))
        rir = rir * decay
        
        # 归一化
        if np.abs(rir).max() > 0:
            rir = rir / np.abs(rir).max()
        
        return rir, details
    
    def get_room_info(self):
        """获取房间信息"""
        return {
            'dimensions': {
                'width_x': float(2 * self.Rx),
                'height_y': float(self.Hh),
                'depth_z': float(2 * self.Rz)
            },
            'volume': float((2 * self.Rx) * self.Hh * (2 * self.Rz)),
            'surface_area': float(2 * (2*self.Rx*self.Hh + 2*self.Rz*self.Hh + 2*self.Rx*self.Rz)),
            'surface_absorption': {
                'left_wall': float(self.surf_absorption[0]),
                'right_wall': float(self.surf_absorption[1]),
                'floor': float(self.surf_absorption[2]),
                'ceiling': float(self.surf_absorption[3]),
                'back_wall': float(self.surf_absorption[4]),
                'front_wall': float(self.surf_absorption[5])
            },
            'objects': self.obj_info,
            't60_reverberation_time': float(self.t60)
        }


def generate_video_with_acoustic_info(duration_seconds=18, fps=30, res=64, output_base="walkthrough_acoustic"):
    """生成带详细声学信息的视频"""
    print("=" * 60)
    print(f"生成带声学信息的视频 - {duration_seconds}秒")
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
    
    # 2. 初始化声学信息记录器
    print("\n初始化声学信息记录器...")
    acoustic_info = DetailedAcousticInfo(scene, sample_rate=16000)
    print(f"  T60 混响时间：{acoustic_info.t60:.2f} 秒")
    print(f"  房间体积：{acoustic_info.get_room_info()['volume']:.1f} m3")
    
    # 相机位置
    camera_pos = torch.tensor([0.0, 1.7, 0.0], device=device)
    camera_yaw = 0.0
    camera_pitch = 0.0
    
    # 3. 生成
    frames = []
    audio_buffer = []
    acoustic_data = []  # 每帧的声学数据
    
    print("\n开始渲染...")
    start_time = time.time()
    step_count = 0
    
    for step in range(num_frames):
        # 螺旋路径
        t = step / num_frames
        circle_radius = 1.0
        circle_speed = 2 * math.pi * 1.5
        
        angle = circle_speed * t
        target_x = circle_radius * math.cos(angle)
        target_z = circle_radius * math.sin(angle)
        
        # 平滑移动
        dx = (target_x - camera_pos[0]) * 0.1
        dz = (target_z - camera_pos[2]) * 0.1
        
        new_x = camera_pos[0] + dx * math.cos(camera_yaw) - dz * math.sin(camera_yaw)
        new_z = camera_pos[2] + dx * math.sin(camera_yaw) + dz * math.cos(camera_yaw)
        
        camera_pos[0] = new_x
        camera_pos[2] = new_z
        
        # 朝向
        target_yaw = angle
        dyaw = (target_yaw - camera_yaw)
        while dyaw > math.pi: dyaw -= 2 * math.pi
        while dyaw < -math.pi: dyaw += 2 * math.pi
        camera_yaw += dyaw * 0.5
        
        # 上下视角
        pitch_target = 0.2 * math.sin(step * 0.1)
        camera_pitch += (pitch_target - camera_pitch) * 0.1
        
        # 渲染
        c2w = look_c2w(
            camera_pos.unsqueeze(0).unsqueeze(0),
            torch.tensor([camera_yaw], device=device),
            torch.tensor([camera_pitch], device=device)
        )
        
        rgb, depth = render(scene, c2w, H=res, W=res, ss=1)
        rgb_img = rgb[0, 0].permute(1, 2, 0).cpu().numpy()
        rgb_img = (rgb_img * 255).clip(0, 255).astype(np.uint8)
        
        # 添加相机位置信息到图像
        cv2.putText(rgb_img, f'X:{camera_pos[0].item():.2f}', (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(rgb_img, f'Z:{camera_pos[2].item():.2f}', (5, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(rgb_img, f'Yaw:{math.degrees(camera_yaw):.0f}°', (5, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        frames.append(rgb_img)
        
        # 音频缓冲
        samples_per_frame = 16000 // fps
        audio_buffer.extend([0.0] * samples_per_frame)
        
        # 计算详细声学信息（每 10 帧计算一次，节省时间）
        if step % 10 == 0:
            rir, details = acoustic_info.compute_rir_details(
                [camera_pos[0].item(), camera_pos[1].item(), camera_pos[2].item()],
                [camera_pos[0].item(), camera_pos[1].item(), camera_pos[2].item()]
            )
            
            # 记录声学数据
            acoustic_data.append({
                'frame': int(step),
                'time': float(step / fps),
                'camera_position': {
                    'x': float(camera_pos[0].item()),
                    'y': float(camera_pos[1].item()),
                    'z': float(camera_pos[2].item())
                },
                'camera_orientation': {
                    'yaw': float(math.degrees(camera_yaw)),
                    'pitch': float(math.degrees(camera_pitch))
                },
                'rir_samples': int(len(rir)),
                'reflection_count': int(len(details)),
                'reflections': [
                    {
                        'type': r.get('type', ''),
                        'distance': float(r.get('distance', 0)),
                        'time': float(r.get('time', 0)),
                        'amplitude': float(r.get('amplitude', 0))
                    }
                    for r in details[:10]
                ]
            })
            
            # 添加脚步声
            if step > 0 and step % 18 == 0:
                step_sound = np.random.randn(samples_per_frame) * 0.1 * np.exp(-np.arange(samples_per_frame) * 0.1)
                for i in range(min(len(step_sound), samples_per_frame)):
                    idx = step * samples_per_frame + i
                    if idx < len(audio_buffer):
                        audio_buffer[idx] += step_sound[i]
                step_count += 1
        
        if (step + 1) % 100 == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / (step + 1)) * (num_frames - step - 1)
            print(f"  进度 {step + 1}/{num_frames}, 已用 {elapsed:.1f}s, 剩余 {eta:.1f}s")
    
    total_time = time.time() - start_time
    print(f"\n渲染完成，总耗时 {total_time:.1f} 秒")
    print(f"总步数：{step_count} 步")
    
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
    
    # 保存音频
    audio_path = Path(f"{output_base}.wav")
    wavfile.write(audio_path, 16000, (audio_array * 32767).astype(np.int16))
    print(f"音频已保存到: {audio_path}")
    
    # 保存房间信息
    room_info_path = Path(f"{output_base}_room_info.json")
    with open(room_info_path, 'w') as f:
        json.dump(acoustic_info.get_room_info(), f, indent=2)
    print(f"房间信息已保存到: {room_info_path}")
    
    # 保存声学数据（每帧的反射信息）
    acoustic_data_path = Path(f"{output_base}_acoustic_data.json")
    with open(acoustic_data_path, 'w') as f:
        json.dump(acoustic_data, f, indent=2)
    print(f"声学数据已保存到: {acoustic_data_path}")
    
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
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
    print(f"\n生成的文件:")
    print(f"  - {final_path} (视频 + 音频)")
    print(f"  - {room_info_path} (房间几何和材质信息)")
    print(f"  - {acoustic_data_path} (每帧的声学反射信息)")
    
    return final_path, audio_path, room_info_path, acoustic_data_path


if __name__ == "__main__":
    video_path, audio_path, room_info_path, acoustic_data_path = generate_video_with_acoustic_info(duration_seconds=18, fps=30, res=64)
