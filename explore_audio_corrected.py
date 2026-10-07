"""视听同步探索 - 修正声学信息（声源在脚部）"""
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


class CorrectedAcousticInfo:
    """修正的声学信息记录器 - 声源在脚部"""
    
    def __init__(self, scene, sample_rate=16000, eye_height=1.7, foot_height=0.05):
        self.scene = scene
        self.sample_rate = sample_rate
        self.eye_height = eye_height
        self.foot_height = foot_height  # 脚离地高度
        
        # 房间信息
        self.Rx, self.Hh, self.Rz = scene.room[0].cpu().numpy()
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
        
        self.acoustic_events = []
    
    def compute_rir_details(self, camera_pos):
        """
        计算详细的 RIR 信息
        
        声源：脚部（地面）
        接收器：耳朵（相机位置）
        """
        # 声源位置（脚部，地面）
        source_pos = [camera_pos[0], self.foot_height, camera_pos[2]]
        # 接收器位置（耳朵）
        receiver_pos = [camera_pos[0], self.eye_height, camera_pos[2]]
        
        rir = np.zeros(int(self.sample_rate * 1.5))
        details = []
        
        # 1. 直接声（从脚到耳朵）
        dist = np.sqrt((source_pos[0] - receiver_pos[0])**2 + 
                       (source_pos[1] - receiver_pos[1])**2 + 
                       (source_pos[2] - receiver_pos[2])**2)
        t_direct = dist / 343.0
        if t_direct < 1.5:
            sample_idx = int(t_direct * self.sample_rate)
            amplitude = 1.0 / (dist ** 2 + 0.01)
            rir[sample_idx] = amplitude
            details.append({
                'type': 'direct',
                'source': 'foot',
                'receiver': 'ear',
                'distance': float(dist),
                'time': float(t_direct),
                'sample': int(sample_idx),
                'amplitude': float(amplitude)
            })
        
        # 2. 早期反射（6 个墙面）
        walls = [
            ('left_wall', -self.Rx, 0, 'x'),
            ('right_wall', self.Rx, 1, 'x'),
            ('floor', 0, 2, 'y'),
            ('ceiling', self.Hh, 3, 'y'),
            ('back_wall', -self.Rz, 4, 'z'),
            ('front_wall', self.Rz, 5, 'z')
        ]
        
        for wall_name, wall_pos, surf_idx, axis in walls:
            # 镜像源位置（从脚部反射）
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
                # 振幅 = (1 - 吸收) / 距离²
                attenuation = (1 - absorption) / (dist ** 2 + 0.01)
                rir[sample_idx] += attenuation
                
                details.append({
                    'type': 'early_reflection',
                    'wall': wall_name,
                    'surface_index': int(surf_idx),
                    'absorption': float(absorption),
                    'distance': float(dist),
                    'time': float(t),
                    'sample': int(sample_idx),
                    'amplitude': float(attenuation),
                    'image_source': img_src.tolist()
                })
        
        # 3. 混响尾
        t_array = np.arange(len(rir)) / self.sample_rate
        decay = np.exp(-3 * t_array / (self.t60 / 3))
        rir = rir * decay
        
        # 归一化（相对于直接声）
        if np.abs(rir).max() > 0:
            rir = rir / np.abs(rir).max()
        
        return rir, details, source_pos, receiver_pos
    
    def get_room_info(self):
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
            't60_reverberation_time': float(self.t60),
            'acoustic_setup': {
                'source': 'foot',
                'source_height': float(self.foot_height),
                'receiver': 'ear',
                'receiver_height': float(self.eye_height),
                'direct_path_distance': float(self.eye_height - self.foot_height)
            }
        }


def generate_video_corrected(duration_seconds=18, fps=30, res=64, output_base="walkthrough_corrected"):
    print("=" * 60)
    print(f"生成修正版声学视频 - {duration_seconds}秒")
    print("=" * 60)
    
    device = "cpu"
    num_frames = duration_seconds * fps
    
    # 1. 生成场景
    print("\n生成场景...")
    scene = sample_scenes(1, device)
    scene.surf_absorption = torch.tensor([[0.05, 0.05, 0.15, 0.05, 0.05, 0.05]], device=device)
    
    Rx, Hh, Rz = scene.room[0]
    print(f"房间尺寸：{2*Rx:.1f} x {Hh:.1f} x {2*Rz:.1f} m")
    
    # 2. 初始化
    print("\n初始化声学信息（声源在脚部）...")
    acoustic_info = CorrectedAcousticInfo(scene, sample_rate=16000, eye_height=1.7, foot_height=0.05)
    print(f"  T60 混响时间：{acoustic_info.t60:.2f} 秒")
    print(f"  声源高度：{acoustic_info.foot_height}m (脚)")
    print(f"  接收器高度：{acoustic_info.eye_height}m (耳)")
    print(f"  直接声距离：{acoustic_info.eye_height - acoustic_info.foot_height}m")
    
    camera_pos = torch.tensor([0.0, 1.7, 0.0], device=device)
    camera_yaw = 0.0
    camera_pitch = 0.0
    
    acoustic_data = []
    frames = []
    audio_buffer = []
    
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
        
        dx = (target_x - camera_pos[0]) * 0.1
        dz = (target_z - camera_pos[2]) * 0.1
        
        new_x = camera_pos[0] + dx * math.cos(camera_yaw) - dz * math.sin(camera_yaw)
        new_z = camera_pos[2] + dx * math.sin(camera_yaw) + dz * math.cos(camera_yaw)
        
        camera_pos[0] = new_x
        camera_pos[2] = new_z
        
        target_yaw = angle
        dyaw = (target_yaw - camera_yaw)
        while dyaw > math.pi: dyaw -= 2 * math.pi
        while dyaw < -math.pi: dyaw += 2 * math.pi
        camera_yaw += dyaw * 0.5
        
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
        
        cv2.putText(rgb_img, f'X:{camera_pos[0].item():.2f}', (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(rgb_img, f'Z:{camera_pos[2].item():.2f}', (5, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        frames.append(rgb_img)
        
        samples_per_frame = 16000 // fps
        audio_buffer.extend([0.0] * samples_per_frame)
        
        # 每 10 帧计算声学信息
        if step % 10 == 0:
            rir, details, source_pos, receiver_pos = acoustic_info.compute_rir_details(
                [camera_pos[0].item(), camera_pos[1].item(), camera_pos[2].item()]
            )
            
            acoustic_data.append({
                'frame': int(step),
                'time': float(step / fps),
                'camera_position': {
                    'x': float(camera_pos[0].item()),
                    'y': float(camera_pos[1].item()),
                    'z': float(camera_pos[2].item())
                },
                'source_position': {'x': source_pos[0], 'y': source_pos[1], 'z': source_pos[2]},
                'receiver_position': {'x': receiver_pos[0], 'y': receiver_pos[1], 'z': receiver_pos[2]},
                'reflection_count': len(details),
                'reflections': details[:10]
            })
            
            # 脚步声
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
    
    # 处理音频
    audio_array = np.array(audio_buffer)
    if np.abs(audio_array).max() > 0:
        audio_array = audio_array / np.abs(audio_array).max() * 0.8
    
    # 保存
    frame_dir = Path(f"{output_base}_frames")
    frame_dir.mkdir(exist_ok=True)
    
    for i, frame in enumerate(frames):
        cv2.imwrite(str(frame_dir / f"frame_{i:04d}.png"), frame)
    
    audio_path = Path(f"{output_base}.wav")
    wavfile.write(audio_path, 16000, (audio_array * 32767).astype(np.int16))
    
    room_info_path = Path(f"{output_base}_room_info.json")
    with open(room_info_path, 'w', encoding='utf-8') as f:
        json.dump(acoustic_info.get_room_info(), f, indent=2, ensure_ascii=False)
    
    acoustic_data_path = Path(f"{output_base}_acoustic_data.json")
    with open(acoustic_data_path, 'w', encoding='utf-8') as f:
        json.dump(acoustic_data, f, indent=2, ensure_ascii=False)
    
    # 合成视频
    print("合成视频...")
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
    
    print(f"\n最终视频：{final_path.stat().st_size / 1024:.1f} KB")
    print("\n完成！")
    
    return final_path, audio_path, room_info_path, acoustic_data_path


if __name__ == "__main__":
    generate_video_corrected(duration_seconds=18, fps=30, res=64)
