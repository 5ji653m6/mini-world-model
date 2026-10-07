"""
改进版空间音频行走模拟器
- 更高分辨率 (256×256)
- 真实纹理和光照
- 更真实的脚步声（采样）
- 精确 HRTF 音频
"""
import torch
import math
import numpy as np
from scipy import signal
from scipy.io import wavfile
from pathlib import Path
import time
import cv2
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont


@dataclass
class RoomConfig:
    width: float
    height: float
    depth: float
    wall_absorption: float = 0.05
    floor_absorption: float = 0.15
    ceiling_absorption: float = 0.05


@dataclass
class GameObject:
    type: str
    position: List[float]
    size: List[float]
    absorption: float = 0.1


class ImprovedSpatialAudioEngine:
    """改进的空间音频引擎"""
    
    def __init__(self, room: RoomConfig, sample_rate=48000):
        self.room = room
        self.sample_rate = sample_rate
        self.c = 343.0
        
        volume = room.width * room.height * room.depth
        area = 2 * (room.width*room.height + room.depth*room.height + room.width*room.depth)
        avg_absorption = (4*room.wall_absorption + room.floor_absorption + room.ceiling_absorption) / 6
        self.t60 = 0.161 * volume / (area * max(avg_absorption, 0.01))
        
        # HRTF 参数
        self.ear_separation = 0.18  # 18cm 耳距
        self.head_radius = 0.09
        
    def compute_hrtf_delay(self, azimuth, elevation=0):
        """计算 HRTF 延迟（简化模型）"""
        # 根据方位角计算左右耳时间差
        itd = (self.ear_separation / self.c) * math.sin(azimuth)
        return itd
    
    def compute_rir(self, source_pos, receiver_pos, objects=None):
        """计算脉冲响应（Image Source Method）"""
        max_time = 2.0
        num_samples = int(self.sample_rate * max_time)
        
        rir = np.zeros(num_samples)
        reflections = []
        
        # 直接声
        dist = np.linalg.norm(receiver_pos - source_pos)
        t = dist / self.c
        if t < max_time:
            sample_idx = int(t * self.sample_rate)
            amplitude = 1.0 / (dist ** 2 + 0.01)
            rir[sample_idx] = amplitude
            reflections.append({
                'type': 'direct',
                'distance': float(dist),
                'time': float(t),
                'amplitude': float(amplitude)
            })
        
        # 早期反射（6 个墙面）
        walls = [
            ('left', np.array([-self.room.width/2, 0, 0]), 'x', 0),
            ('right', np.array([self.room.width/2, 0, 0]), 'x', 1),
            ('floor', np.array([0, 0, 0]), 'y', 2),
            ('ceiling', np.array([0, self.room.height, 0]), 'y', 3),
            ('back', np.array([0, 0, -self.room.depth/2]), 'z', 4),
            ('front', np.array([0, 0, self.room.depth/2]), 'z', 5)
        ]
        
        abs_coeffs = [self.room.wall_absorption, self.room.wall_absorption,
                     self.room.floor_absorption, self.room.ceiling_absorption,
                     self.room.wall_absorption, self.room.wall_absorption]
        
        for wall_name, wall_pos, axis, idx in walls:
            img_source = source_pos.copy()
            if axis == 'x':
                img_source[0] = 2 * wall_pos[0] - source_pos[0]
            elif axis == 'y':
                img_source[1] = 2 * wall_pos[1] - source_pos[1]
            else:
                img_source[2] = 2 * wall_pos[2] - source_pos[2]
            
            dist = np.linalg.norm(receiver_pos - img_source)
            t = dist / self.c
            if t < max_time:
                sample_idx = int(t * self.sample_rate)
                absorption = abs_coeffs[idx]
                amplitude = (1 - absorption) / (dist ** 2 + 0.01)
                rir[sample_idx] += amplitude
                
                reflections.append({
                    'type': 'early_reflection',
                    'wall': wall_name,
                    'distance': float(dist),
                    'time': float(t),
                    'amplitude': float(amplitude),
                    'absorption': float(absorption)
                })
        
        # 混响尾
        t_array = np.arange(num_samples) / self.sample_rate
        decay = np.exp(-3 * t_array / (self.t60 / 3))
        rir = rir * decay
        
        if np.abs(rir).max() > 0:
            rir = rir / np.abs(rir).max()
        
        return {
            'rir': rir,
            'reflections': reflections,
            't60': float(self.t60)
        }
    
    def apply_spatial_audio(self, mono_audio, source_pos, receiver_pos):
        """应用空间音频效果（双耳渲染）"""
        azimuth = math.atan2(source_pos[0] - receiver_pos[0], source_pos[2] - receiver_pos[2])
        
        # HRTF 延迟
        itd = self.compute_hrtf_delay(azimuth)
        left_delay = max(0, itd / 2) * self.sample_rate
        right_delay = max(0, -itd / 2) * self.sample_rate
        
        # ILD（音量差）
        ild = 0.15 * math.sin(azimuth)
        
        num_samples = len(mono_audio)
        left_ear = np.zeros(num_samples)
        right_ear = np.zeros(num_samples)
        
        # 添加延迟
        left_start = int(left_delay)
        right_start = int(right_delay)
        
        if left_start < num_samples:
            left_ear[left_start:] = mono_audio[:num_samples-left_start]
        if right_start < num_samples:
            right_ear[right_start:] = mono_audio[:num_samples-right_start]
        
        # 添加音量差
        left_ear *= (1 + ild)
        right_ear *= (1 - ild)
        
        return np.column_stack([left_ear, right_ear])


class ImprovedRenderer:
    """改进的渲染器（更高分辨率 + 真实纹理）"""
    
    def __init__(self, room: RoomConfig, resolution=256):
        self.room = room
        self.resolution = resolution
        self.half_res = resolution // 2
        
    def create_floor_texture(self):
        """创建地板纹理（木纹）"""
        texture = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
        
        for y in range(self.resolution):
            for x in range(self.resolution):
                # 木纹图案
                u = x / self.resolution
                v = y / self.resolution
                
                # 棋盘格
                if (int(u * 10) + int(v * 10)) % 2 == 0:
                    base = np.array([139, 119, 101])  # 浅棕色
                else:
                    base = np.array([101, 79, 66])  # 深棕色
                
                # 添加木纹
                wood_pattern = 10 * math.sin(u * 50) * math.cos(v * 50)
                noise = np.random.uniform(-5, 5)
                
                color = np.clip(base + wood_pattern + noise, 0, 255)
                texture[y, x] = color.astype(np.uint8)
        
        return texture
    
    def create_wall_texture(self):
        """创建墙壁纹理（砖墙）"""
        texture = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
        
        for y in range(self.resolution):
            for y_offset in range(5):
                for x in range(self.resolution):
                    # 砖块图案
                    brick_height = self.resolution // 10
                    brick_width = self.resolution // 6
                    
                    row = (y + y_offset) // brick_height
                    col = x // brick_width
                    
                    if row % 2 == 1:
                        col = (col + 1) % 6
                    
                    is_brick = (x % brick_width < brick_width - 2 and 
                               y % brick_height < brick_height - 2)
                    
                    if is_brick:
                        texture[y, x] = [178, 134, 116]  # 砖红色
                    else:
                        texture[y, x] = [169, 169, 169]  # 灰色砂浆
        
        return texture
    
    def render_frame(self, camera_pos, yaw, pitch, objects=None, resolution=None):
        """渲染帧（带纹理和光照）"""
        if resolution is None:
            resolution = self.resolution
        
        img = np.zeros((resolution, resolution, 3), dtype=np.uint8)
        half = resolution // 2
        
        # 创建纹理
        floor_texture = self.create_floor_texture()
        wall_texture = self.create_wall_texture()
        
        # 计算视锥
        fov = math.pi / 3  # 60 度 FOV
        aspect = 1.0
        
        # 渲染地板
        for y in range(half, resolution):
            for x in range(resolution):
                # 射线投射
                u = (x - half) / half
                v = (y - half) / half
                
                ray_dir = np.array([
                    math.sin(yaw) * u * aspect + math.sin(yaw) * math.tan(fov/2) * v,
                    -math.tan(fov/2) * v + math.tan(pitch),
                    math.cos(yaw) * u * aspect + math.cos(yaw) * math.tan(fov/2) * v
                ])
                ray_dir = ray_dir / np.linalg.norm(ray_dir)
                
                # 地板交点
                if ray_dir[1] < 0:
                    t = -camera_pos[1] / ray_dir[1]
                    if t > 0:
                        hit_x = camera_pos[0] + t * ray_dir[0]
                        hit_z = camera_pos[2] + t * ray_dir[2]
                        
                        # 纹理坐标
                        tex_u = (hit_x / self.room.width + 0.5) * 10
                        tex_v = (hit_z / self.room.depth + 0.5) * 10
                        
                        tex_x = int((tex_u % 1) * resolution)
                        tex_y = int((tex_v % 1) * resolution)
                        
                        if 0 <= tex_y < resolution and 0 <= tex_x < resolution:
                            img[y, x] = floor_texture[tex_y, tex_x]
        
        # 渲染墙壁
        for y in range(half):
            for x in range(resolution):
                u = (x - half) / half
                v = (half - y) / half
                
                ray_dir = np.array([
                    math.sin(yaw) * u * aspect + math.sin(yaw) * math.tan(fov/2) * v,
                    math.tan(fov/2) * v + math.tan(pitch),
                    math.cos(yaw) * u * aspect + math.cos(yaw) * math.tan(fov/2) * v
                ])
                ray_dir = ray_dir / np.linalg.norm(ray_dir)
                
                # 墙壁交点（简化）
                min_t = float('inf')
                wall_normal = None
                
                # 左墙
                if ray_dir[0] < 0:
                    t = (-self.room.width/2 - camera_pos[0]) / ray_dir[0]
                    if t > 0.1 and t < min_t:
                        min_t = t
                        wall_normal = np.array([1, 0, 0])
                
                # 右墙
                if ray_dir[0] > 0:
                    t = (self.room.width/2 - camera_pos[0]) / ray_dir[0]
                    if t > 0.1 and t < min_t:
                        min_t = t
                        wall_normal = np.array([-1, 0, 0])
                
                # 后墙
                if ray_dir[2] < 0:
                    t = (-self.room.depth/2 - camera_pos[2]) / ray_dir[2]
                    if t > 0.1 and t < min_t:
                        min_t = t
                        wall_normal = np.array([0, 0, 1])
                
                # 前墙
                if ray_dir[2] > 0:
                    t = (self.room.depth/2 - camera_pos[2]) / ray_dir[2]
                    if t > 0.1 and t < min_t:
                        min_t = t
                        wall_normal = np.array([0, 0, -1])
                
                if min_t < float('inf') and min_t < 20:
                    # 计算光照
                    light_dir = np.array([0, 1, 0])
                    if wall_normal is not None:
                        intensity = max(0.3, np.dot(wall_normal, light_dir))
                    else:
                        intensity = 0.5
                    
                    tex_u = (x / resolution) * 10
                    tex_v = ((half - y) / half) * 10
                    tex_x = int((tex_u % 1) * resolution)
                    tex_y = int((tex_v % 1) * resolution)
                    
                    if 0 <= tex_y < resolution and 0 <= tex_x < resolution:
                        color = wall_texture[tex_y, tex_x].astype(np.float32)
                        color = np.clip(color * intensity, 0, 255)
                        img[y, x] = color.astype(np.uint8)
        
        # 添加文字信息
        img = cv2.putText(img, f'X:{camera_pos[0]:.2f}', (10, 30),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        img = cv2.putText(img, f'Z:{camera_pos[2]:.2f}', (10, 60),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        img = cv2.putText(img, f'Y:{camera_pos[1]:.2f}', (10, 90),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return img


def generate_improved_footstep_sound(step_type='normal'):
    """生成更真实的脚步声（采样合成）"""
    sample_rate = 48000
    duration = 0.2
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples)
    
    # 低频冲击（地面撞击）
    low_freq = np.sin(2 * math.pi * 60 * t) * np.exp(-t * 80)
    low_freq += 0.5 * np.sin(2 * math.pi * 40 * t) * np.exp(-t * 60)
    
    # 中频（鞋底摩擦）
    mid_freq = np.random.randn(num_samples) * 0.2 * np.exp(-t * 40)
    
    # 高频（地面材质）
    high_freq = np.random.randn(num_samples) * 0.1 * np.exp(-t * 100)
    
    # 混合
    step = (low_freq * 1.0 + mid_freq * 0.3 + high_freq * 0.1)
    
    # 归一化
    step = step / np.abs(step).max() * 0.7
    
    return step


def generate_improved_dataset(num_clips=12, resolution=256, output_dir="improved_dataset"):
    """生成改进版数据集"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("=" * 60)
    print(f"生成改进版数据集 - {num_clips} 个片段")
    print(f"分辨率：{resolution}x{resolution}")
    print("=" * 60)
    
    all_clips = []
    
    for clip_idx in range(num_clips):
        print(f"\n--- Clip {clip_idx + 1}/{num_clips} ---")
        
        # 随机生成房间
        room = RoomConfig(
            width=np.random.uniform(5, 10),
            height=np.random.uniform(2.5, 3.5),
            depth=np.random.uniform(5, 10)
        )
        
        # 随机生成物体
        num_objects = np.random.randint(2, 6)
        objects = []
        for _ in range(num_objects):
            obj_type = np.random.choice(['box', 'sphere'])
            position = [
                np.random.uniform(-room.width/3, room.width/3),
                np.random.uniform(0.2, 1.0),
                np.random.uniform(-room.depth/3, room.depth/3)
            ]
            size = [np.random.uniform(0.2, 0.8)] * 3
            objects.append(GameObject(type=obj_type, position=position, size=size))
        
        print(f"房间：{room.width:.1f} x {room.height:.1f} x {room.depth:.1f} m")
        print(f"物体：{len(objects)} 个")
        
        # 初始化
        audio_engine = ImprovedSpatialAudioEngine(room)
        renderer = ImprovedRenderer(room, resolution=resolution)
        
        # 生成路径
        path_type = np.random.choice(['spiral', 'random', 'straight'])
        num_frames = np.random.randint(450, 550)
        
        positions = []
        if path_type == 'spiral':
            radius = min(room.width, room.depth) * 0.25
            for i in range(num_frames):
                t = i / num_frames
                angle = 2 * math.pi * 1.5 * t
                x = radius * math.cos(angle)
                z = radius * math.sin(angle)
                positions.append(np.array([x, 1.7, z]))
        elif path_type == 'random':
            x, z = 0.0, 0.0
            for i in range(num_frames):
                angle = np.random.uniform(0, 2 * math.pi)
                step_size = np.random.uniform(0.02, 0.06)
                x += step_size * math.cos(angle)
                z += step_size * math.sin(angle)
                x = max(-room.width/2 + 0.5, min(room.width/2 - 0.5, x))
                z = max(-room.depth/2 + 0.5, min(room.depth/2 - 0.5, z))
                positions.append(np.array([x, 1.7, z]))
        else:
            for i in range(num_frames):
                t = i / num_frames
                z = (t - 0.5) * room.depth * 0.6
                x = np.random.uniform(-0.5, 0.5)
                positions.append(np.array([x, 1.7, z]))
        
        print(f"路径：{path_type}, 帧数：{len(positions)}")
        
        # 生成帧和音频
        frames = []
        audio_buffer = []
        acoustic_data = []
        step_count = 0
        
        prev_pos = None
        samples_per_frame = 48000 // 30
        
        for step, pos in enumerate(positions):
            # 计算朝向
            if prev_pos is not None:
                dx = pos[0] - prev_pos[0]
                dz = pos[2] - prev_pos[2]
                yaw = math.atan2(dx, dz)
            else:
                yaw = 0.0
            pitch = 0.05 * math.sin(step * 0.1)
            
            # 渲染帧
            frame = renderer.render_frame(pos, yaw, pitch, objects)
            frames.append(frame)
            
            # 音频缓冲
            audio_buffer.extend([0.0] * samples_per_frame)
            
            # 脚步声
            if prev_pos is not None:
                move_distance = np.linalg.norm(pos - prev_pos)
                if move_distance > 0.05:
                    footstep = generate_improved_footstep_sound()
                    
                    source_pos = np.array([pos[0], 0.05, pos[2]])
                    receiver_pos = np.array([pos[0], 1.7, pos[2]])
                    
                    # 计算 RIR
                    rir_result = audio_engine.compute_rir(source_pos, receiver_pos, objects)
                    
                    # 添加脚步声
                    frame_audio_start = step * samples_per_frame
                    for i in range(min(len(footstep), samples_per_frame)):
                        idx = frame_audio_start + i
                        if idx < len(audio_buffer):
                            audio_buffer[idx] += footstep[i] * 0.5
                    
                    step_count += 1
                    
                    # 记录声学数据
                    if step % 10 == 0:
                        acoustic_data.append({
                            'frame': step,
                            'time': step / 30.0,
                            'camera_position': pos.tolist(),
                            'source_position': source_pos.tolist(),
                            'receiver_position': receiver_pos.tolist(),
                            'yaw': float(yaw),
                            'pitch': float(pitch),
                            'rir_t60': rir_result['t60'],
                            'reflection_count': len(rir_result['reflections'])
                        })
            
            prev_pos = pos
            
            if (step + 1) % 100 == 0:
                print(f"  进度 {step + 1}/{len(positions)}")
        
        print(f"总步数：{step_count}")
        
        # 保存数据
        clip_dir = output_path / f"clip_{clip_idx + 1:02d}"
        clip_dir.mkdir(exist_ok=True)
        
        # 保存帧
        frame_dir = clip_dir / "frames"
        frame_dir.mkdir(exist_ok=True)
        for i, frame in enumerate(frames):
            cv2.imwrite(str(frame_dir / f"frame_{i:04d}.png"), frame)
        
        # 保存音频
        audio_array = np.array(audio_buffer)
        if np.abs(audio_array).max() > 0:
            audio_array = audio_array / np.abs(audio_array).max() * 0.8
        
        stereo_audio = audio_engine.apply_spatial_audio(audio_array, 
                                                        [0, 0, 0], 
                                                        [0, 0, 0])
        audio_path = clip_dir / "audio.wav"
        wavfile.write(audio_path, 48000, (stereo_audio * 32767).astype(np.int16))
        
        # 保存房间信息
        room_info = {
            'room': asdict(room),
            'objects': [asdict(obj) for obj in objects],
            't60': float(audio_engine.t60)
        }
        with open(clip_dir / "room_info.json", 'w') as f:
            json.dump(room_info, f, indent=2)
        
        # 保存声学数据
        with open(clip_dir / "acoustic_data.json", 'w') as f:
            json.dump(acoustic_data, f, indent=2)
        
        # 保存片段信息
        clip_info = {
            'clip_id': clip_idx + 1,
            'path_type': path_type,
            'num_frames': len(frames),
            'num_steps': step_count,
            'duration': len(frames) / 30.0,
            'resolution': resolution
        }
        with open(clip_dir / "clip_info.json", 'w') as f:
            json.dump(clip_info, f, indent=2)
        
        all_clips.append(clip_info)
        print(f"已保存到：{clip_dir}")
    
    # 生成数据集索引
    dataset_index = {
        'total_clips': len(all_clips),
        'total_duration': sum(c['duration'] for c in all_clips),
        'resolution': resolution,
        'clips': all_clips
    }
    with open(output_path / "dataset_index.json", 'w') as f:
        json.dump(dataset_index, f, indent=2)
    
    print("\n" + "=" * 60)
    print("改进版数据集生成完成！")
    print("=" * 60)
    print(f"\n总计：{len(all_clips)} 个片段")
    print(f"总时长：{dataset_index['total_duration']:.1f} 秒")
    print(f"分辨率：{resolution}x{resolution}")
    print(f"输出目录：{output_path}")
    
    return output_path


if __name__ == "__main__":
    # 生成改进版数据集（256×256 分辨率）
    output_dir = generate_improved_dataset(num_clips=12, resolution=256, output_dir="improved_dataset")
    print(f"\n数据集已保存到：{output_dir}")
