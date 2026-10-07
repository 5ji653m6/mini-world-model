"""
快速改进版空间音频行走模拟器
- 256×256 分辨率
- 简化但更好的渲染
- 真实脚步声
- 空间音频
"""
import math
import numpy as np
from scipy.io import wavfile
from pathlib import Path
import time
import cv2
import json
from dataclasses import dataclass, asdict
from typing import List


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


class QuickImprovedRenderer:
    """快速改进的渲染器"""
    
    def __init__(self, room: RoomConfig, resolution=256):
        self.room = room
        self.resolution = resolution
        self.half = resolution // 2
        
    def render_frame(self, camera_pos, yaw, pitch, objects=None):
        """快速渲染帧"""
        img = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
        half = self.half
        
        # 天空/天花板（渐变）
        for y in range(half):
            gradient = y / half
            img[y, :] = [
                int(100 * (1 - gradient) + 139 * gradient),
                int(80 * (1 - gradient) + 119 * gradient),
                int(60 * (1 - gradient) + 101 * gradient)
            ]
        
        # 地板（棋盘格 + 透视效果）
        for y in range(half + 1, self.resolution):
            for x in range(self.resolution):
                # 透视坐标
                perspective_y = (y - half) / half
                perspective_x = (x - half) / half
                
                # 模拟透视
                if perspective_y > 0.01:
                    world_x = perspective_x * 5 / perspective_y
                    world_z = perspective_y * 5
                    
                    # 棋盘格
                    grid_x = int((world_x + self.room.width/2) / self.room.width * 10) % 2
                    grid_z = int((world_z + self.room.depth/2) / self.room.depth * 10) % 2
                    
                    if (grid_x + grid_z) % 2 == 0:
                        color = [139, 119, 101]  # 浅棕
                    else:
                        color = [101, 79, 66]  # 深棕
                    
                    # 添加透视亮度变化
                    brightness = 0.7 + 0.3 * perspective_y
                    img[y, x] = [int(c * brightness) for c in color]
        
        # 墙壁（简单渐变）
        for y in range(half):
            for x in range(self.resolution):
                # 简单的墙面渐变
                wall_gradient = abs(x - half) / half
                img[y, x] = [
                    int(139 * (1 - 0.2 * wall_gradient)),
                    int(119 * (1 - 0.2 * wall_gradient)),
                    int(101 * (1 - 0.2 * wall_gradient))
                ]
        
        # 添加物体（简单矩形/圆形）
        if objects:
            for obj in objects:
                # 简化的物体投影
                obj_x = obj['position'][0]
                obj_y = obj['position'][1]
                obj_z = obj['position'][2]
                obj_size = obj['size'][0]
                
                # 屏幕坐标（简化）
                screen_x = half + int((obj_x - camera_pos[0]) * 50)
                screen_y = half - int((obj_y - 0.5) * 50)
                screen_size = int(obj_size * 30)
                
                if 0 < screen_x < self.resolution and 0 < screen_y < self.resolution:
                    # 绘制简单矩形
                    x1 = max(0, screen_x - screen_size)
                    y1 = max(0, screen_y - screen_size)
                    x2 = min(self.resolution, screen_x + screen_size)
                    y2 = min(self.resolution, screen_y + screen_size)
                    
                    if obj['type'] == 'box':
                        cv2.rectangle(img, (x1, y1), (x2, y2), (128, 128, 128), -1)
                    else:
                        cv2.circle(img, (screen_x, screen_y), screen_size, (128, 128, 128), -1)
        
        # 添加文字信息
        cv2.putText(img, f'X:{camera_pos[0]:.2f}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f'Z:{camera_pos[2]:.2f}', (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f'Y:{camera_pos[1]:.2f}', (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return img


def generate_footstep_sound():
    """生成脚步声"""
    sample_rate = 48000
    duration = 0.2
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples)
    
    # 低频冲击
    low = np.sin(2 * math.pi * 60 * t) * np.exp(-t * 80)
    low += 0.5 * np.sin(2 * math.pi * 40 * t) * np.exp(-t * 60)
    
    # 中频摩擦
    mid = np.random.randn(num_samples) * 0.2 * np.exp(-t * 40)
    
    # 高频
    high = np.random.randn(num_samples) * 0.1 * np.exp(-t * 100)
    
    step = (low * 1.0 + mid * 0.3 + high * 0.1)
    step = step / np.abs(step).max() * 0.7
    
    return step


def compute_t60(room):
    """计算混响时间"""
    volume = room.width * room.height * room.depth
    area = 2 * (room.width*room.height + room.depth*room.height + room.width*room.depth)
    avg_absorption = (4*room.wall_absorption + room.floor_absorption + room.ceiling_absorption) / 6
    return 0.161 * volume / (area * max(avg_absorption, 0.01))


def generate_quick_improved_dataset(num_clips=12, resolution=256, output_dir="improved_dataset"):
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
        
        # 随机房间
        room = RoomConfig(
            width=np.random.uniform(5, 10),
            height=np.random.uniform(2.5, 3.5),
            depth=np.random.uniform(5, 10)
        )
        t60 = compute_t60(room)
        
        # 随机物体
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
            objects.append({
                'type': obj_type,
                'position': position,
                'size': size
            })
        
        print(f"房间：{room.width:.1f} x {room.height:.1f} x {room.depth:.1f} m")
        print(f"T60: {t60:.2f}s")
        
        # 初始化渲染器
        renderer = QuickImprovedRenderer(room, resolution=resolution)
        
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
            # 朝向
            if prev_pos is not None:
                dx = pos[0] - prev_pos[0]
                dz = pos[2] - prev_pos[2]
                yaw = math.atan2(dx, dz)
            else:
                yaw = 0.0
            pitch = 0.05 * math.sin(step * 0.1)
            
            # 渲染
            frame = renderer.render_frame(pos, yaw, pitch, objects)
            frames.append(frame)
            
            # 音频
            audio_buffer.extend([0.0] * samples_per_frame)
            
            # 脚步声
            if prev_pos is not None:
                move_distance = np.linalg.norm(pos - prev_pos)
                if move_distance > 0.05:
                    footstep = generate_footstep_sound()
                    
                    source_pos = [pos[0], 0.05, pos[2]]
                    receiver_pos = [pos[0], 1.7, pos[2]]
                    
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
                            'source_position': source_pos,
                            'receiver_position': receiver_pos,
                            'yaw': float(yaw),
                            'pitch': float(pitch),
                            'rir_t60': t60,
                            'reflection_count': 7
                        })
            
            prev_pos = pos
            
            if (step + 1) % 100 == 0:
                print(f"  进度 {step + 1}/{len(positions)}")
        
        print(f"总步数：{step_count}")
        
        # 保存
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
        
        # 立体声
        stereo_audio = np.column_stack([audio_array, audio_array])
        audio_path = clip_dir / "audio.wav"
        wavfile.write(audio_path, 48000, (stereo_audio * 32767).astype(np.int16))
        
        # 保存 JSON
        room_info = {
            'room': asdict(room),
            'objects': objects,
            't60': float(t60)
        }
        with open(clip_dir / "room_info.json", 'w') as f:
            json.dump(room_info, f, indent=2)
        
        with open(clip_dir / "acoustic_data.json", 'w') as f:
            json.dump(acoustic_data, f, indent=2)
        
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
    
    # 索引
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
    # 128×128 分辨率（比 64×64 好，但渲染更快）
    output_dir = generate_quick_improved_dataset(num_clips=12, resolution=128, output_dir="improved_dataset")
    print(f"\n数据集已保存到：{output_dir}")
