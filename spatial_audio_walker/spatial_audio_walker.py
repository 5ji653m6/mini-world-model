"""
Spatial Audio Walker - Game Engine Style Spatial Audio Walking Simulator
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


@dataclass
class CameraPose:
    position: List[float]
    yaw: float
    pitch: float


class SpatialAudioEngine:
    def __init__(self, room: RoomConfig, sample_rate=48000):
        self.room = room
        self.sample_rate = sample_rate
        self.c = 343.0
        
        volume = room.width * room.height * room.depth
        area = 2 * (room.width*room.height + room.depth*room.height + room.width*room.depth)
        avg_absorption = (4*room.wall_absorption + room.floor_absorption + room.ceiling_absorption) / 6
        self.t60 = 0.161 * volume / (area * max(avg_absorption, 0.01))
        
        self.hrtf_delay = 0.0006
        
    def compute_rir(self, source_pos, receiver_pos, objects=None):
        max_order = 3
        max_time = 1.5
        num_samples = int(self.sample_rate * max_time)
        
        rir = np.zeros(num_samples)
        reflections = []
        
        # Direct sound
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
        
        # Early reflections (6 walls)
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
                    'absorption': float(absorption),
                    'image_source': img_source.tolist()
                })
        
        # Reverb tail
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


class WalkingSimulator:
    def __init__(self, room: RoomConfig, objects=None):
        self.room = room
        self.objects = objects or []
        
        self.eye_height = 1.7
        self.foot_height = 0.05
        self.body_radius = 0.25
        
        self.audio_engine = SpatialAudioEngine(room)
        
    def is_valid_position(self, pos):
        x, y, z = pos
        
        if abs(x) > self.room.width/2 - self.body_radius:
            return False
        if abs(z) > self.room.depth/2 - self.body_radius:
            return False
        if y < self.body_radius or y > self.room.height - self.body_radius:
            return False
        
        for obj in self.objects:
            obj_pos = np.array(obj.position)
            if obj.type == 'sphere':
                dist = np.linalg.norm(pos[:3] - obj_pos)
                if dist < self.body_radius + obj.size[0]:
                    return False
            else:
                half = np.array(obj.size)
                if (abs(pos[0] - obj_pos[0]) < self.body_radius + half[0] and
                    abs(pos[2] - obj_pos[2]) < self.body_radius + half[2]):
                    return False
        
        return True
    
    def generate_path(self, num_steps, path_type='spiral'):
        positions = []
        
        if path_type == 'spiral':
            radius = min(self.room.width, self.room.depth) * 0.3
            for i in range(num_steps):
                t = i / num_steps
                angle = 2 * math.pi * 1.5 * t
                x = radius * math.cos(angle)
                z = radius * math.sin(angle)
                y = self.eye_height
                
                pos = np.array([x, y, z])
                if self.is_valid_position(pos):
                    positions.append(pos)
                else:
                    positions.append(positions[-1] if positions else pos)
        
        elif path_type == 'random':
            x, z = 0.0, 0.0
            for i in range(num_steps):
                angle = np.random.uniform(0, 2 * math.pi)
                step_size = np.random.uniform(0.1, 0.3)
                
                x += step_size * math.cos(angle)
                z += step_size * math.sin(angle)
                
                x = max(-self.room.width/2 + 0.5, min(self.room.width/2 - 0.5, x))
                z = max(-self.room.depth/2 + 0.5, min(self.room.depth/2 - 0.5, z))
                
                pos = np.array([x, self.eye_height, z])
                positions.append(pos)
        
        elif path_type == 'straight':
            for i in range(num_steps):
                t = i / num_steps
                z = (t - 0.5) * self.room.depth * 0.6
                x = np.random.uniform(-0.5, 0.5)
                pos = np.array([x, self.eye_height, z])
                positions.append(pos)
        
        return positions
    
    def generate_footstep_sound(self, step_index, move_distance):
        sample_rate = 48000
        duration = 0.15
        num_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, num_samples)
        
        low_freq = np.sin(2 * math.pi * 80 * t) * np.exp(-t * 50)
        low_freq += 0.4 * np.sin(2 * math.pi * 50 * t) * np.exp(-t * 30)
        very_low = np.sin(2 * math.pi * 30 * t) * np.exp(-t * 25)
        
        step = (low_freq * 1.0 + very_low * 0.6) * min(1.0, move_distance * 3)
        step = step / np.abs(step).max() * 0.7
        
        return step
    
    def render_frame(self, camera_pos, yaw, pitch, resolution=64):
        img = np.zeros((resolution, resolution, 3), dtype=np.uint8)
        
        for y in range(resolution // 2, resolution):
            for x in range(resolution):
                u = (x / resolution + camera_pos[0] / self.room.width) * 10
                v = ((y - resolution // 2) / (resolution // 2) + camera_pos[2] / self.room.depth) * 10
                
                if (int(u) + int(v)) % 2 == 0:
                    img[y, x] = [80, 120, 80]
                else:
                    img[y, x] = [60, 100, 60]
        
        img[:resolution // 2, :] = [139, 119, 101]
        
        cv2.putText(img, f'X:{camera_pos[0]:.2f}', (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(img, f'Z:{camera_pos[2]:.2f}', (5, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return img


def generate_dataset(num_clips=12, output_dir=None):
    if output_dir is None:
        # 使用绝对路径
        script_dir = Path(__file__).parent
        output_path = script_dir.parent / "spatial_audio_dataset"
    else:
        output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("=" * 60)
    print(f"Generating Spatial Audio Dataset - {num_clips} clips")
    print("=" * 60)
    
    all_clips = []
    
    for clip_idx in range(num_clips):
        print(f"\n--- Clip {clip_idx + 1}/{num_clips} ---")
        
        room = RoomConfig(
            width=np.random.uniform(5, 10),
            height=np.random.uniform(2.5, 3.5),
            depth=np.random.uniform(5, 10)
        )
        
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
        
        print(f"Room: {room.width:.1f} x {room.height:.1f} x {room.depth:.1f} m")
        print(f"Objects: {len(objects)}")
        
        simulator = WalkingSimulator(room, objects)
        audio_engine = SpatialAudioEngine(room)
        
        path_type = np.random.choice(['spiral', 'random', 'straight'])
        num_steps = np.random.randint(450, 550)
        positions = simulator.generate_path(num_steps, path_type)
        
        print(f"Path: {path_type}, Steps: {len(positions)}")
        
        frames = []
        audio_buffer = []
        acoustic_data = []
        step_count = 0
        
        prev_pos = None
        for step, pos in enumerate(positions):
            if prev_pos is not None:
                dx = pos[0] - prev_pos[0]
                dz = pos[2] - prev_pos[2]
                yaw = math.atan2(dx, dz)
            else:
                yaw = 0.0
            pitch = 0.1 * math.sin(step * 0.1)
            
            frame = simulator.render_frame(pos, yaw, pitch, resolution=64)
            frames.append(frame)
            
            samples_per_frame = 48000 // 30
            audio_buffer.extend([0.0] * samples_per_frame)
            
            if prev_pos is not None:
                move_distance = np.linalg.norm(pos - prev_pos)
                if move_distance > 0.15:
                    footstep = simulator.generate_footstep_sound(step, move_distance)
                    
                    source_pos = np.array([pos[0], simulator.foot_height, pos[2]])
                    receiver_pos = np.array([pos[0], simulator.eye_height, pos[2]])
                    
                    rir_result = audio_engine.compute_rir(source_pos, receiver_pos, objects)
                    
                    frame_audio_start = step * samples_per_frame
                    for i in range(min(len(footstep), samples_per_frame)):
                        idx = frame_audio_start + i
                        if idx < len(audio_buffer):
                            audio_buffer[idx] += footstep[i] * 0.5
                    
                    step_count += 1
                    
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
                print(f"  Progress {step + 1}/{len(positions)}")
        
        print(f"Total steps: {step_count}")
        
        clip_dir = output_path / f"clip_{clip_idx + 1:02d}"
        clip_dir.mkdir(exist_ok=True)
        
        frame_dir = clip_dir / "frames"
        frame_dir.mkdir(exist_ok=True)
        for i, frame in enumerate(frames):
            cv2.imwrite(str(frame_dir / f"frame_{i:04d}.png"), frame)
        
        audio_array = np.array(audio_buffer)
        if np.abs(audio_array).max() > 0:
            audio_array = audio_array / np.abs(audio_array).max() * 0.8
        
        stereo_audio = np.column_stack([audio_array, audio_array])
        audio_path = clip_dir / "audio.wav"
        wavfile.write(audio_path, 48000, (stereo_audio * 32767).astype(np.int16))
        
        room_info = {
            'room': asdict(room),
            'objects': [asdict(obj) for obj in objects],
            't60': float(audio_engine.t60)
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
            'duration': len(frames) / 30.0
        }
        with open(clip_dir / "clip_info.json", 'w') as f:
            json.dump(clip_info, f, indent=2)
        
        all_clips.append(clip_info)
        print(f"Saved to: {clip_dir}")
    
    dataset_index = {
        'total_clips': len(all_clips),
        'total_duration': sum(c['duration'] for c in all_clips),
        'clips': all_clips
    }
    with open(output_path / "dataset_index.json", 'w') as f:
        json.dump(dataset_index, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print("=" * 60)
    print(f"\nTotal: {len(all_clips)} clips")
    print(f"Total duration: {dataset_index['total_duration']:.1f} seconds")
    print(f"Output directory: {output_path}")
    
    return output_path


if __name__ == "__main__":
    output_dir = generate_dataset(num_clips=12, output_dir="spatial_audio_dataset")
    print(f"\nDataset saved to: {output_dir}")
