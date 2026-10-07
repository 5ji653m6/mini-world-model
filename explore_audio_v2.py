"""视听同步探索 v2 - 明显的视觉变化"""
import torch
import math
import numpy as np
from scipy import signal
from scipy.io import wavfile
from pathlib import Path
import time
from PIL import Image
import cv2


class SimpleScene:
    """简化的房间场景"""
    
    def __init__(self, batch_size=1, device="cpu"):
        self.B = batch_size
        self.device = device
        
        # 房间尺寸
        self.Rx = torch.rand(batch_size, device=device) * 2.5 + 2.5
        self.Hh = torch.rand(batch_size, device=device) * 0.8 + 2.6
        self.Rz = torch.rand(batch_size, device=device) * 2.5 + 2.5
        
        # 表面吸声系数
        self.surf_absorption = torch.tensor([
            [0.05, 0.05, 0.15, 0.05, 0.05, 0.05],
        ], device=device).expand(batch_size, -1).clone()
        self.surf_absorption += torch.rand(batch_size, 6, device=device) * 0.1
    
    def get_room_bounds(self, idx=0):
        return {
            'x': (-self.Rx[idx].item(), self.Rx[idx].item()),
            'y': (0, self.Hh[idx].item()),
            'z': (-self.Rz[idx].item(), self.Rz[idx].item())
        }


class FastAcoustics:
    """超快声学计算器"""
    
    def __init__(self, scene, sample_rate=16000, max_time=1.0):
        self.scene = scene
        self.c = 343.0
        self.sample_rate = sample_rate
        self.max_time = max_time
        self.max_samples = int(sample_rate * max_time)
        
        Rx, Hh, Rz = scene.Rx[0], scene.Hh[0], scene.Rz[0]
        volume = (2 * Rx) * Hh * (2 * Rz)
        area = 2 * (2*Rx*Hh + 2*Rz*Hh + 2*Rx*Rz)
        avg_absorption = scene.surf_absorption[0].mean()
        self.t60 = 0.161 * volume / (area * avg_absorption.clamp(min=0.01)).item()
    
    def compute_rir_very_fast(self, source_pos, receiver_pos):
        rir = torch.zeros(self.max_samples)
        dist = (source_pos - receiver_pos).norm()
        t = dist / self.c
        sample_idx = int(t * self.sample_rate)
        if sample_idx < self.max_samples:
            rir[sample_idx] = 1.0 / (dist ** 2 + 0.01)
        
        t_array = torch.arange(self.max_samples) / self.sample_rate
        decay = torch.exp(-3 * t_array / (self.t60 / 3))
        rir = rir * decay
        
        if rir.abs().max() > 0:
            rir = rir / rir.abs().max()
        
        return rir


class VisualRendererV2:
    """视觉渲染器 v2 - 明显的视觉变化"""
    
    def __init__(self, scene, res=128):
        self.scene = scene
        self.res = res
        self.H, self.W = res, res
    
    def render(self, camera_pos, camera_yaw, step):
        """
        渲染当前视角 - 使用伪 3D 效果
        
        关键：让相机位置和朝向的变化产生明显的视觉位移
        """
        img = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        
        # 房间边界
        Rx, Rz = self.scene.Rx[0].item(), self.scene.Rz[0].item()
        
        # 相机在房间中的相对位置（归一化到 0-1）
        rel_x = (camera_pos[0].item() + Rx) / (2 * Rx)
        rel_z = (camera_pos[2].item() + Rz) / (2 * Rz)
        
        # 主色调根据位置变化
        base_hue = (camera_yaw + math.pi) / (2 * math.pi)  # 0-1
        
        # 绘制房间（简单的俯视网格效果）
        for y in range(self.H):
            for x in range(self.W):
                # 屏幕坐标到世界坐标的映射（考虑相机朝向）
                screen_u = (x / self.W - 0.5) * 2  # -1 到 1
                screen_v = (y / self.H - 0.5) * 2
                
                # 旋转（相机朝向）
                cos_y, sin_y = math.cos(camera_yaw), math.sin(camera_yaw)
                world_u = screen_u * cos_y - screen_v * sin_y
                world_v = screen_u * sin_y + screen_v * cos_y
                
                # 距离（深度）
                depth = 1.0 / (0.3 + abs(world_v) * 2)  # 伪透视
                
                # 根据深度和位置计算颜色
                # 地板：蓝色渐变
                floor_color = int(100 + 100 * depth)
                floor_color = min(255, max(0, floor_color))
                
                # 根据相机位置添加网格线
                grid_spacing = 0.2
                grid_u = ((rel_x + world_u * depth * 0.5) / grid_spacing) % 1
                grid_v = ((rel_z + world_v * depth * 0.5) / grid_spacing) % 1
                
                if grid_u < 0.05 or grid_v < 0.05:
                    # 网格线
                    img[y, x] = [floor_color, floor_color + 30, floor_color + 60]
                else:
                    # 地板颜色
                    img[y, x] = [floor_color - 20, floor_color, floor_color + 40]
        
        # 添加朝向指示器（箭头）
        cx, cy = self.W // 2, self.H // 2
        arrow_len = 20
        arrow_angle = -camera_yaw  # 反转 Y 轴
        
        end_x = int(cx + arrow_len * math.cos(arrow_angle))
        end_y = int(cy + arrow_len * math.sin(arrow_angle))
        
        cv2.arrowedLine(img, (cx, cy), (end_x, end_y), (255, 0, 0), 2)
        
        # 添加位置文本
        pos_text = f'X:{camera_pos[0].item():.1f} Z:{camera_pos[2].item():.1f}'
        cv2.putText(img, pos_text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 添加朝向文本
        yaw_deg = math.degrees(camera_yaw) % 360
        yaw_text = f'Yaw:{yaw_deg:.0f}°'
        cv2.putText(img, yaw_text, (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 添加帧号
        cv2.putText(img, f'Frame {step:03d}', (5, self.H - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


class InteractiveExplorerV2:
    """交互式探索器 v2"""
    
    def __init__(self, scene, res=128):
        self.scene = scene
        self.renderer = VisualRendererV2(scene, res)
        self.acoustics = FastAcoustics(scene)
        
        # 初始位置（房间中心）
        self.camera_pos = torch.tensor([0.0, 1.5, 0.0])
        self.camera_yaw = 0.0
    
    def move(self, dx=0, dz=0, dyaw=0):
        """移动相机"""
        new_x = self.camera_pos[0] + dx * math.cos(self.camera_yaw) - dz * math.sin(self.camera_yaw)
        new_z = self.camera_pos[2] + dx * math.sin(self.camera_yaw) + dz * math.cos(self.camera_yaw)
        
        # 边界检查
        bounds = self.scene.get_room_bounds()
        margin = 0.5
        new_x = max(bounds['x'][0] + margin, min(bounds['x'][1] - margin, new_x))
        new_z = max(bounds['z'][0] + margin, min(bounds['z'][1] - margin, new_z))
        
        self.camera_pos[0] = new_x
        self.camera_pos[2] = new_z
        self.camera_yaw += dyaw
    
    def get_frame_and_rir(self, step):
        """获取当前帧和 RIR"""
        rgb = self.renderer.render(self.camera_pos, self.camera_yaw, step)
        rir = self.acoustics.compute_rir_very_fast(self.camera_pos, self.camera_pos)
        return rgb, rir


def generate_video_v2(duration_seconds=18, fps=30, output_base="walkthrough_v2"):
    """生成 v2 视频（明显的视觉变化）"""
    print("=" * 60)
    print(f"生成视听视频 v2 - {duration_seconds}秒")
    print("=" * 60)
    
    num_frames = duration_seconds * fps
    print(f"需要生成 {num_frames} 帧")
    
    # 1. 初始化场景
    scene = SimpleScene(batch_size=1)
    explorer = InteractiveExplorerV2(scene, res=128)
    
    bounds = scene.get_room_bounds()
    print(f"\n房间尺寸：{2*bounds['x'][0]:.1f} x {bounds['y'][1]:.1f} x {2*bounds['z'][0]:.1f} m")
    print(f"混响时间 T60: {explorer.acoustics.t60:.2f} 秒")
    
    # 2. 生成帧和 RIR
    frames = []
    rirs = []
    
    print("\n开始生成...")
    start_time = time.time()
    
    for step in range(num_frames):
        # 螺旋路径：绕圈 + 前后移动
        t = step / num_frames
        circle_radius = 1.5
        circle_speed = 2 * math.pi * 2  # 2 圈
        
        # 圆形路径
        angle = circle_speed * t
        target_x = circle_radius * math.cos(angle)
        target_z = circle_radius * math.sin(angle)
        
        # 平滑移动
        dx = (target_x - explorer.camera_pos[0]) * 0.15
        dz = (target_z - explorer.camera_pos[2]) * 0.15
        
        # 朝向目标
        target_yaw = math.atan2(target_z, target_x)
        dyaw = (target_yaw - explorer.camera_yaw) * 0.3
        while dyaw > math.pi: dyaw -= 2 * math.pi
        while dyaw < -math.pi: dyaw += 2 * math.pi
        
        explorer.move(dx=dx, dz=dz, dyaw=dyaw)
        
        # 获取帧和 RIR
        rgb, rir = explorer.get_frame_and_rir(step)
        
        # 转换为图像
        frame = ((rgb.permute(1, 2, 0) + 0) * 255).clamp(0, 255).numpy().astype(np.uint8)
        frames.append(frame)
        rirs.append(rir.numpy())
        
        # 进度
        if (step + 1) % 100 == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / (step + 1)) * (num_frames - step - 1)
            print(f"  进度 {step + 1}/{num_frames}, 已用 {elapsed:.1f}s, 剩余 {eta:.1f}s")
    
    total_time = time.time() - start_time
    print(f"\n生成完成，总耗时 {total_time:.1f} 秒")
    print(f"平均速度: {num_frames / total_time:.1f} fps")
    
    # 3. 保存帧
    frame_dir = Path(f"{output_base}_frames")
    frame_dir.mkdir(exist_ok=True)
    
    print("保存帧...")
    for i, frame in enumerate(frames):
        cv2.imwrite(str(frame_dir / f"frame_{i:04d}.png"), frame)
    
    # 4. 合成音频
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
    
    # 5. 合成视频
    print("合成视频...")
    video_path = Path(f"{output_base}.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(video_path, fourcc, fps, (128, 128))
    
    for i, frame in enumerate(frames):
        out.write(frame)
        if (i + 1) % 100 == 0:
            print(f"  已写入 {i + 1}/{len(frames)} 帧")
    
    out.release()
    print(f"视频已保存到: {video_path}")
    
    # 6. 验证
    size = video_path.stat().st_size
    print(f"\n文件大小：{size / 1024:.1f} KB")
    
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        actual_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"实际帧数：{actual_frames}, FPS: {actual_fps}, 时长：{actual_frames / actual_fps:.2f} 秒")
        cap.release()
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
    
    return video_path, audio_path


if __name__ == "__main__":
    video_path, audio_path = generate_video_v2(duration_seconds=18, fps=30, output_base="walkthrough_v2")
