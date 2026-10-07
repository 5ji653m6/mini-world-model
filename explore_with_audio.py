"""探索模式 - 视听同步的世界模型

功能：
1. 根据相机位置实时生成 RIR（脉冲响应）
2. 将音频与视觉帧同步
3. 支持 WASD 行走，声音随位置变化
4. 输出带声音的视频文件

使用 CPU 即可运行（实时性可能受限）
"""
import torch
import math
import numpy as np
from scipy import signal
from scipy.io import wavfile
from pathlib import Path
import time
from PIL import Image
import imageio


# ============================================================================
# 1. 场景类（复用之前的实现）
# ============================================================================

class SimpleScene:
    """简化的房间场景，包含声学属性"""
    
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
        
        # 添加随机变化
        self.surf_absorption += torch.rand(batch_size, 6, device=device) * 0.1
    
    def get_room_bounds(self, idx=0):
        """返回房间边界"""
        return {
            'x': (-self.Rx[idx].item(), self.Rx[idx].item()),
            'y': (0, self.Hh[idx].item()),
            'z': (-self.Rz[idx].item(), self.Rz[idx].item())
        }


# ============================================================================
# 2. 实时 RIR 计算器
# ============================================================================

class RealtimeAcoustics:
    """实时声学计算器 - 根据位置动态生成 RIR"""
    
    def __init__(self, scene, sample_rate=16000, max_time=1.5, max_order=4):
        self.scene = scene
        self.c = 343.0
        self.sample_rate = sample_rate
        self.max_time = max_time
        self.max_samples = int(sample_rate * max_time)
        self.max_order = max_order
        
        # 预计算混响参数
        self.t60 = self._calculate_t60()
    
    def _calculate_t60(self):
        """使用 Sabine 公式计算混响时间"""
        Rx, Hh, Rz = self.scene.Rx[0], self.scene.Hh[0], self.scene.Rz[0]
        volume = (2 * Rx) * Hh * (2 * Rz)
        area = 2 * (2*Rx*Hh + 2*Rz*Hh + 2*Rx*Rz)
        avg_absorption = self.scene.surf_absorption[0].mean()
        return 0.161 * volume / (area * avg_absorption.clamp(min=0.01)).item()
    
    def compute_rir_fast(self, source_pos, receiver_pos):
        """
        快速计算 RIR（简化版，用于实时）
        
        只计算直接声 + 前几阶反射，牺牲精度换取速度
        """
        rir = torch.zeros(self.max_samples)
        
        Rx, Hh, Rz = self.scene.Rx[0], self.scene.Hh[0], self.scene.Rz[0]
        
        # 直接声
        dist = (source_pos - receiver_pos).norm()
        t = dist / self.c
        sample_idx = int(t * self.sample_rate)
        if sample_idx < self.max_samples:
            rir[sample_idx] = 1.0 / (dist ** 2 + 0.01)
        
        # 一阶反射（6 个墙面）
        walls = [
            ('x', -Rx, 0), ('x', Rx, 1),
            ('y', 0, 2), ('y', Hh, 3),
            ('z', -Rz, 4), ('z', Rz, 5)
        ]
        
        for axis, wall_pos, surf_idx in walls:
            # 镜像源位置
            if axis == 'x':
                img_pos = torch.tensor([-source_pos[0], source_pos[1], source_pos[2]])
            elif axis == 'y':
                img_pos = torch.tensor([source_pos[0], 2*wall_pos - source_pos[1], source_pos[2]])
            else:
                img_pos = torch.tensor([source_pos[0], source_pos[1], -source_pos[2]])
            
            dist = (img_pos - receiver_pos).norm()
            t = dist / self.c
            sample_idx = int(t * self.sample_rate)
            
            if sample_idx < self.max_samples:
                wall_abs = self.scene.surf_absorption[0, surf_idx]
                rir[sample_idx] += (1 - wall_abs) / (dist ** 2 + 0.01)
        
        # 混响尾
        t_array = torch.arange(self.max_samples) / self.sample_rate
        decay = torch.exp(-3 * t_array / (self.t60 / 3))
        rir = rir * decay
        
        # 归一化
        if rir.abs().max() > 0:
            rir = rir / rir.abs().max()
        
        return rir
    
    def compute_audio_from_rir(self, rir, audio_length=3.0):
        """将 RIR 转换为可播放音频（用短脉冲激励）"""
        num_samples = int(audio_length * self.sample_rate)
        
        # 用短脉冲作为激励（模拟脚步声/点击声）
        excitation = np.zeros(num_samples)
        excitation[0] = 1.0  # 在开头放一个脉冲
        
        # 卷积
        audio = signal.convolve(excitation, rir.numpy(), mode='full')
        audio = audio[:num_samples]
        
        # 归一化
        audio = audio / np.abs(audio).max() * 0.8 if np.abs(audio).max() > 0 else audio
        
        return audio


# ============================================================================
# 3. 视觉渲染器
# ============================================================================

class SimpleVisualRenderer:
    """简化的视觉渲染器 - 用颜色/深度表示位置"""
    
    def __init__(self, scene, res=128):
        self.scene = scene
        self.res = res
    
    def render(self, camera_pos, camera_yaw=0):
        """
        渲染当前视角的图像
        
        Args:
            camera_pos: 相机位置 [x, y, z]
            camera_yaw: 相机朝向（弧度）
        
        Returns:
            rgb: RGB 图像 [3, H, W]
            depth: 深度图 [H, W]
        """
        rgb = torch.zeros(3, self.res, self.res)
        depth = torch.zeros(self.res, self.res)
        
        Rx, Hh, Rz = self.scene.Rx[0], self.scene.Hh[0], self.scene.Rz[0]
        
        # 根据相机位置生成视口
        for y in range(self.res):
            for x in range(self.res):
                # 归一化坐标 [-1, 1]
                u = (x / self.res) * 2 - 1
                v = (y / self.res) * 2 - 1
                
                # 视场角 90 度
                ray_dir = torch.tensor([
                    math.tan(camera_yaw) + u * 0.5,
                    -v * 0.5,
                    1.0
                ])
                ray_dir = ray_dir / ray_dir.norm()
                
                # 与墙面相交（简化版）
                min_dist = float('inf')
                hit_color = torch.tensor([0.3, 0.3, 0.3])
                
                # 检查 6 个墙面
                for axis_idx, (axis, lo, hi) in enumerate([('x', -Rx, Rx), ('z', -Rz, Rz)]):
                    if abs(ray_dir[axis_idx]) > 1e-6:
                        t = (lo - camera_pos[axis_idx]) / ray_dir[axis_idx]
                        if t > 0.1 and t < min_dist:
                            min_dist = t
                            # 根据墙面索引给不同颜色
                            colors = [
                                torch.tensor([0.8, 0.6, 0.4]),  # 左墙
                                torch.tensor([0.6, 0.8, 0.4]),  # 右墙
                                torch.tensor([0.4, 0.6, 0.8]),  # 前墙
                                torch.tensor([0.8, 0.4, 0.6]),  # 后墙
                            ]
                            hit_color = colors[axis_idx % len(colors)]
                
                depth[y, x] = min_dist
                rgb[:, y, x] = hit_color
        
        return rgb, depth


# ============================================================================
# 4. 视频合成器
# ============================================================================

class AudioVideoSynthesizer:
    """音频视频同步合成器"""
    
    def __init__(self, fps=30, audio_sample_rate=16000):
        self.fps = fps
        self.audio_sample_rate = audio_sample_rate
        self.audio_per_frame = audio_sample_rate // fps  # 每帧的音频采样数
    
    def synthesize(self, frames, rirs, output_path):
        """
        合成视听视频
        
        Args:
            frames: 图像列表 [(H, W, 3), ...]
            rirs: RIR 列表 [T, ...]
            output_path: 输出文件路径
        """
        num_frames = len(frames)
        
        # 1. 合并所有音频
        total_audio_samples = num_frames * self.audio_per_frame
        full_audio = np.zeros(total_audio_samples)
        
        for i, rir in enumerate(rirs):
            start = i * self.audio_per_frame
            end = min(start + len(rir), total_audio_samples)
            audio_segment = self._rir_to_audio(rir)
            audio_len = min(len(audio_segment), end - start)
            full_audio[start:start + audio_len] += audio_segment[:audio_len]
        
        # 2. 归一化音频
        if np.abs(full_audio).max() > 0:
            full_audio = full_audio / np.abs(full_audio).max() * 0.8
        
        # 3. 保存为 WAV
        audio_path = Path(output_path).with_suffix('.wav')
        wavfile.write(audio_path, self.audio_sample_rate, (full_audio * 32767).astype(np.int16))
        
        # 4. 保存图像序列
        frame_dir = Path(output_path).parent / f"{Path(output_path).stem}_frames"
        frame_dir.mkdir(exist_ok=True)
        
        for i, frame in enumerate(frames):
            img_path = frame_dir / f"frame_{i:04d}.png"
            Image.fromarray(frame).save(img_path)
        
        # 5. 生成 FFmpeg 命令
        ffmpeg_cmd = f'''ffmpeg -framerate {self.fps} -i "{frame_dir}/frame_%04d.png" -i "{audio_path}" -c:v libx264 -pix_fmt yuv420p -c:a aac -strict experimental -shortest "{output_path}"'''
        
        print(f"\n图像序列已保存到: {frame_dir}")
        print(f"音频已保存到: {audio_path}")
        print(f"\n运行以下命令合成视频:")
        print(f"  {ffmpeg_cmd}")
        
        return ffmpeg_cmd
    
    def _rir_to_audio(self, rir):
        """将 RIR 转换为音频段"""
        num_samples = self.audio_per_frame
        excitation = np.zeros(num_samples)
        excitation[0] = 1.0
        
        audio = signal.convolve(excitation, rir.numpy(), mode='full')
        return audio[:num_samples]


# ============================================================================
# 5. 交互式探索（命令行版本）
# ============================================================================

class InteractiveExplorer:
    """交互式探索器 - WASD 控制"""
    
    def __init__(self, scene, res=128):
        self.scene = scene
        self.renderer = SimpleVisualRenderer(scene, res)
        self.acoustics = RealtimeAcoustics(scene)
        
        # 初始位置（房间中心）
        bounds = scene.get_room_bounds()
        self.camera_pos = torch.tensor([0.0, 1.5, 0.0])  # x, y, z
        self.camera_yaw = 0.0  # 弧度
        
        # 移动参数
        self.move_speed = 0.05
        self.turn_speed = 0.03
        
        # 记录轨迹
        self.trajectory = [self.camera_pos.clone()]
        self.yaws = [self.camera_yaw]
    
    def move(self, dx=0, dz=0, dyaw=0):
        """移动相机"""
        # 根据朝向计算移动方向
        new_x = self.camera_pos[0] + dx * math.cos(self.camera_yaw) - dz * math.sin(self.camera_yaw)
        new_z = self.camera_pos[2] + dx * math.sin(self.camera_yaw) + dz * math.cos(self.camera_yaw)
        
        # 边界检查
        bounds = self.scene.get_room_bounds()
        new_x = max(bounds['x'][0] + 0.3, min(bounds['x'][1] - 0.3, new_x))
        new_z = max(bounds['z'][0] + 0.3, min(bounds['z'][1] - 0.3, new_z))
        
        self.camera_pos[0] = new_x
        self.camera_pos[2] = new_z
        self.camera_yaw += dyaw
        
        self.trajectory.append(self.camera_pos.clone())
        self.yaws.append(self.camera_yaw)
    
    def get_frame(self):
        """获取当前帧"""
        rgb, depth = self.renderer.render(self.camera_pos, self.camera_yaw)
        return rgb, depth
    
    def get_rir(self):
        """获取当前位置的 RIR"""
        # 声源和接收器都在相机位置（第一人称）
        return self.acoustics.compute_rir_fast(self.camera_pos, self.camera_pos)


def demo_walkthrough(num_steps=60, save_path="walkthrough.mp4"):
    """
    演示：自动行走并记录视听数据
    
    Args:
        num_steps: 行走步数
        save_path: 输出视频路径
    """
    print("=" * 60)
    print("视听探索演示 - 自动行走")
    print("=" * 60)
    
    # 1. 初始化场景
    scene = SimpleScene(batch_size=1)
    explorer = InteractiveExplorer(scene, res=128)
    
    print(f"\n房间尺寸：{scene.get_room_bounds()}")
    print(f"混响时间 T60: {explorer.acoustics.t60:.2f} 秒")
    
    # 2. 记录帧和 RIR
    frames = []
    rirs = []
    
    print("\n开始记录...")
    start_time = time.time()
    
    for step in range(num_steps):
        # 移动（绕圈）
        explorer.move(dx=0.03, dyaw=0.05)
        
        # 获取帧
        rgb, depth = explorer.get_frame()
        frame = ((rgb.permute(1, 2, 0) + 1) / 2 * 255).clamp(0, 255).numpy().astype(np.uint8)
        frames.append(frame)
        
        # 获取 RIR
        rir = explorer.get_rir()
        rirs.append(rir)
        
        # 进度
        if (step + 1) % 10 == 0:
            print(f"  步骤 {step + 1}/{num_steps}, 位置：{explorer.camera_pos.tolist()}")
    
    elapsed = time.time() - start_time
    print(f"\n记录完成，耗时 {elapsed:.2f} 秒")
    print(f"平均帧率: {num_steps / elapsed:.1f} fps")
    
    # 3. 合成视频
    print("\n合成视听视频...")
    synthesizer = AudioVideoSynthesizer(fps=30)
    ffmpeg_cmd = synthesizer.synthesize(frames, rirs, save_path)
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
    print(f"\n下一步:")
    print(f"1. 安装 FFmpeg (如果未安装)")
    print(f"2. 运行上面的 FFmpeg 命令合成视频")
    print(f"3. 查看 {save_path}")
    
    return frames, rirs


if __name__ == "__main__":
    # 运行演示
    frames, rirs = demo_walkthrough(num_steps=60, save_path="walkthrough.mp4")
    
    # 也可以单独保存 RIR 和音频
    output_dir = Path("audio_demo")
    output_dir.mkdir(exist_ok=True)
    
    print("\n保存单独的音频文件...")
    for i, rir in enumerate(rirs[::10]):  # 每 10 帧保存一个
        audio = rir.numpy()
        wavfile.write(output_dir / f"step_{i*10}.wav", 16000, (audio * 32767).astype(np.int16))
    
    print(f"音频已保存到: {output_dir}")
