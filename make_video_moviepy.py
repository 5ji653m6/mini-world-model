"""用 moviepy 创建视频"""
from moviepy import VideoClip, ImageClip
from moviepy.video.tools.interpolators import Trajectory
import numpy as np
from pathlib import Path
from PIL import Image
import os

def create_video_with_moviepy(frame_dir, output_path, fps=30):
    """使用 moviepy 创建视频"""
    
    # 获取所有帧
    frame_paths = sorted(Path(frame_dir).glob("frame_*.png"))
    print(f"找到 {len(frame_paths)} 帧图像")
    
    if len(frame_paths) == 0:
        print("错误：没有找到图像帧")
        return None
    
    # 读取第一帧获取尺寸
    first_frame = np.array(Image.open(frame_paths[0]))
    h, w = first_frame.shape[:2]
    print(f"视频尺寸：{w}x{h}")
    
    # 创建图像序列剪辑
    print("创建视频剪辑...")
    
    def make_frame(t):
        # 计算帧索引
        frame_idx = int(t * fps)
        if frame_idx >= len(frame_paths):
            frame_idx = len(frame_paths) - 1
        frame = np.array(Image.open(frame_paths[frame_idx]))
        return frame
    
    # 创建视频
    duration = len(frame_paths) / fps
    print(f"视频时长：{duration:.2f} 秒")
    
    video = VideoClip(make_frame, duration=duration).with_fps(fps)
    
    # 写入视频
    print(f"写入视频：{output_path}")
    video.write_videofile(
        output_path,
        fps=fps,
        codec='libx264',
        audio=False,
        preset='ultrafast',
        threads=4,
        temp_audiofile=False
    )
    
    # 检查文件大小
    size = Path(output_path).stat().st_size
    print(f"文件大小：{size / 1024:.1f} KB")
    
    return output_path


if __name__ == "__main__":
    frame_dir = "long_walkthrough_frames"
    output_path = "long_walkthrough_moviepy.mp4"
    
    if not Path(frame_dir).exists():
        print(f"错误：帧目录不存在：{frame_dir}")
        exit(1)
    
    create_video_with_moviepy(frame_dir, output_path, fps=30)
