"""用正确的 FFmpeg 方式合成视频"""
import imageio
import numpy as np
from pathlib import Path
from scipy.io import wavfile

def create_video_from_frames(frame_dir, output_path, fps=30):
    """使用 imageio-ffmpeg 正确创建视频"""
    
    # 获取所有帧
    frame_paths = sorted(Path(frame_dir).glob("frame_*.png"))
    print(f"找到 {len(frame_paths)} 帧图像")
    
    if len(frame_paths) == 0:
        print("错误：没有找到图像帧")
        return None
    
    # 使用 v2 的 get_writer 创建视频
    print(f"写入视频：{output_path}")
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264')
    
    for i, frame_path in enumerate(frame_paths):
        frame = imageio.imread(frame_path)
        writer.append_data(frame)
        if (i + 1) % 100 == 0:
            print(f"  已写入 {i + 1}/{len(frame_paths)} 帧")
    
    writer.close()
    print(f"视频完成：{output_path}")
    
    # 检查文件大小
    size = output_path.stat().st_size
    print(f"文件大小：{size / 1024:.1f} KB")
    
    return output_path


if __name__ == "__main__":
    frame_dir = "long_walkthrough_frames"
    output_path = "long_walkthrough_fixed.mp4"
    
    if not Path(frame_dir).exists():
        print(f"错误：帧目录不存在：{frame_dir}")
        exit(1)
    
    create_video_from_frames(frame_dir, output_path, fps=30)
