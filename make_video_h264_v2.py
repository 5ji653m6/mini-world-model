"""用 imageio 生成 H.264 编码的视频"""
import imageio.v3 as iio
import numpy as np
from pathlib import Path

def create_video_h264(frame_dir, output_path, fps=30):
    """使用 imageio 的 FFMPEG 插件生成 H.264 视频"""
    
    # 获取所有帧
    frame_paths = sorted(Path(frame_dir).glob("frame_*.png"))
    print(f"找到 {len(frame_paths)} 帧图像")
    
    if len(frame_paths) == 0:
        print("错误：没有找到图像帧")
        return None
    
    # 使用 imageio 的 FFMPEG 写入器（v3 API）
    print(f"写入视频：{output_path}")
    print(f"编码：H.264")
    print(f"时长：{len(frame_paths) / fps:.2f} 秒")
    
    # 直接写入所有帧
    frames = [iio.imread(fp) for fp in frame_paths]
    frames_array = np.stack(frames, axis=0)  # [N, H, W, C]
    
    writer = iio.imopen(output_path, "w", plugin="FFMPEG")
    writer.write(frames_array, fps=fps, codec='libx264')
    writer.close()
    print("视频完成")
    
    # 检查文件大小
    size = Path(output_path).stat().st_size
    print(f"文件大小：{size / 1024:.1f} KB ({size / 1024 / 1024:.2f} MB)")
    
    return output_path


if __name__ == "__main__":
    frame_dir = "walkthrough_real_frames"
    output_path = "walkthrough_real_h264.mp4"
    
    if not Path(frame_dir).exists():
        print(f"错误：帧目录不存在：{frame_dir}")
        exit(1)
    
    create_video_h264(frame_dir, output_path, fps=30)
