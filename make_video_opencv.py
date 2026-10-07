"""用 OpenCV 创建视频"""
import cv2
import numpy as np
from pathlib import Path

def create_video_with_opencv(frame_dir, output_path, fps=30):
    """使用 OpenCV 创建视频"""
    
    # 获取所有帧
    frame_paths = sorted(Path(frame_dir).glob("frame_*.png"))
    print(f"找到 {len(frame_paths)} 帧图像")
    
    if len(frame_paths) == 0:
        print("错误：没有找到图像帧")
        return None
    
    # 读取第一帧获取尺寸
    first_frame = cv2.imread(str(frame_paths[0]))
    if first_frame is None:
        print("错误：无法读取第一帧")
        return None
    
    h, w = first_frame.shape[:2]
    print(f"视频尺寸：{w}x{h}")
    
    # 创建 VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    if not out.isOpened():
        print("错误：无法打开 VideoWriter")
        return None
    
    print(f"写入视频：{output_path}")
    print(f"时长：{len(frame_paths) / fps:.2f} 秒")
    
    # 写入所有帧
    for i, frame_path in enumerate(frame_paths):
        frame = cv2.imread(str(frame_path))
        if frame is not None:
            out.write(frame)
            if (i + 1) % 100 == 0:
                print(f"  已写入 {i + 1}/{len(frame_paths)} 帧")
    
    out.release()
    print("视频完成")
    
    # 检查文件大小
    size = Path(output_path).stat().st_size
    print(f"文件大小：{size / 1024:.1f} KB")
    
    return output_path


if __name__ == "__main__":
    frame_dir = "long_walkthrough_frames"
    output_path = "long_walkthrough_opencv.mp4"
    
    if not Path(frame_dir).exists():
        print(f"错误：帧目录不存在：{frame_dir}")
        exit(1)
    
    create_video_with_opencv(frame_dir, output_path, fps=30)
