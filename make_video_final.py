"""用 OpenCV 创建最终视频"""
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
    
    # 创建 VideoWriter（使用 MJPG 编码器，兼容性更好）
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
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
    print(f"文件大小：{size / 1024:.1f} KB ({size / 1024 / 1024:.2f} MB)")
    
    # 验证视频
    print("\n验证视频...")
    cap = cv2.VideoCapture(output_path)
    if cap.isOpened():
        actual_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"  实际帧数：{actual_frames}")
        print(f"  实际 FPS: {actual_fps}")
        print(f"  实际时长：{actual_frames / actual_fps:.2f} 秒")
        cap.release()
    
    return output_path


if __name__ == "__main__":
    frame_dir = "long_walkthrough_frames"
    output_path = "long_walkthrough_final.mp4"
    
    if not Path(frame_dir).exists():
        print(f"错误：帧目录不存在：{frame_dir}")
        exit(1)
    
    create_video_with_opencv(frame_dir, output_path, fps=30)
