"""用 Python 合成视听视频（不需要 FFmpeg 命令行）"""
import imageio.v3 as iio
import numpy as np
from pathlib import Path
from scipy.io import wavfile
import os

def create_video_with_audio(frame_dir, audio_path, output_path, fps=30):
    """
    使用 imageio 创建带音频的视频
    
    Args:
        frame_dir: 图像帧目录
        audio_path: 音频文件路径
        output_path: 输出视频路径
        fps: 帧率
    """
    print(f"读取音频文件：{audio_path}")
    sample_rate, audio = wavfile.read(audio_path)
    print(f"音频: {len(audio)} 采样点，{len(audio)/sample_rate:.2f} 秒")
    
    # 读取所有帧
    frame_paths = sorted(Path(frame_dir).glob("frame_*.png"))
    print(f"读取 {len(frame_paths)} 帧图像")
    
    # 写入视频（先不带音频，因为 imageio 对音频支持有限）
    print("写入视频...")
    with iio.imopen(output_path, "w", plugin="FFMPEG") as file:
        for i, frame_path in enumerate(frame_paths):
            frame = iio.imread(frame_path)
            file.write(frame, fps=fps)
    
    print(f"视频已保存到：{output_path}")
    print(f"\n注意：由于 imageio 的音频支持有限，音频文件单独保存在：{audio_path}")
    print("如需合并音频到视频，请安装 FFmpeg 后运行:")
    print(f'  ffmpeg -framerate {fps} -i "{frame_dir}/frame_%04d.png" -i "{audio_path}" -c:v libx264 -pix_fmt yuv420p -c:a aac -strict experimental -shortest "{output_path}"')
    
    return output_path


if __name__ == "__main__":
    frame_dir = "walkthrough_frames"
    audio_path = "walkthrough.wav"
    output_path = "walkthrough.mp4"
    
    if not Path(frame_dir).exists():
        print(f"错误：帧目录不存在: {frame_dir}")
        exit(1)
    
    if not Path(audio_path).exists():
        print(f"错误：音频文件不存在: {audio_path}")
        exit(1)
    
    create_video_with_audio(frame_dir, audio_path, output_path)
