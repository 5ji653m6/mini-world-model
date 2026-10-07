"""合并视频和音频"""
import subprocess
from pathlib import Path

def merge_audio_video(video_path, audio_path, output_path):
    """使用 FFmpeg 合并视频和音频"""
    
    video_file = Path(video_path)
    audio_file = Path(audio_path)
    output_file = Path(output_path)
    
    if not video_file.exists():
        print(f"错误：视频文件不存在：{video_file}")
        return None
    
    if not audio_file.exists():
        print(f"错误：音频文件不存在：{audio_file}")
        return None
    
    print(f"合并:")
    print(f"  视频：{video_file}")
    print(f"  音频：{audio_file}")
    print(f"输出：{output_file}")
    
    # FFmpeg 命令
    cmd = [
        'ffmpeg',
        '-y',  # 覆盖输出
        '-i', str(video_file),
        '-i', str(audio_file),
        '-c:v', 'copy',  # 视频流直接复制
        '-c:a', 'aac',   # 音频编码为 AAC
        '-shortest',     # 以最短的流结束
        str(output_file)
    ]
    
    print(f"\n运行命令:")
    print(f"  {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("\n合并成功!")
            size = output_file.stat().st_size
            print(f"文件大小：{size / 1024:.1f} KB")
            return output_file
        else:
            print(f"\n错误：{result.stderr}")
            return None
            
    except FileNotFoundError:
        print("\n错误：FFmpeg 未安装！")
        print("请运行: winget install ffmpeg")
        return None
    except subprocess.TimeoutExpired:
        print("\n错误：合并超时！")
        return None


if __name__ == "__main__":
    video_path = "walkthrough_footsteps_h264.mp4"
    audio_path = "walkthrough_footsteps.wav"
    output_path = "walkthrough_footsteps_final.mp4"
    
    merge_audio_video(video_path, audio_path, output_path)
