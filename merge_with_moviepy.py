"""用 moviepy 合并视频和音频"""
from moviepy import VideoFileClip, AudioFileClip
from pathlib import Path

def merge_audio_video(video_path, audio_path, output_path):
    """使用 moviepy 合并视频和音频"""
    
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
    
    try:
        # 加载视频
        print("\n加载视频...")
        video = VideoFileClip(str(video_file))
        
        # 加载音频
        print("加载音频...")
        audio = AudioFileClip(str(audio_file))
        
        # 设置音频时长与视频匹配
        audio = audio.with_duration(video.duration)
        
        # 设置音频到视频
        print("合并...")
        video = video.with_audio(audio)
        
        # 写入输出
        print(f"写入：{output_file}")
        video.write_videofile(
            output_file,
            codec='libx264',
            audio_codec='aac',
            fps=30,
            temp_audiofile='temp-audio.m4a',
            remove_temp=True
        )
        
        # 关闭
        video.close()
        
        # 检查文件大小
        size = output_file.stat().st_size
        print(f"\n合并成功!")
        print(f"文件大小：{size / 1024:.1f} KB ({size / 1024 / 1024:.2f} MB)")
        
        return output_file
        
    except Exception as e:
        print(f"\n错误：{e}")
        return None


if __name__ == "__main__":
    video_path = "walkthrough_footsteps_h264.mp4"
    audio_path = "walkthrough_footsteps.wav"
    output_path = "walkthrough_footsteps_final.mp4"
    
    merge_audio_video(video_path, audio_path, output_path)
