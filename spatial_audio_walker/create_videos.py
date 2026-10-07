"""为改进版数据集创建视频"""
import math
import numpy as np
import json
from pathlib import Path
import imageio.v3 as iio
from moviepy import VideoFileClip, AudioFileClip
from PIL import Image


def create_video_for_clip(clip_dir):
    """为单个片段创建视频"""
    clip_dir = Path(clip_dir)
    
    # 读取片段信息
    with open(clip_dir / "clip_info.json") as f:
        clip_info = json.load(f)
    
    num_frames = clip_info['num_frames']
    duration = clip_info['duration']
    fps = 30
    
    print(f"\n创建视频：{clip_dir.name}")
    print(f"  帧数：{num_frames}")
    print(f"  时长：{duration:.1f}秒")
    
    # 读取帧（使用 PIL）
    frame_dir = clip_dir / "frames"
    frames = []
    for i in range(num_frames):
        frame_path = frame_dir / f"frame_{i:04d}.png"
        if frame_path.exists():
            try:
                img = Image.open(frame_path)
                frame = np.array(img)
                frames.append(frame)
            except Exception as e:
                print(f"  警告：无法读取帧 {i:04d}: {e}")
        else:
            print(f"  警告：缺少帧 {i:04d}")
    
    if len(frames) == 0:
        print(f"  跳过：没有帧")
        return None
    
    print(f"  加载了 {len(frames)} 帧")
    
    if len(frames) == 0:
        print(f"  跳过：没有帧")
        return None
    
    print(f"  加载了 {len(frames)} 帧")
    
    # 合成视频
    frames_array = np.stack(frames, axis=0)
    video_path = clip_dir / "video.mp4"
    
    writer = iio.imopen(video_path, "w", plugin="FFMPEG")
    writer.write(frames_array, fps=fps, codec='libx264')
    writer.close()
    
    print(f"  视频已保存：{video_path}")
    
    # 合并音频
    audio_path = clip_dir / "audio.wav"
    if audio_path.exists():
        print(f"  合并音频...")
        final_path = clip_dir / "final.mp4"
        
        video = VideoFileClip(str(video_path))
        audio = AudioFileClip(str(audio_path))
        audio = audio.with_duration(video.duration)
        video = video.with_audio(audio)
        video.write_videofile(
            final_path,
            codec='libx264',
            audio_codec='aac',
            fps=fps,
            temp_audiofile='temp-audio.m4a',
            remove_temp=True
        )
        video.close()
        
        print(f"  完成：{final_path}")
        return final_path
    else:
        print(f"  警告：音频文件不存在")
        return video_path


def create_all_videos(dataset_dir):
    """为所有片段创建视频"""
    dataset_dir = Path(dataset_dir)
    
    # 读取数据集索引
    with open(dataset_dir / "dataset_index.json") as f:
        index = json.load(f)
    
    print("=" * 60)
    print(f"创建视频 - {index['total_clips']} 个片段")
    print("=" * 60)
    
    videos = []
    for clip_info in index['clips']:
        clip_id = clip_info['clip_id']
        clip_dir = dataset_dir / f"clip_{clip_id:02d}"
        
        if clip_dir.exists():
            video_path = create_video_for_clip(clip_dir)
            if video_path:
                videos.append({
                    'clip_id': clip_id,
                    'video_path': str(video_path),
                    'duration': clip_info['duration']
                })
    
    print("\n" + "=" * 60)
    print("视频创建完成！")
    print("=" * 60)
    print(f"\n总计：{len(videos)} 个视频")
    
    # 保存视频索引
    video_index = {
        'total_videos': len(videos),
        'videos': videos
    }
    with open(dataset_dir / "video_index.json", 'w') as f:
        json.dump(video_index, f, indent=2)
    
    return videos


if __name__ == "__main__":
    dataset_dir = Path(__file__).parent / "improved_dataset"
    videos = create_all_videos(dataset_dir)
    print(f"\n所有视频已保存到：{dataset_dir}")
