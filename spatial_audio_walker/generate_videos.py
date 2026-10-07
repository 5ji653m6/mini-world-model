"""
为已有的数据集生成视频文件
"""
import cv2
import numpy as np
import json
from pathlib import Path
from moviepy import VideoFileClip, AudioFileClip


def generate_video_for_clip(clip_dir, resolution=64):
    """为单个片段生成视频"""
    clip_dir = Path(clip_dir)
    
    # 读取片段信息
    with open(clip_dir / "clip_info.json") as f:
        clip_info = json.load(f)
    
    num_frames = clip_info['num_frames']
    duration = clip_info['duration']
    fps = 30
    
    print(f"\n生成视频：{clip_dir.name}")
    print(f"  帧数：{num_frames}")
    print(f"  时长：{duration:.1f}秒")
    
    # 生成帧
    frames = []
    for step in range(num_frames):
        # 从 acoustic_data 中读取位置信息
        t = step / fps
        
        # 读取声学数据
        acoustic_file = clip_dir / "acoustic_data.json"
        if not acoustic_file.exists():
            print(f"  跳过：无声学数据")
            return None
        
        with open(acoustic_file) as f:
            acoustic_data = json.load(f)
        
        if len(acoustic_data) == 0:
            print(f"  跳过：声学数据为空")
            return None
        
        # 找到最近的帧数据
        closest_frame = min(acoustic_data, key=lambda x: abs(x['time'] - t))
        pos = closest_frame['camera_position']
        yaw = closest_frame['yaw']
        pitch = closest_frame['pitch']
        
        # 读取房间信息
        with open(clip_dir / "room_info.json") as f:
            room_info = json.load(f)
        
        room = room_info['room']
        
        # 渲染帧
        img = np.zeros((resolution, resolution, 3), dtype=np.uint8)
        
        # 地板（棋盘格）
        for y in range(resolution // 2, resolution):
            for x in range(resolution):
                u = (x / resolution + pos[0] / room['width']) * 10
                v = ((y - resolution // 2) / (resolution // 2) + pos[2] / room['depth']) * 10
                
                if (int(u) + int(v)) % 2 == 0:
                    img[y, x] = [80, 120, 80]
                else:
                    img[y, x] = [60, 100, 60]
        
        # 墙壁
        img[:resolution // 2, :] = [139, 119, 101]
        
        # 添加文字
        cv2.putText(img, f'X:{pos[0]:.2f}', (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(img, f'Z:{pos[2]:.2f}', (5, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        frames.append(img)
    
    # 保存帧
    frame_dir = clip_dir / "frames"
    frame_dir.mkdir(exist_ok=True)
    
    print(f"  保存帧到：{frame_dir}")
    for i, frame in enumerate(frames):
        cv2.imwrite(str(frame_dir / f"frame_{i:04d}.png"), frame)
    
    # 合成视频
    print(f"  合成视频...")
    
    # 使用 imageio 合成视频
    import imageio.v3 as iio
    
    frames_array = np.stack(frames, axis=0)
    video_path = clip_dir / "video.mp4"
    
    writer = iio.imopen(video_path, "w", plugin="FFMPEG")
    writer.write(frames_array, fps=fps, codec='libx264')
    writer.close()
    
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


def generate_all_videos(dataset_dir):
    """为所有片段生成视频"""
    dataset_dir = Path(dataset_dir)
    
    # 读取数据集索引
    with open(dataset_dir / "dataset_index.json") as f:
        index = json.load(f)
    
    print("=" * 60)
    print(f"生成视频 - {index['total_clips']} 个片段")
    print("=" * 60)
    
    videos = []
    for clip_info in index['clips']:
        clip_id = clip_info['clip_id']
        clip_dir = dataset_dir / f"clip_{clip_id:02d}"
        
        if clip_dir.exists():
            video_path = generate_video_for_clip(clip_dir)
            videos.append({
                'clip_id': clip_id,
                'video_path': str(video_path),
                'duration': clip_info['duration']
            })
    
    print("\n" + "=" * 60)
    print("视频生成完成！")
    print("=" * 60)
    print(f"\n总计：{len(videos)} 个视频")
    
    # 保存视频索引
    video_index = {
        'total_videos': len(videos),
        'videos': videos
    }
    with open(dataset_dir / "video_index.json", 'w') as f:
        json.dump(video_index, f, indent=2)
    
    print(f"\n视频索引：{dataset_dir / 'video_index.json'}")
    
    return videos


if __name__ == "__main__":
    dataset_dir = Path(__file__).parent.parent / "spatial_audio_dataset"
    videos = generate_all_videos(dataset_dir)
    
    print(f"\n所有视频已保存到：{dataset_dir}")
