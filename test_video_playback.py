"""测试视频文件是否有效"""
import cv2

def test_video(video_path):
    """测试视频文件"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"错误：无法打开视频文件：{video_path}")
        return False
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"视频信息:")
    print(f"  分辨率：{width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  总帧数：{total_frames}")
    print(f"  时长：{total_frames / fps:.2f} 秒")
    
    # 读取几帧检查
    print("\n读取测试帧:")
    for frame_idx in [0, total_frames//4, total_frames//2, 3*total_frames//4, total_frames-1]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            print(f"  帧 {frame_idx}: 成功，均值 {frame.mean():.2f}")
        else:
            print(f"  帧 {frame_idx}: 失败")
    
    cap.release()
    return True


if __name__ == "__main__":
    videos = [
        "long_walkthrough_opencv.mp4",
        "long_walkthrough_moviepy.mp4",
        "long_walkthrough_fixed.mp4",
        "long_walkthrough.mp4"
    ]
    
    for v in videos:
        print(f"\n{'='*40}")
        print(f"测试：{v}")
        print('='*40)
        test_video(v)
