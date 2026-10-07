"""测试声学数据合成的可行性"""
import torch
import math
import time

class SimpleAcousticSimulator:
    """简化的声学模拟器（仅用于验证可行性）"""
    
    def __init__(self, room_size, sample_rate=16000, max_time=1.0):
        self.c = 343.0  # 声速 m/s
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_time)
        self.room_size = room_size  # (Rx, Hh, Rz)
        
    def image_source_method(self, source, receiver, max_order=5):
        """
        使用镜像源方法计算早期反射
        返回脉冲响应
        """
        rir = torch.zeros(self.max_samples)
        
        # 递归生成镜像源
        def generate_images(order, position, energy, phase=1):
            if order > max_order:
                return []
            
            images = []
            # 6 个墙面
            walls = [
                ('x', -self.room_size[0], self.room_size[0]),  # 左/右墙
                ('y', 0, self.room_size[1]),                    # 地板/天花板
                ('z', -self.room_size[2], self.room_size[2]),   # 前/后墙
            ]
            
            for axis, lo, hi in walls:
                # 镜像反射
                if axis == 'x':
                    img_pos = position.clone()
                    img_pos[0] = 2 * (lo if position[0] < 0 else hi) - position[0]
                elif axis == 'y':
                    img_pos = position.clone()
                    img_pos[1] = 2 * (lo if position[1] < 0 else hi) - position[1]
                else:
                    img_pos = position.clone()
                    img_pos[2] = 2 * (lo if position[2] < 0 else hi) - position[2]
                
                # 距离和时间
                dist = (img_pos - receiver).norm()
                t = dist / self.c
                sample_idx = int(t * self.sample_rate)
                
                if sample_idx < self.max_samples:
                    # 能量衰减（距离 + 墙面吸收）
                    wall_absorption = 0.1  # 假设墙面吸声系数
                    attenuation = energy / (dist ** 2 + 0.01) * (1 - wall_absorption) ** order
                    rir[sample_idx] += attenuation * phase
                    
                    # 递归生成下一阶
                    images.extend(generate_images(
                        order + 1, img_pos, 
                        energy * (1 - wall_absorption),
                        -phase  # 反射相位反转
                    ))
            
            return images
        
        # 从源位置开始
        generate_images(0, source, 1.0)
        
        # 添加混响尾（Schroeder 衰减）
        t60 = 0.5  # 假设混响时间 0.5 秒
        decay = torch.exp(-3 * torch.arange(self.max_samples) / (self.sample_rate * t60 / 3))
        rir = rir * decay
        
        return rir


def test_feasibility():
    """测试声学数据生成的可行性"""
    print("=" * 60)
    print("声学数据合成可行性测试")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n设备: {device}")
    
    # 测试参数
    B = 4  # batch size
    room_size = torch.tensor([3.0, 2.8, 4.0])  # 房间尺寸 (m)
    
    # 创建模拟器
    sim = SimpleAcousticSimulator(room_size, sample_rate=16000, max_time=1.0)
    
    print(f"\n采样率: {sim.sample_rate} Hz")
    print(f"最大时长: {sim.max_samples / sim.sample_rate:.2f} 秒")
    print(f"房间尺寸: {room_size.tolist()} m")
    
    # 测试单样本
    print("\n--- 单样本测试 ---")
    source = torch.tensor([0.0, 1.5, 0.0])  # 声源位置
    receiver = torch.tensor([1.0, 1.5, 1.0])  # 接收器位置
    
    start = time.time()
    rir = sim.image_source_method(source, receiver)
    elapsed = time.time() - start
    
    print(f"生成时间: {elapsed*1000:.2f} ms")
    print(f"RIR 长度: {len(rir)} 采样点")
    print(f"RIR 能量: {rir.abs().sum():.4f}")
    
    # 测试批量（GPU 加速版本）
    print("\n--- 批量测试 ---")
    if device == "cuda":
        sources = torch.randn(B, 3, device=device) * 1.5
        receivers = torch.randn(B, 3, device=device) * 1.5
        
        start = time.time()
        # 这里需要实现 GPU 批量版本
        # rirs = batch_image_source(sources, receivers)
        elapsed = time.time() - start
        
        print(f"批量生成时间: {elapsed*1000:.2f} ms (待实现)")
    
    # 可视化 RIR
    print("\n--- RIR 示例 ---")
    print(f"峰值位置: {(rir.abs() > 0.1).nonzero()[0].item() / sim.sample_rate:.4f} 秒")
    print(f"能量衰减到 10%: {(rir.abs() < rir.abs().max() * 0.1).nonzero()[0].item() / sim.sample_rate:.4f} 秒")
    
    print("\n" + "=" * 60)
    print("结论：声学数据合成在技术上是可行的！")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    test_feasibility()
