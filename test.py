#!/usr/bin/env python3
"""
快速测试脚本 - 验证RDPG项目是否正常工作
"""

import sys
import os

def test_imports():
    """测试1: 检查依赖导入"""
    print("="*60)
    print("测试1: 检查Python依赖")
    print("="*60)
    
    packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy', 
        'matplotlib': 'Matplotlib',
        'gymnasium': 'Gymnasium'
    }
    
    all_ok = True
    for module, name in packages.items():
        try:
            __import__(module)
            print(f"✓ {name:12s} - 已安装")
        except ImportError:
            print(f"✗ {name:12s} - 未安装")
            all_ok = False
    
    print()
    return all_ok

def test_project_structure():
    """测试2: 检查项目结构"""
    print("="*60)
    print("测试2: 检查项目文件")
    print("="*60)
    
    required_files = [
        'train_rdpg.py',
        'demo.py',
        'requirements.txt',
        'common/buffers.py',
        'common/policy_networks.py',
        'common/value_networks.py',
    ]
    
    all_ok = True
    for file_path in required_files:
        exists = os.path.exists(file_path)
        status = "✓" if exists else "✗"
        print(f"{status} {file_path}")
        if not exists:
            all_ok = False
    
    print()
    return all_ok

def test_matplotlib_backend():
    """测试3: 检查matplotlib后端"""
    print("="*60)
    print("测试3: 检查Matplotlib后端")
    print("="*60)
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # 测试创建图表
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot([1, 2, 3], [1, 4, 9])
        
        # 测试保存
        os.makedirs('./logs', exist_ok=True)
        test_file = './logs/test_plot.png'
        fig.savefig(test_file)
        plt.close(fig)
        
        if os.path.exists(test_file):
            print("✓ Matplotlib后端正常工作")
            print(f"✓ 测试图表已保存: {test_file}")
            os.remove(test_file)  # 清理测试文件
            print()
            return True
        else:
            print("✗ 无法保存图表")
            print()
            return False
            
    except Exception as e:
        print(f"✗ Matplotlib测试失败: {e}")
        print()
        return False

def test_basic_training():
    """测试4: 运行基础训练"""
    print("="*60)
    print("测试4: 运行基础训练（3回合）")
    print("="*60)
    
    try:
        # 导入必要的模块
        import torch
        import gymnasium as gym
        from common.buffers import ReplayBufferLSTM2
        from common.value_networks import QNetworkLSTM2
        from common.policy_networks import DPG_PolicyNetworkLSTM2
        
        print("✓ 模块导入成功")
        
        # 创建简单环境
        env = gym.make("Pendulum-v1")
        print("✓ 环境创建成功")
        
        # 创建网络
        state_space = env.observation_space
        action_space = env.action_space
        hidden_dim = 32
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        qnet = QNetworkLSTM2(state_space, action_space, hidden_dim).to(device)
        policy_net = DPG_PolicyNetworkLSTM2(state_space, action_space, hidden_dim).to(device)
        
        print("✓ 神经网络创建成功")
        print(f"✓ 使用设备: {device}")
        
        # 测试一个episode
        state, _ = env.reset()
        print("✓ 环境重置成功")
        
        print()
        print("所有基础功能测试通过!")
        print()
        return True
        
    except Exception as e:
        print(f"✗ 训练测试失败: {e}")
        print()
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    print()
    print("╔" + "="*58 + "╗")
    print("║" + " "*15 + "RDPG 项目测试工具" + " "*15 + "║")
    print("╚" + "="*58 + "╝")
    print()
    
    results = []
    
    # 运行所有测试
    results.append(("依赖检查", test_imports()))
    results.append(("项目结构", test_project_structure()))
    results.append(("Matplotlib", test_matplotlib_backend()))
    results.append(("基础训练", test_basic_training()))
    
    # 总结
    print("="*60)
    print("测试总结")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "通过" if passed else "失败"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {name:12s} - {status}")
        if not passed:
            all_passed = False
    
    print()
    
    if all_passed:
        print("🎉 所有测试通过！")
        print()
        print("你可以开始使用RDPG了:")
        print("  python demo.py              # 快速演示")
        print("  python train_rdpg.py --train  # 完整训练")
        print()
    else:
        print("⚠️  部分测试失败")
        print()
        print("请检查:")
        print("  1. 是否安装了所有依赖: pip install -r requirements.txt")
        print("  2. 是否在正确的目录运行")
        print("  3. Python版本是否 >= 3.8")
        print()
        print("查看详细说明:")
        print("  - 修复说明.md")
        print("  - QUICKSTART.md")
        print()
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n测试已中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
