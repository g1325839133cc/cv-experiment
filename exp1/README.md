# 实验一：计算机视觉库的安装

## 实验内容
掌握 Anaconda 的基础操作，完成 GPU 加速环境、OpenCV、PyTorch 的安装与验证：
1. Anaconda 安装，查看 conda 版本与配置
2. 创建 Python3.10 虚拟环境，安装 OpenCV
3. 查看显卡、CUDA 版本信息
4. 安装匹配 CUDA 的 PyTorch
5. 代码验证 GPU 与 cuDNN 是否可用

## 运行环境
- Python 3.10
- Anaconda
- opencv-python
- torch (带CUDA支持)

## 运行方法
```bash
# 创建虚拟环境
conda create -n cv python==3.10
conda activate cv

# 安装依赖
pip install opencv-python torch torchvision
```
```python
# GPU验证代码
import torch
print(torch.cuda.is_available())
print(torch.backends.cudnn.is_available())
```

## 目录结构
```
├── 实验一_计算机视觉库安装_实验报告.md   # 实验报告
└── images/                               # 截图存放目录
```

## 结果要点
- 独立虚拟环境可以解决不同项目包版本冲突
- `nvidia-smi` 查看驱动支持的最高CUDA版本
- 输出两个True代表GPU加速环境配置成功
```

