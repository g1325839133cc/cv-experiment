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


# 实验二：基于 OpenCV 的图像增强

## 实验内容

利用 OpenCV、scikit-image、NumPy、Matplotlib 等库对图像进行平滑、滤波，实现图像增强：

1. 读取原始图像，进行色彩空间转换（BGR→RGB、BGR→GRAY）
2. 添加椒盐噪声（Salt-and-Pepper）与高斯噪声（Gaussian）
3. 分别用均值滤波、中值滤波、高斯滤波进行去噪对比
4. 手动实现彩色图像的分通道中值滤波

## 运行环境

- Python 3.13
- opencv-python
- scikit-image
- numpy
- matplotlib

## 运行方法

```bash
# 安装依赖
pip install opencv-python scikit-image numpy matplotlib

# 运行实验
python experiment2_image_enhancement.py
```

## 目录结构

```
├── 实验二_基于OpenCV的图像增强_实验报告.md   # 实验报告
├── experiment2_image_enhancement.py          # 实验源代码
├── p1.jpg                                    # 输入图像（高山湖泊）
├── output_images/                            # 颜色空间转换、噪声对比结果图
└── result/                                   # 滤波结果图
```

## 结果要点

- 中值滤波对椒盐噪声去除效果最佳，且能保留边缘细节
- 均值滤波 / 高斯滤波更适合平滑高斯噪声
- 手动分通道中值滤波与 OpenCV 内置 `medianBlur` 结果一致
