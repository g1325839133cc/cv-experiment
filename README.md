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
