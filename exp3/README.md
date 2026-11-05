# 计算机视觉实验三：图像拼接

基于 **SIFT 特征 + FLANN 匹配 + RANSAC 单应矩阵估计**，将两张具有重叠区域的瀑布图拼接为一张全景图。

## 效果预览

| 输入左图 | 输入右图 | 拼接结果 |
|:---:|:---:|:---:|
| ![left](images/left.png) | ![right](images/right.png) | ![stitched](images/stitched_result.png) |

## 目录结构

```
.
├── image_stitching.py     # 主程序（完整可运行）
├── left.png               # 输入：左视图
├── right.png              # 输入：右视图
├── matched_keypoints.png  # SIFT 特征匹配可视化
├── stitched_result.png    # 拼接后全景图
├── 实验三.md              # 完整实验报告（原理 / 代码 / 结果分析）
└── README.md
```

## 环境依赖

```bash
pip install opencv-python numpy
```

## 运行

```bash
python image_stitching.py
```

运行后会在当前目录生成 `matched_keypoints.png` 与 `stitched_result.png`。

## 方法流程

1. SIFT 检测两图关键点并计算 128 维描述子；
2. FLANN 近似最近邻匹配 + Lowe 比率测试（阈值 0.7）筛选优质匹配；
3. RANSAC 估计将右图投影到左图坐标系的单应矩阵 H；
4. 计算外接矩形并加入平移，`cv2.warpPerspective` 变换右图；
5. 将左图贴回对应位置，输出全景图。

详细原理、代码注释与结果分析见 [实验三.md](实验三.md)。
