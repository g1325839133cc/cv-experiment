# -*- coding: utf-8 -*-
"""
实验三：基于 SIFT 特征的图像拼接
将 左.png 与 右.png 两张具有重叠区域的瀑布图拼接为一张全景图。
"""
import cv2
import numpy as np
import os

# ---------- 0. 路径配置 ----------
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(OUT_DIR, "images")

img_left_path = os.path.join(IMG_DIR, "left.png")    # 基准图（左图，固定不动）
img_right_path = os.path.join(IMG_DIR, "right.png")  # 待变换图（右图，投影到左图坐标系）

# ---------- 1. 读图并转灰度 ----------
def imread_unicode(path):
    """支持中文路径的 imread"""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

img_a = imread_unicode(img_left_path)    # img_a : 左图
img_b = imread_unicode(img_right_path)   # img_b : 右图
assert img_a is not None, "无法读取左.png"
assert img_b is not None, "无法读取右.png"

def imwrite_unicode(path, img):
    """支持中文路径的 imwrite"""
    ext = os.path.splitext(path)[1]
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)

gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
print(f"左图尺寸: {img_a.shape}, 右图尺寸: {img_b.shape}")

# ---------- 2. SIFT 特征检测与描述子计算 ----------
sift = cv2.SIFT_create()
kp_a, des_a = sift.detectAndCompute(gray_a, None)
kp_b, des_b = sift.detectAndCompute(gray_b, None)
print(f"左图关键点数量: {len(kp_a)}, 右图关键点数量: {len(kp_b)}")

# ---------- 3. FLANN 近似最近邻匹配 + Lowe 比率测试 ----------
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des_b, des_a, k=2)   # query=右图(b), train=左图(a)

good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)
print(f"经 Lowe 比率筛选后的优质匹配点数量: {len(good_matches)}")

# ---------- 4. 绘制匹配连线图 ----------
# drawMatches(img1, kp1, img2, kp2, matches1to2, ...)
# 这里把右图画在左图左边展示连线
matched_img = cv2.drawMatches(
    img_b, kp_b, img_a, kp_a, good_matches, None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
imwrite_unicode(os.path.join(IMG_DIR, "matched_keypoints.png"), matched_img)

# ---------- 5. 提取匹配点坐标并估计单应矩阵 H ----------
# 参考代码约定：src_pts = kp_b（待变换图 b），dst_pts = kp_a（基准图 a）
# findHomography(src, dst) 求的是把 src 投影到 dst 的变换矩阵 H
src_pts = np.float32([kp_b[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp_a[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
inliers = int(mask.sum()) if mask is not None else 0
print(f"RANSAC 内点数量: {inliers} / {len(good_matches)}")
print("单应矩阵 H =\n", H)

# ---------- 6. 计算拼接画布尺寸并加入平移 ----------
h_a, w_a = img_a.shape[:2]
h_b, w_b = img_b.shape[:2]

# 右图四个角点经 H 变换后的位置
pts = np.float32([[0, 0], [0, h_b], [w_b, h_b], [w_b, 0]]).reshape(-1, 1, 2)
dst_corners = cv2.perspectiveTransform(pts, H)

# 与左图四个角点合并，求整体外接矩形
all_corners = np.concatenate(
    [dst_corners,
     np.float32([[0, 0], [w_a, 0], [w_a, h_a], [0, h_a]]).reshape(-1, 1, 2)],
    axis=0
)
[x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
[x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

# 平移矩阵：把负坐标拉回正
translation = np.array([[1, 0, -x_min],
                        [0, 1, -y_min],
                        [0, 0, 1]], dtype=np.float32)

# ---------- 7. 透视变换右图并贴上左图 ----------
stitched = cv2.warpPerspective(
    img_b,
    translation @ H,
    (x_max - x_min, y_max - y_min)
)
stitched[-y_min: h_a - y_min, -x_min: w_a - x_min] = img_a

imwrite_unicode(os.path.join(IMG_DIR, "stitched_result.png"), stitched)
print(f"拼接结果尺寸: {stitched.shape}")
print("已保存: matched_keypoints.png, stitched_result.png")
