# -*- coding: utf-8 -*-
"""
mlp_mnist.py

实验四：手动搭建多层感知机（MLP）完成 MNIST 手写数字识别（PyTorch 实现）

本文件是「实验四参考代码」的补全版，在原稿 1~71 行的基础上：
    1. 修复原稿中的问题：无效的 shape 语句、测试集 shuffle=True、无随机种子、超参说明
    2. 补上原稿缺失的第 72 行之后的绘图与结果输出部分（原稿 import 了 plt 却从未使用）
    3. 增加逐 epoch 的训练/测试指标记录，使 loss、accuracy 曲线有数据可画
    4. 增加混淆矩阵、错分样例可视化、每类别准确率统计，满足实验报告需要

运行方式：
    python mlp_mnist.py
输出：
    ./data      MNIST 数据集（首次运行自动下载）
    ./results/  曲线图、混淆矩阵、错分样例、模型权重、训练历史
"""

# %% ========== 1. 导入依赖与全局配置 ==========
import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# 只保存图片、不弹窗（脚本化/无界面环境下更稳）。
# 若想在 Jupyter 里交互查看，删掉下面这行 use('Agg')，把 savefig 换成 plt.show() 即可。
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torchvision import datasets, transforms

# Windows 下中文显示与负号显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 随机种子：保证每次实验的初始化、数据打乱顺序一致，报告里的准确率才可复现
SEED = 2026
torch.manual_seed(SEED)
np.random.seed(SEED)

batch_size = 2048          # 原稿取值；想提升精度可改 256，并把 lr 调大
learning_rate = 0.01       # SGD 的初始学习率
epochs = 30                # 训练轮数
DATA_DIR = './data'        # MNIST 存放目录
RESULT_DIR = './results'   # 结果输出目录

os.makedirs(RESULT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('PyTorch 版本:', torch.__version__)
print('当前设备:', device)


# %% ========== 2. 数据加载 ==========
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(DATA_DIR, train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),          # PIL(28,28) -> Tensor(1,28,28)，像素归一到 [0,1]
                   ])),
    batch_size=batch_size, shuffle=True,          # 训练集必须打乱，避免同类样本扎堆影响收敛
)

# 测试集不打乱：一是评估无需打乱，二是后面画错分样例时要靠索引回查原图
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(DATA_DIR, train=False, transform=transforms.Compose([
        transforms.ToTensor(),
    ])),
    batch_size=batch_size, shuffle=False,
)

# 原稿这里的写法是 `train_loader.dataset.data.shape,`（带尾逗号的元组，不会输出任何东西），
# 改成真正的打印，用于确认数据形状为 (60000, 28, 28)
print('训练集张量形状:', tuple(train_loader.dataset.data.shape))
print('测试集张量形状:', tuple(test_loader.dataset.data.shape))


# %% ========== 3. 模型定义 ==========
class mlp(nn.Module):
    """两层全连接网络：784 -> 128 -> 10，中间用 ReLU 激活"""

    def __init__(self):
        super(mlp, self).__init__()
        self.l1 = nn.Linear(784, 128)   # 输入层 -> 隐藏层
        self.l2 = nn.Linear(128, 10)    # 隐藏层 -> 输出层（10 个类别）

    def forward(self, x):
        a1 = self.l1(x)
        x1 = F.relu(a1)                 # 隐藏层激活，引入非线性
        a2 = self.l2(x1)                # 输出 logits，交给 cross_entropy
        return a2


model = mlp().to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
print(model)


# %% ========== 4. 训练 + 逐轮评估 ==========
history = {'train_loss': [], 'test_loss': [], 'test_acc': []}

for epoch in range(epochs):
    # ---- 训练阶段 ----
    model.train()
    running_loss = 0.0
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.view(x.shape[0], -1).to(device), y.to(device)   # (B,1,28,28) -> (B,784)
        optimizer.zero_grad()
        output = model(x)
        loss = F.cross_entropy(output, y)                        # 内部已做 softmax，故传 logits
        loss.backward()
        optimizer.step()
        # 以样本数加权累计，避免最后一个不满 batch 的批次造成偏差
        running_loss += loss.item() * x.size(0)

    train_loss = running_loss / len(train_loader.dataset)

    # ---- 测试阶段（每个 epoch 结束评估一次，供画曲线用）----
    model.eval()
    correct, total, test_loss_sum = 0, 0, 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x = x.view(x.shape[0], -1).to(device)
            y = y.to(device)
            output = model(x)
            # reduction='sum' 后按样本总数相除，得到的是精确平均 loss，而不是按 batch 平均的近似值
            test_loss_sum += F.cross_entropy(output, y, reduction='sum').item()
            pred = output.argmax(1, keepdim=True)                # 取概率最大的类别
            correct += pred.eq(y.view_as(pred)).sum().item()
            total += y.size(0)
            all_preds.append(pred.view(-1).cpu())
            all_labels.append(y.cpu())

    test_loss = test_loss_sum / total
    acc = correct / total

    history['train_loss'].append(train_loss)
    history['test_loss'].append(test_loss)
    history['test_acc'].append(acc)

    print('epoch:{}, train_loss:{:.4f}, loss:{:.4f}, acc:{:.4f}'.format(
        epoch, train_loss, test_loss, acc))

print('\n' + '=' * 64)
print('训练结束：共 {} 个 epoch，最终测试集 loss = {:.4f}，accuracy = {:.2%}'.format(
    epochs, test_loss, acc))
print('=' * 64)

all_preds = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()


# %% ========== 5. 结果可视化（原稿缺失部分）==========

# 5.1 数据集样例（每个类别取 1 张 + 6 张随机，覆盖 0~9 全部类别）
train_images = train_loader.dataset.data.numpy()        # (60000, 28, 28)，uint8
train_targets = train_loader.dataset.targets.numpy()

per_class_idx = [np.where(train_targets == d)[0][0] for d in range(10)]
random_idx = np.random.choice(len(train_images), 6, replace=False)
sample_idx = np.concatenate([np.array(per_class_idx), random_idx])

fig, axes = plt.subplots(4, 4, figsize=(7, 7.6))
for k, ax in enumerate(axes.flat):
    idx = sample_idx[k]
    ax.imshow(train_images[idx], cmap='gray')
    ax.set_title('标签:{}'.format(train_targets[idx]), fontsize=10)
    ax.axis('off')
fig.suptitle('MNIST 训练集样例（28×28 灰度图，0~9 各一例 + 随机 6 例）', fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(RESULT_DIR, 'dataset_samples.png'), dpi=200)
plt.close(fig)
print('已保存:', os.path.join(RESULT_DIR, 'dataset_samples.png'))

# 5.2 网络结构示意图（矩形高度按神经元数量示意）
def draw_layer(ax, x, w, h, label, facecolor):
    y0 = (1.0 - h) / 2.0
    box = FancyBboxPatch((x, y0), w, h, boxstyle='round,pad=0.015',
                         linewidth=1.4, edgecolor='#333333', facecolor=facecolor)
    ax.add_patch(box)
    ax.text(x + w / 2.0, 0.5, label, ha='center', va='center', fontsize=11.5)


def draw_arrow(ax, x1, x2, text):
    arr = FancyArrowPatch((x1, 0.52), (x2, 0.52), arrowstyle='-|>', mutation_scale=16,
                          linewidth=1.4, color='#666666')
    ax.add_patch(arr)
    ax.text((x1 + x2) / 2.0, 0.38, text, ha='center', va='top',
            fontsize=9, color='#333333', linespacing=1.3)


fig, ax = plt.subplots(figsize=(10, 4.0))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

draw_layer(ax, 0.02, 0.22, 0.50, '输入层\n784', '#dbe7f3')
draw_layer(ax, 0.38, 0.20, 0.30, '隐藏层\n128\nReLU', '#e8f0db')
draw_layer(ax, 0.72, 0.22, 0.14, '输出层\n10', '#f3e6db')

draw_arrow(ax, 0.24, 0.38, 'W1  784×128\n+128 偏置')
draw_arrow(ax, 0.58, 0.72, 'W2  128×10\n+10 偏置')

ax.text(0.5, 0.13, '28×28 图像拉平为 784 维向量；输出 10 个 logits 对应数字 0~9',
        ha='center', va='center', fontsize=10, color='#333333')
ax.text(0.5, 0.05, '总参数量 = (784×128 + 128) + (128×10 + 10) = 101,770',
        ha='center', va='center', fontsize=10.5, color='#111111')
ax.set_title('MLP 网络结构：784 → 128 → 10', fontsize=13, pad=14)

fig.tight_layout()
fig.savefig(os.path.join(RESULT_DIR, 'network_structure.png'), dpi=200)
plt.close(fig)
print('已保存:', os.path.join(RESULT_DIR, 'network_structure.png'))

# 5.3 训练/测试 loss 曲线 + 测试 accuracy 曲线
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
ep = np.arange(1, len(history['train_loss']) + 1)

axes[0].plot(ep, history['train_loss'], 'o-', lw=1.8, label='训练集 loss')
axes[0].plot(ep, history['test_loss'], 's-', lw=1.8, label='测试集 loss')
axes[0].set_xlabel('epoch')
axes[0].set_ylabel('loss')
axes[0].set_title('交叉熵损失曲线')
axes[0].grid(alpha=0.3)
axes[0].legend()

axes[1].plot(ep, np.array(history['test_acc']) * 100, 'o-', color='tab:green', lw=1.8)
axes[1].set_xlabel('epoch')
axes[1].set_ylabel('accuracy (%)')
axes[1].set_title('测试集准确率曲线')
axes[1].grid(alpha=0.3)
axes[1].axhline(history['test_acc'][-1] * 100, ls='--', lw=1, color='gray',
                label='最终准确率 {:.2f}%'.format(history['test_acc'][-1] * 100))
axes[1].legend()

fig.tight_layout()
fig.savefig(os.path.join(RESULT_DIR, 'loss_acc_curve.png'), dpi=200)
plt.close(fig)
print('已保存:', os.path.join(RESULT_DIR, 'loss_acc_curve.png'))

# 5.4 混淆矩阵（行 = 真实类别，列 = 预测类别）
num_classes = 10
cm = np.zeros((num_classes, num_classes), dtype=np.int64)
np.add.at(cm, (all_labels, all_preds), 1)

fig, ax = plt.subplots(figsize=(8.2, 7))
im = ax.imshow(cm, cmap='Blues')
ax.set_xlabel('预测类别')
ax.set_ylabel('真实类别')
ax.set_xticks(range(num_classes))
ax.set_yticks(range(num_classes))
ax.set_title('MNIST 测试集混淆矩阵（样本数）')
# 在每格写上数值，深色格子用白字
thresh = cm.max() / 2.0
for i in range(num_classes):
    for j in range(num_classes):
        ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=9,
                color='white' if cm[i, j] > thresh else 'black')
fig.colorbar(im, ax=ax, fraction=0.045)
fig.tight_layout()
fig.savefig(os.path.join(RESULT_DIR, 'confusion_matrix.png'), dpi=200)
plt.close(fig)
print('已保存:', os.path.join(RESULT_DIR, 'confusion_matrix.png'))

# 5.5 每类别识别准确率（对角元素 / 该类别真实样本数）
per_class_acc = cm.diagonal() / cm.sum(axis=1)
print('\n每类别准确率:')
for d in range(num_classes):
    n = int(cm[d].sum())
    print('  数字 {:<2d}: {:.2%}   (样本数 {})'.format(d, per_class_acc[d], n))

# 5.6 正确预测样例（与错分样例对照）
test_images = test_loader.dataset.data.numpy()          # (10000, 28, 28)，uint8
correct_idx = np.where(all_preds == all_labels)[0]

if len(correct_idx) > 0:
    # 按类别各取 1 张，再补 6 张随机样本，避免出现"全是同一数字"的样例图
    per_class_correct = []
    for d in range(num_classes):
        d_idx = correct_idx[all_labels[correct_idx] == d]
        if len(d_idx) > 0:
            per_class_correct.append(d_idx[0])
    random_correct = np.random.choice(correct_idx, 6, replace=False)
    sample_idx = np.concatenate([np.array(per_class_correct, dtype=np.int64), random_correct])

    n_show = min(16, len(sample_idx))
    fig, axes = plt.subplots(4, 4, figsize=(8.5, 8.8))
    for k, ax in enumerate(axes.flat):
        if k < n_show:
            idx = sample_idx[k]
            ax.imshow(test_images[idx], cmap='gray')
            ax.set_title('真值:{} 预测:{}'.format(all_labels[idx], all_preds[idx]),
                         fontsize=10, color='#1a7f37')
        ax.axis('off')
    fig.suptitle('预测正确样例（0~9 各一例 + 随机 6 例）', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, 'correct_samples.png'), dpi=200)
    plt.close(fig)
    print('已保存:', os.path.join(RESULT_DIR, 'correct_samples.png'))

# 5.7 错分样例可视化
wrong_idx = np.where(all_preds != all_labels)[0]
print('\n错分样本总数: {} / {}'.format(len(wrong_idx), len(all_labels)))

if len(wrong_idx) > 0:
    n_show = min(16, len(wrong_idx))
    fig, axes = plt.subplots(4, 4, figsize=(8.5, 8.8))
    for k, ax in enumerate(axes.flat):
        if k < n_show:
            idx = wrong_idx[k]
            ax.imshow(test_images[idx], cmap='gray')
            ax.set_title('真值:{} 预测:{}'.format(all_labels[idx], all_preds[idx]), fontsize=10)
        ax.axis('off')
    fig.suptitle('前 {} 个错分样本'.format(n_show), fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, 'misclassified.png'), dpi=200)
    plt.close(fig)
    print('已保存:', os.path.join(RESULT_DIR, 'misclassified.png'))


# %% ========== 6. 保存模型与训练历史 ==========
torch.save(model.state_dict(), os.path.join(RESULT_DIR, 'mlp_mnist.pt'))
with open(os.path.join(RESULT_DIR, 'history.json'), 'w', encoding='utf-8') as f:
    json.dump(history, f, ensure_ascii=False, indent=2)

print('\n产物清单:')
for name in ['dataset_samples.png', 'network_structure.png', 'loss_acc_curve.png',
             'confusion_matrix.png', 'correct_samples.png', 'misclassified.png',
             'mlp_mnist.pt', 'history.json']:
    path = os.path.join(RESULT_DIR, name)
    print('  {}  {}'.format('√' if os.path.exists(path) else '×', path))
