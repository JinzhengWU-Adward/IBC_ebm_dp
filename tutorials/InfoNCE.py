"""
InfoNCE损失函数和梯度计算
观察拉近正样本，推远负样本的行为
"""
import numpy as np
import matplotlib.pyplot as plt

# 初始化
query = np.array([-3.0, -4.0])
positive = np.array([1.0, 0.0])
negatives = np.array([[ -1.0,  1.0],
                      [ -1.0, -1.0],
                      [ 0.5, -1.0]])

# 超参数
lr = 0.1
tau = 0.1
steps = 20

def info_nce_loss(q, pos, negs, tau):
    """计算InfoNCE损失"""
    pos_sim = np.exp(np.dot(q, pos) / tau)
    neg_sims = np.exp(np.dot(q, negs.T) / tau)
    loss = -np.log(pos_sim / (pos_sim + np.sum(neg_sims)))
    return loss

# 存储损失值用于绘图
losses = []

# 创建两个子图：左边是位置图，右边是损失值图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

for step in range(steps):
    # 计算InfoNCE损失
    loss = info_nce_loss(query, positive, negatives, tau)
    losses.append(loss)
    
    # 计算正确的梯度
    # InfoNCE loss = -log(exp(q·pos/τ) / (exp(q·pos/τ) + Σexp(q·neg_i/τ)))
    # 对q求导：∂loss/∂q = (1/denom) * (-pos/τ * Σneg_sims + Σ(neg_i/τ * neg_sims_i))
    pos_sim = np.exp(np.dot(query, positive)/tau)
    neg_sims = np.exp(np.dot(query, negatives.T)/tau)  # shape: (num_negatives,)
    denom = pos_sim + np.sum(neg_sims)
    
    # 正确的梯度计算
    # Σ(neg_i/τ * neg_sims_i) = (1/τ) * Σ(neg_i * neg_sims_i) = (1/τ) * negatives^T @ neg_sims
    grad = (1.0 / (tau * denom)) * (
        -positive * np.sum(neg_sims) + 
        np.dot(negatives.T, neg_sims)  # negatives^T @ neg_sims，结果shape为(2,)
    )
    
    # 更新 query
    query -= lr * grad

    # 绘制位置图
    ax1.cla()
    ax1.scatter(*positive, color='green', s=100, label='positive', marker='o')
    ax1.scatter(negatives[:,0], negatives[:,1], color='red', s=100, label='negatives', marker='s')
    ax1.scatter(query[0], query[1], color='blue', s=100, label='query', marker='*')
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-4, 4)
    ax1.set_title(f'Step {step+1}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # 绘制损失值图
    ax2.cla()
    ax2.plot(range(1, step+2), losses, 'b-o', linewidth=2, markersize=6)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('InfoNCE Loss')
    ax2.set_title('InfoNCE Loss over Steps')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, steps+1)
    
    plt.tight_layout()
    plt.pause(0.3)

plt.show()
