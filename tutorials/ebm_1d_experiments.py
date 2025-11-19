"""
1D EBM 实验（R → R）
测试不同类型的 1D 映射任务，包括不连续点、分段线性、高频不连续、随机噪声和多值映射

任务：
A. Single discontinuity (单一不连续点)
B. Piecewise linear with varying slopes (分段线性)
C. High-frequency discontinuities (高频不连续/阶梯函数)
D. Random noise / uncorrelated labels (随机噪声)
E. Multi-valued examples (多值映射)

评估指标：
- RMSE（单值区域）
- Mode coverage（多模数据）
- Sharpness at discontinuity（不连续处的梯度）
- Nearest-neighbor index（随机噪声任务）
- NLL proxy（负对数似然）
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Callable
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
script_name = Path(__file__).stem
output_dir = Path(__file__).parent / 'plots' / script_name
output_dir.mkdir(parents=True, exist_ok=True)


# ==================== 数据集定义 ====================

class Dataset1D(Dataset):
    """1D 数据集基类"""
    
    def __init__(self, num_samples: int = 100, x_min: float = 0.0, x_max: float = 1.0):
        self.num_samples = num_samples
        self.x_min = x_min
        self.x_max = x_max
        self.x_data = []
        self.y_data = []
        self.task_name = "Base"
        self.is_multimodal = False
        
    def generate_data(self):
        """生成数据（子类实现）"""
        raise NotImplementedError
    
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor([self.x_data[idx]])
        y = torch.FloatTensor([self.y_data[idx]])
        return x, y


class SingleDiscontinuityDataset(Dataset1D):
    """A. 单一不连续点数据集"""
    
    def __init__(self, num_samples: int = 100):
        super().__init__(num_samples)
        self.task_name = "Single Discontinuity"
        self.discontinuity_x = 0.5
        self.generate_data()
    
    def generate_data(self):
        # 在不连续点附近和远处都采样
        # 50% 样本在边界附近 [0.4, 0.6]
        # 50% 样本在远离边界的区域
        n_near = self.num_samples // 2
        n_far = self.num_samples - n_near
        
        # 边界附近
        x_near = np.random.uniform(0.4, 0.6, n_near)
        # 远离边界
        x_far = np.concatenate([
            np.random.uniform(0.0, 0.4, n_far // 2),
            np.random.uniform(0.6, 1.0, n_far - n_far // 2)
        ])
        
        self.x_data = np.concatenate([x_near, x_far])
        self.y_data = np.where(self.x_data < self.discontinuity_x, 0.0, 1.0)


class PiecewiseLinearDataset(Dataset1D):
    """B. 分段线性数据集"""
    
    def __init__(self, num_samples: int = 100):
        super().__init__(num_samples)
        self.task_name = "Piecewise Linear"
        self.generate_data()
    
    def generate_data(self):
        self.x_data = np.random.uniform(0.0, 1.0, self.num_samples)
        self.y_data = np.zeros(self.num_samples)
        
        for i, x in enumerate(self.x_data):
            if x < 0.25:
                self.y_data[i] = 0.0
            elif x < 0.6:
                self.y_data[i] = 3 * x - 0.8
            else:
                self.y_data[i] = -2 * x + 2


class HighFrequencyDataset(Dataset1D):
    """C. 高频不连续数据集（阶梯函数）"""
    
    def __init__(self, num_samples: int = 100, num_steps: int = 4):
        super().__init__(num_samples)
        self.task_name = f"High-Frequency ({num_steps} steps)"
        self.num_steps = num_steps
        self.generate_data()
    
    def generate_data(self):
        self.x_data = np.random.uniform(0.0, 1.0, self.num_samples)
        self.y_data = np.floor(self.num_steps * self.x_data) / self.num_steps


class RandomNoiseDataset(Dataset1D):
    """D. 随机噪声数据集（不相关标签）"""
    
    def __init__(self, num_samples: int = 100, noise_std: float = 0.05, num_clusters: int = 3):
        super().__init__(num_samples)
        self.num_clusters = num_clusters
        self.noise_std = noise_std
        self.task_name = f"Clustered Noise ({num_clusters} clusters)"
        self.generate_data()

    def generate_data(self):
        self.x_data = np.random.uniform(0.0, 1.0, self.num_samples)
        
        # 均匀采样 cluster center
        cluster_centers = np.linspace(0.1, 0.9, self.num_clusters)

        # 为每个样本随机分配一个 cluster
        assigned_clusters = np.random.choice(self.num_clusters, size=self.num_samples)

        # y = cluster_center + gaussian_noise
        self.y_data = np.array([
            cluster_centers[c] + np.random.normal(0, self.noise_std)
            for c in assigned_clusters
        ])

        self.y_data = np.clip(self.y_data, 0.0, 1.0)


class MultiValuedDataset(Dataset1D):
    """E. 多值映射数据集
    
    对于某些 x 值，存在两个有效的 y 值：y = ± max(0, 1 - (2x - 1)^2)
    这样训练集中同一个 x 会出现两次，分别对应不同的 y 值
    """
    
    def __init__(self, num_samples: int = 100):
        super().__init__(num_samples)
        self.task_name = "Multi-Valued"
        self.is_multimodal = True
        self.generate_data()
    
    def generate_data(self):
        # 生成 num_samples/2 个不同的 x，每个 x 对应两个 y 值
        base_num = self.num_samples // 2
        base_x = np.random.uniform(0.0, 1.0, base_num)
        
        self.x_data = []
        self.y_data = []
        
        for x in base_x:
            # y = ± max(0, 1 - (2x - 1)^2)
            base_y = max(0.0, 1.0 - (2 * x - 1) ** 2)
            
            if base_y > 0.1:  # 只有当 base_y 足够大时才添加两个模态
                # 添加正值
                y_pos = 0.5 + base_y * 0.4  # 归一化到 [0.5, 0.9] 范围
                self.x_data.append(x)
                self.y_data.append(y_pos)
                
                # 添加负值
                y_neg = 0.5 - base_y * 0.4  # 归一化到 [0.1, 0.5] 范围
                self.x_data.append(x)
                self.y_data.append(y_neg)
            else:
                # 对于接近边界的 x，只添加一个值
                self.x_data.append(x)
                self.y_data.append(0.5)
        
        self.x_data = np.array(self.x_data)
        self.y_data = np.array(self.y_data)
        
        # 更新 num_samples
        self.num_samples = len(self.x_data)
    
    def get_true_modes(self, x: float) -> List[float]:
        """获取给定 x 的所有真实模态"""
        base_y = max(0.0, 1.0 - (2 * x - 1) ** 2)
        if base_y > 0.1:
            y_pos = 0.5 + base_y * 0.4
            y_neg = 0.5 - base_y * 0.4
            return [y_neg, y_pos]
        else:
            return [0.5]


# ==================== EBM 模型 ====================

class EBM1D(nn.Module):
    """1D 能量基模型（使用 MLP）"""
    
    def __init__(self, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        
        # MLP 输入：x (1维) + y (1维) = 2维
        layers = []
        input_dim = 2
        
        for i in range(num_layers):
            if i == 0:
                layers.extend([
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                ])
            else:
                layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                ])
        
        layers.append(nn.Linear(hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1) - 输入
        y: (B, N, 1) - N 个候选输出
        返回: (B, N) - 每个候选的能量值
        """
        B = x.size(0)
        N = y.size(1)
        
        # 扩展 x 以匹配 y 的维度
        x_expanded = x.unsqueeze(1).expand(B, N, 1)  # (B, N, 1)
        
        # 拼接 x 和 y
        fused = torch.cat([x_expanded, y], dim=-1)  # (B, N, 2)
        fused = fused.reshape(B * N, 2)
        
        # 计算能量
        energy = self.mlp(fused)  # (B*N, 1)
        return energy.view(B, N)


class DerivativeFreeOptimizer1D:
    """1D 无导数优化器"""
    
    def __init__(self, bounds: Tuple[float, float], num_samples: int = 1024, 
                 num_iters: int = 3, noise_scale: float = 0.1, 
                 noise_shrink: float = 0.5, device: str = 'cpu'):
        self.bounds = bounds
        self.num_samples = num_samples
        self.num_iters = num_iters
        self.noise_scale = noise_scale
        self.noise_shrink = noise_shrink
        self.device = torch.device(device)
    
    def _sample(self, num_samples: int) -> torch.Tensor:
        """从均匀分布采样"""
        samples = np.random.uniform(self.bounds[0], self.bounds[1], (num_samples, 1))
        return torch.as_tensor(samples, dtype=torch.float32, device=self.device)
    
    @torch.no_grad()
    def infer(self, x: torch.Tensor, ebm: nn.Module, 
              return_history: bool = False, return_all_modes: bool = False,
              num_modes: int = 3, mode_threshold: float = 0.15) -> torch.Tensor:
        """
        优化推理
        x: (B, 1) - 输入
        ebm: EBM 模型
        return_history: 是否返回优化历史
        return_all_modes: 是否返回所有模态（用于多值映射）
        num_modes: 返回的模态数量上限
        mode_threshold: 模态之间的最小距离
        返回: (B, 1) 或 (B, K, 1) - 预测的输出（K 个模态）
        """
        ebm.eval()
        B = x.size(0)
        noise_scale = self.noise_scale
        bounds_tensor = torch.tensor(self.bounds, dtype=torch.float32, device=self.device)
        
        # 初始化样本
        samples = self._sample(B * self.num_samples)
        samples = samples.reshape(B, self.num_samples, 1)
        
        history = [] if return_history else None
        
        for i in range(self.num_iters):
            # 计算能量
            energies = ebm(x, samples)  # (B, num_samples)
            
            # 转换为概率
            probs = F.softmax(-1.0 * energies, dim=-1)  # (B, num_samples)
            
            # 重要性采样
            idxs = torch.multinomial(probs, self.num_samples, replacement=True)
            samples = samples[torch.arange(B).unsqueeze(-1), idxs]
            
            # 添加噪声
            samples = samples + torch.randn_like(samples) * noise_scale
            
            # 限制在边界内
            samples = samples.clamp(min=bounds_tensor[0], max=bounds_tensor[1])
            
            # 记录历史
            if return_history:
                energies_after_noise = ebm(x, samples)
                history.append({
                    'samples': samples.clone().cpu().numpy(),
                    'energies': energies_after_noise.cpu().numpy(),
                    'noise_scale': noise_scale
                })
            
            # 噪声衰减
            noise_scale *= self.noise_shrink
        
        # 最终能量
        energies = ebm(x, samples)  # (B, num_samples)
        
        if return_all_modes:
            # 返回多个模态：找到能量的局部极小值
            all_modes = []
            
            for b in range(B):
                # 对每个样本，找到低能量的点
                energies_b = energies[b].cpu().numpy()
                samples_b = samples[b, :, 0].cpu().numpy()
                
                # 排序找到最低能量的点
                sorted_indices = np.argsort(energies_b)
                
                modes = []
                for idx in sorted_indices:
                    candidate = samples_b[idx]
                    
                    # 检查是否与已有模态足够不同
                    is_new_mode = True
                    for existing_mode in modes:
                        if abs(candidate - existing_mode) < mode_threshold:
                            is_new_mode = False
                            break
                    
                    if is_new_mode:
                        modes.append(candidate)
                        if len(modes) >= num_modes:
                            break
                
                # 如果模态不足，用最佳点填充
                while len(modes) < num_modes:
                    modes.append(modes[0] if modes else 0.5)
                
                all_modes.append(modes[:num_modes])
            
            predictions = torch.FloatTensor(all_modes).unsqueeze(-1).to(self.device)  # (B, K, 1)
        else:
            # 返回单个最优值
            probs = F.softmax(-1.0 * energies, dim=-1)
            best_idxs = probs.argmax(dim=-1)
            predictions = samples[torch.arange(B), best_idxs, :]  # (B, 1)
        
        if return_history:
            return predictions, history
        return predictions


# ==================== 训练函数 ====================

def train_1d_ebm(dataset: Dataset1D, num_epochs: int = 1000, 
                 batch_size: int = 16, lr: float = 0.001,
                 num_negatives: int = 64, hidden_dim: int = 128,
                 device: str = 'cpu') -> Tuple[nn.Module, List[float]]:
    """训练 1D EBM"""
    
    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 创建模型
    model = EBM1D(hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 训练历史
    train_losses = []
    
    print(f"\n训练 {dataset.task_name} 数据集...")
    print(f"  样本数: {len(dataset)}")
    print(f"  隐藏层维度: {hidden_dim}")
    print(f"  负样本数: {num_negatives}")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            B = x_batch.size(0)
            
            # 正样本
            positive_y = y_batch.unsqueeze(1)  # (B, 1, 1)
            
            # 负样本：从 [0, 1] 均匀采样
            negative_y = torch.rand(B, num_negatives, 1, device=device)
            
            # 合并正负样本
            all_y = torch.cat([positive_y, negative_y], dim=1)  # (B, 1+num_negatives, 1)
            
            # 随机打乱
            permutation = torch.rand(B, all_y.size(1), device=device).argsort(dim=1)
            all_y = all_y[torch.arange(B).unsqueeze(-1), permutation]
            
            # 找到正样本的位置
            ground_truth = (permutation == 0).nonzero()[:, 1]
            
            # 计算能量
            energies = model(x_batch, all_y)  # (B, 1+num_negatives)
            
            # InfoNCE 损失
            logits = -energies
            loss = F.cross_entropy(logits, ground_truth)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 200 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return model, train_losses


# ==================== 评估指标 ====================

def compute_rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """计算 RMSE"""
    return np.sqrt(np.mean((predictions - targets) ** 2))


def compute_mode_coverage(model: nn.Module, dataset: Dataset1D, 
                         optimizer: DerivativeFreeOptimizer1D,
                         num_modes: int = 2, threshold: float = 0.1,
                         device: str = 'cpu') -> Dict:
    """计算模态覆盖率（对多值映射任务）"""
    if not dataset.is_multimodal:
        return {"coverage": 1.0, "detected_modes": 1}
    
    # 对每个 x 位置，使用 return_all_modes 获取多个模态
    test_x = np.linspace(0.1, 0.9, 30)
    detected_modes_list = []
    true_modes_list = []
    mode_errors = []
    
    model.eval()
    with torch.no_grad():
        for x in test_x:
            x_tensor = torch.FloatTensor([[x]]).to(device)
            
            # 获取真实模态
            if hasattr(dataset, 'get_true_modes'):
                true_modes = dataset.get_true_modes(x)
                true_modes_list.append(len(true_modes))
            else:
                true_modes = None
                true_modes_list.append(num_modes)
            
            # 使用 return_all_modes 获取多个预测模态
            pred_modes = optimizer.infer(
                x_tensor, model, 
                return_history=False, 
                return_all_modes=True,
                num_modes=num_modes,
                mode_threshold=threshold
            )  # (1, num_modes, 1)
            
            pred_modes_values = pred_modes[0, :, 0].cpu().numpy()
            
            # 去重：移除重复的模态
            unique_pred_modes = []
            for p in pred_modes_values:
                is_unique = True
                for existing in unique_pred_modes:
                    if abs(p - existing) < threshold:
                        is_unique = False
                        break
                if is_unique:
                    unique_pred_modes.append(p)
            
            detected_modes_list.append(len(unique_pred_modes))
            
            # 如果有真实模态，计算匹配误差
            if true_modes is not None and len(true_modes) > 1:
                # 计算每个真实模态到最近预测模态的距离
                min_distances = []
                for true_mode in true_modes:
                    distances = [abs(true_mode - pred) for pred in unique_pred_modes]
                    min_distances.append(min(distances) if distances else 1.0)
                mode_errors.append(np.mean(min_distances))
    
    avg_detected_modes = np.mean(detected_modes_list)
    avg_true_modes = np.mean(true_modes_list)
    coverage = min(avg_detected_modes / max(avg_true_modes, 1), 1.0)
    
    result = {
        "coverage": coverage,
        "avg_detected_modes": avg_detected_modes,
        "max_detected_modes": max(detected_modes_list),
        "avg_true_modes": avg_true_modes
    }
    
    if mode_errors:
        result["avg_mode_error"] = np.mean(mode_errors)
        result["max_mode_error"] = np.max(mode_errors)
    
    return result


def compute_sharpness_at_discontinuity(model: nn.Module, 
                                      discontinuity_x: float,
                                      device: str = 'cpu',
                                      delta: float = 0.01) -> float:
    """计算不连续点处的清晰度（梯度）"""
    model.eval()
    
    # 在不连续点左右采样
    x_left = torch.FloatTensor([[discontinuity_x - delta]]).to(device)
    x_right = torch.FloatTensor([[discontinuity_x + delta]]).to(device)
    
    # 使用优化器找到最优 y
    optimizer_temp = DerivativeFreeOptimizer1D(
        bounds=(0.0, 1.0), num_samples=1024, num_iters=3, device=device
    )
    
    with torch.no_grad():
        y_left = optimizer_temp.infer(x_left, model, return_history=False)
        y_right = optimizer_temp.infer(x_right, model, return_history=False)
    
    # 计算梯度
    sharpness = abs(y_right[0, 0].item() - y_left[0, 0].item()) / (2 * delta)
    
    return sharpness


def compute_nearest_neighbor_index(model: nn.Module, dataset: Dataset1D,
                                   optimizer: DerivativeFreeOptimizer1D,
                                   device: str = 'cpu') -> Dict:
    """计算最近邻指数（对随机噪声任务）"""
    # 在测试点上预测
    test_x = np.linspace(0.0, 1.0, 50)
    predictions = []
    
    model.eval()
    with torch.no_grad():
        for x in test_x:
            x_tensor = torch.FloatTensor([[x]]).to(device)
            pred = optimizer.infer(x_tensor, model, return_history=False)
            predictions.append(pred[0, 0].cpu().item())
    
    predictions = np.array(predictions)
    
    # 对每个预测，找到最近的训练样本
    train_x = dataset.x_data
    train_y = dataset.y_data
    
    distances_to_nearest = []
    for i, (x_test, y_pred) in enumerate(zip(test_x, predictions)):
        # 找到最近的训练样本（在 x 空间）
        x_distances = np.abs(train_x - x_test)
        nearest_idx = np.argmin(x_distances)
        nearest_y = train_y[nearest_idx]
        
        # 计算 y 空间的距离
        y_distance = abs(y_pred - nearest_y)
        distances_to_nearest.append(y_distance)
    
    return {
        "mean_distance": np.mean(distances_to_nearest),
        "std_distance": np.std(distances_to_nearest),
        "median_distance": np.median(distances_to_nearest)
    }


def compute_nll_proxy(model: nn.Module, dataset: Dataset1D,
                     device: str = 'cpu', temperature: float = 0.1) -> float:
    """计算负对数似然代理（NLL proxy）"""
    model.eval()
    
    # 在网格上评估
    y_grid = torch.linspace(0.0, 1.0, 100, device=device).view(1, 100, 1)
    
    nlls = []
    with torch.no_grad():
        for x, y_true in dataset:
            x_tensor = x.unsqueeze(0).to(device)
            y_true_value = y_true.item()
            
            # 计算能量
            energies = model(x_tensor, y_grid)  # (1, 100)
            
            # 转换为概率
            probs = F.softmax(-energies / temperature, dim=-1)  # (1, 100)
            
            # 找到最接近 y_true 的网格点
            y_grid_np = y_grid.squeeze().cpu().numpy()
            closest_idx = np.argmin(np.abs(y_grid_np - y_true_value))
            
            # 计算负对数似然
            nll = -torch.log(probs[0, closest_idx] + 1e-10)
            nlls.append(nll.item())
    
    return np.mean(nlls)


# ==================== 可视化函数 ====================

def visualize_1d_results(model: nn.Module, dataset: Dataset1D,
                        train_losses: List[float],
                        optimizer: DerivativeFreeOptimizer1D,
                        device: str = 'cpu'):
    """可视化 1D 实验结果"""
    
    model.eval()
    
    # 创建图形
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # 1. 训练损失曲线
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(train_losses, 'b-', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    
    # 2. 数据和预测曲线
    ax2 = fig.add_subplot(gs[0, 1])
    
    # 绘制训练数据
    ax2.scatter(dataset.x_data, dataset.y_data, c='blue', s=30, 
               alpha=0.6, label='Training Data', marker='x')
    
    # 预测曲线（使用密集散点图）
    test_x = np.linspace(0.0, 1.0, 300)  # 增加点数以获得更密集的散点
    predictions = []
    
    # 对于多值映射，获取所有模态
    if dataset.is_multimodal:
        all_modes_predictions = []
        with torch.no_grad():
            for x in test_x:
                x_tensor = torch.FloatTensor([[x]]).to(device)
                pred_modes = optimizer.infer(
                    x_tensor, model, 
                    return_history=False,
                    return_all_modes=True,
                    num_modes=3,
                    mode_threshold=0.15
                )  # (1, 3, 1)
                all_modes_predictions.append(pred_modes[0, :, 0].cpu().numpy())
        
        all_modes_predictions = np.array(all_modes_predictions)  # (300, 3)
        
        # 绘制每个模态（使用散点图）
        colors_modes = ['red', 'green', 'orange']
        markers_modes = ['o', 's', '^']
        for i in range(3):
            ax2.scatter(test_x, all_modes_predictions[:, i], 
                       color=colors_modes[i], s=15, alpha=0.6,
                       marker=markers_modes[i], label=f'Mode {i+1}', 
                       edgecolors='none')
        
        # 主预测（第一个模态）
        predictions = all_modes_predictions[:, 0]
    else:
        with torch.no_grad():
            for x in test_x:
                x_tensor = torch.FloatTensor([[x]]).to(device)
                pred = optimizer.infer(x_tensor, model, return_history=False)
                predictions.append(pred[0, 0].cpu().item())
        
        predictions = np.array(predictions)
        # 使用密集散点图代替连线
        ax2.scatter(test_x, predictions, c='red', s=15, alpha=0.6, 
                   label='Prediction', marker='o', edgecolors='none')
    
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title(f'{dataset.task_name}: Data and Prediction')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.05, 1.05)
    
    # 3. 能量等高线图 E(x, y)
    ax3 = fig.add_subplot(gs[0, 2])
    
    # 创建网格
    x_grid = np.linspace(0.0, 1.0, 100)
    y_grid = np.linspace(0.0, 1.0, 100)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    
    # 计算能量
    energy_grid = np.zeros_like(X_grid)
    with torch.no_grad():
        for i in range(len(x_grid)):
            x_tensor = torch.FloatTensor([[x_grid[i]]]).to(device)
            y_tensor = torch.FloatTensor(y_grid).view(1, 100, 1).to(device)
            energies = model(x_tensor, y_tensor)
            energy_grid[:, i] = energies[0].cpu().numpy()
    
    # 绘制等高线
    levels = 20
    im = ax3.contourf(X_grid, Y_grid, energy_grid, levels=levels, cmap='viridis')
    ax3.contour(X_grid, Y_grid, energy_grid, levels=levels, 
               colors='white', alpha=0.3, linewidths=0.5)
    
    # 叠加训练数据
    ax3.scatter(dataset.x_data, dataset.y_data, c='red', s=20, 
               alpha=0.8, marker='o', edgecolors='white', linewidths=1)
    
    ax3.set_xlabel('x (Input)')
    ax3.set_ylabel('y (Output)')
    ax3.set_title('Energy Landscape E(x, y)')
    plt.colorbar(im, ax=ax3, label='Energy')
    
    # 4. 残差分析
    ax4 = fig.add_subplot(gs[1, 0])
    
    # 计算残差（使用插值获取真实值）
    residuals = predictions - np.interp(test_x, dataset.x_data, dataset.y_data)
    
    # 使用散点图显示残差
    ax4.scatter(test_x, residuals, c='green', s=15, alpha=0.6, 
               marker='o', edgecolors='none')
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax4.fill_between(test_x, residuals, alpha=0.2, color='green')
    ax4.set_xlabel('x')
    ax4.set_ylabel('Residual (Pred - True)')
    ax4.set_title('Prediction Residuals')
    ax4.grid(True, alpha=0.3)
    
    # 5. 能量剖面（选择几个 x 位置）
    ax5 = fig.add_subplot(gs[1, 1])
    
    x_samples = [0.2, 0.5, 0.8]
    colors_profile = ['blue', 'red', 'green']
    
    with torch.no_grad():
        for x_val, color in zip(x_samples, colors_profile):
            x_tensor = torch.FloatTensor([[x_val]]).to(device)
            y_tensor = torch.FloatTensor(y_grid).view(1, 100, 1).to(device)
            energies = model(x_tensor, y_tensor)
            ax5.plot(y_grid, energies[0].cpu().numpy(), 
                    color=color, linewidth=2, label=f'x={x_val:.1f}')
    
    ax5.set_xlabel('y')
    ax5.set_ylabel('Energy E(x, y)')
    ax5.set_title('Energy Profile at Different x')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 预测不确定性（多次采样）
    ax6 = fig.add_subplot(gs[1, 2])
    
    # 对几个 x 位置进行多次预测
    x_test_samples = np.linspace(0.1, 0.9, 30)  # 增加采样点
    predictions_multi = []
    
    with torch.no_grad():
        for x in x_test_samples:
            x_tensor = torch.FloatTensor([[x]]).to(device)
            preds = []
            for _ in range(10):  # 多次采样
                pred = optimizer.infer(x_tensor, model, return_history=False)
                preds.append(pred[0, 0].cpu().item())
            predictions_multi.append(preds)
    
    predictions_multi = np.array(predictions_multi)
    
    # 绘制均值和标准差（使用散点图）
    mean_preds = predictions_multi.mean(axis=1)
    std_preds = predictions_multi.std(axis=1)
    
    # 绘制所有采样点
    for i, x_val in enumerate(x_test_samples):
        ax6.scatter([x_val] * 10, predictions_multi[i], 
                   c='blue', s=10, alpha=0.3, marker='o', edgecolors='none')
    
    # 绘制均值（散点）
    ax6.scatter(x_test_samples, mean_preds, c='darkblue', s=40, 
               marker='s', label='Mean', edgecolors='white', linewidths=1, zorder=5)
    
    # 绘制标准差范围（填充区域）
    ax6.fill_between(x_test_samples, 
                     mean_preds - std_preds, 
                     mean_preds + std_preds,
                     alpha=0.2, color='blue', label='±1 Std')
    
    ax6.scatter(dataset.x_data, dataset.y_data, c='red', s=20, 
               alpha=0.5, marker='x', label='Training Data', zorder=4)
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    ax6.set_title('Prediction Uncertainty (10 samples per x)')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    # 7-9. 评估指标
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    # 计算各种指标
    metrics_text = f"=== Evaluation Metrics for {dataset.task_name} ===\n\n"
    
    # RMSE
    test_y_true = np.interp(test_x, dataset.x_data, dataset.y_data)
    rmse = compute_rmse(predictions, test_y_true)
    metrics_text += f"RMSE: {rmse:.4f}\n\n"
    
    # 模态覆盖率（如果是多值映射）
    if dataset.is_multimodal:
        mode_coverage = compute_mode_coverage(model, dataset, optimizer, device=device)
        metrics_text += f"Mode Coverage:\n"
        metrics_text += f"  - Coverage: {mode_coverage['coverage']:.4f}\n"
        metrics_text += f"  - Avg Detected Modes: {mode_coverage['avg_detected_modes']:.2f}\n"
        metrics_text += f"  - Max Detected Modes: {mode_coverage['max_detected_modes']}\n"
        metrics_text += f"  - Avg True Modes: {mode_coverage['avg_true_modes']:.2f}\n"
        if 'avg_mode_error' in mode_coverage:
            metrics_text += f"  - Avg Mode Error: {mode_coverage['avg_mode_error']:.4f}\n"
            metrics_text += f"  - Max Mode Error: {mode_coverage['max_mode_error']:.4f}\n"
        metrics_text += "\n"
    
    # 不连续点处的清晰度（如果适用）
    if hasattr(dataset, 'discontinuity_x'):
        sharpness = compute_sharpness_at_discontinuity(
            model, dataset.discontinuity_x, device=device
        )
        metrics_text += f"Sharpness at Discontinuity (x={dataset.discontinuity_x}):\n"
        metrics_text += f"  - |dy/dx|: {sharpness:.2f}\n\n"
    
    # 最近邻指数（如果是随机噪声任务）
    if "Random Noise" in dataset.task_name:
        nn_index = compute_nearest_neighbor_index(model, dataset, optimizer, device=device)
        metrics_text += f"Nearest-Neighbor Index:\n"
        metrics_text += f"  - Mean Distance: {nn_index['mean_distance']:.4f}\n"
        metrics_text += f"  - Std Distance: {nn_index['std_distance']:.4f}\n"
        metrics_text += f"  - Median Distance: {nn_index['median_distance']:.4f}\n\n"
    
    # NLL proxy
    nll = compute_nll_proxy(model, dataset, device=device)
    metrics_text += f"NLL Proxy (temperature=0.1): {nll:.4f}\n"
    
    ax7.text(0.1, 0.5, metrics_text, 
            fontsize=11, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # 保存图形
    plt.suptitle(f'1D EBM Experiment: {dataset.task_name}', 
                fontsize=16, fontweight='bold', y=0.98)
    
    filename = f"1d_ebm_{dataset.task_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')}.png"
    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    print(f"\n图形已保存到: {output_dir / filename}")
    
    plt.show()


def visualize_multimodal_inference(model: nn.Module, dataset: MultiValuedDataset,
                                   optimizer: DerivativeFreeOptimizer1D,
                                   device: str = 'cpu'):
    """专门可视化多值映射的推理结果"""
    
    model.eval()
    
    # 创建图形
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. 训练数据和多模态预测
    ax1 = fig.add_subplot(gs[0, :])
    
    # 绘制训练数据（按 x 排序以便看清楚）
    sorted_indices = np.argsort(dataset.x_data)
    x_sorted = dataset.x_data[sorted_indices]
    y_sorted = dataset.y_data[sorted_indices]
    
    ax1.scatter(x_sorted, y_sorted, c='blue', s=40, 
               alpha=0.6, label='Training Data', marker='o', edgecolors='navy')
    
    # 预测多个模态（使用密集散点图）
    test_x = np.linspace(0.05, 0.95, 200)  # 增加点数
    all_modes = []
    
    with torch.no_grad():
        for x in test_x:
            x_tensor = torch.FloatTensor([[x]]).to(device)
            pred_modes = optimizer.infer(
                x_tensor, model,
                return_history=False,
                return_all_modes=True,
                num_modes=3,
                mode_threshold=0.12
            )
            all_modes.append(pred_modes[0, :, 0].cpu().numpy())
    
    all_modes = np.array(all_modes)  # (200, 3)
    
    # 绘制每个模态（使用散点图）
    colors_modes = ['red', 'green', 'orange']
    markers = ['o', 's', '^']
    for i in range(3):
        # 去除重复点（模态相同的情况）
        unique_mask = np.ones(len(test_x), dtype=bool)
        if i > 0:
            for j in range(i):
                unique_mask &= np.abs(all_modes[:, i] - all_modes[:, j]) > 0.05
        
        ax1.scatter(test_x[unique_mask], all_modes[unique_mask, i],
                   color=colors_modes[i], s=20, alpha=0.7,
                   marker=markers[i], label=f'Predicted Mode {i+1}',
                   edgecolors='none')
    
    ax1.set_xlabel('x (Input)', fontsize=12)
    ax1.set_ylabel('y (Output)', fontsize=12)
    ax1.set_title('Multi-Valued Mapping: Training Data and Multiple Predicted Modes', 
                 fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    
    # 2. 能量景观（展示双峰结构）
    ax2 = fig.add_subplot(gs[1, 0])
    
    # 选择几个 x 位置展示能量剖面
    x_samples = [0.3, 0.5, 0.7]
    colors_profile = ['blue', 'red', 'green']
    y_grid = np.linspace(0.0, 1.0, 200)
    
    with torch.no_grad():
        for x_val, color in zip(x_samples, colors_profile):
            x_tensor = torch.FloatTensor([[x_val]]).to(device)
            y_tensor = torch.FloatTensor(y_grid).view(1, 200, 1).to(device)
            energies = model(x_tensor, y_tensor)
            
            # 标准化能量以便比较
            energies_np = energies[0].cpu().numpy()
            energies_normalized = (energies_np - energies_np.min()) / (energies_np.max() - energies_np.min() + 1e-6)
            
            ax2.plot(y_grid, energies_normalized, 
                    color=color, linewidth=2.5, label=f'x={x_val:.1f}', alpha=0.8)
            
            # 标记局部最小值
            from scipy.signal import find_peaks
            inv_energy = -energies_normalized
            peaks, _ = find_peaks(inv_energy, prominence=0.1)
            if len(peaks) > 0:
                ax2.scatter(y_grid[peaks], energies_normalized[peaks],
                           s=100, color=color, marker='*', 
                           edgecolors='black', linewidths=1.5, zorder=5)
    
    ax2.set_xlabel('y (Output)', fontsize=11)
    ax2.set_ylabel('Normalized Energy E(x, y)', fontsize=11)
    ax2.set_title('Energy Profile at Different x (★ = Local Minima)', 
                 fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.1)
    
    # 3. 模态数量统计
    ax3 = fig.add_subplot(gs[1, 1])
    
    # 统计每个 x 位置检测到的模态数量
    test_x_fine = np.linspace(0.05, 0.95, 50)
    mode_counts = []
    
    with torch.no_grad():
        for x in test_x_fine:
            x_tensor = torch.FloatTensor([[x]]).to(device)
            pred_modes = optimizer.infer(
                x_tensor, model,
                return_history=False,
                return_all_modes=True,
                num_modes=3,
                mode_threshold=0.12
            )
            modes = pred_modes[0, :, 0].cpu().numpy()
            
            # 计算唯一模态数量
            unique_count = 1
            for i in range(1, len(modes)):
                is_unique = True
                for j in range(i):
                    if abs(modes[i] - modes[j]) < 0.1:
                        is_unique = False
                        break
                if is_unique:
                    unique_count += 1
            
            mode_counts.append(unique_count)
    
    # 绘制柱状图
    bars = ax3.bar(test_x_fine, mode_counts, width=0.015, 
                   color='steelblue', alpha=0.7, edgecolor='navy')
    
    # 添加真实模态数量的理论曲线
    true_mode_counts = []
    for x in test_x_fine:
        true_modes = dataset.get_true_modes(x)
        true_mode_counts.append(len(true_modes))
    
    ax3.plot(test_x_fine, true_mode_counts, 'r-', linewidth=3, 
            label='True # Modes', alpha=0.8)
    ax3.scatter(test_x_fine, mode_counts, c='blue', s=30, 
               alpha=0.6, label='Detected # Modes', zorder=5)
    
    ax3.set_xlabel('x (Input)', fontsize=11)
    ax3.set_ylabel('Number of Modes', fontsize=11)
    ax3.set_title('Mode Count: True vs Detected', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(0.5, 3.5)
    ax3.set_xlim(-0.05, 1.05)
    
    plt.suptitle('Multi-Valued Mapping: EBM as Set-Valued Function', 
                fontsize=16, fontweight='bold', y=0.98)
    
    filename = "1d_ebm_multimodal_detailed.png"
    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    print(f"\n多模态详细分析图已保存到: {output_dir / filename}")
    
    plt.show()


# ==================== 主函数 ====================

def run_experiment(dataset_class, dataset_kwargs: Dict = None, 
                  num_epochs: int = 1000, device: str = 'cpu'):
    """运行单个实验"""
    
    # 创建数据集
    if dataset_kwargs is None:
        dataset_kwargs = {}
    dataset = dataset_class(**dataset_kwargs)
    
    # 训练模型
    model, train_losses = train_1d_ebm(
        dataset, 
        num_epochs=num_epochs,
        batch_size=16,
        lr=0.001,
        num_negatives=64,
        hidden_dim=128,
        device=device
    )
    
    # 创建优化器
    optimizer = DerivativeFreeOptimizer1D(
        bounds=(0.0, 1.0),
        num_samples=1024,
        num_iters=3,
        noise_scale=0.1,
        noise_shrink=0.5,
        device=device
    )
    
    # 可视化结果
    visualize_1d_results(model, dataset, train_losses, optimizer, device=device)
    
    # 对多值映射任务，添加额外的详细可视化
    if isinstance(dataset, MultiValuedDataset):
        visualize_multimodal_inference(model, dataset, optimizer, device=device)
    
    return model, dataset, train_losses


def main():
    """主函数：运行所有实验"""
    
    print("=" * 70)
    print("1D EBM 实验：测试不同类型的 1D 映射任务")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    # 实验列表
    experiments = [
        # ("A. Single Discontinuity", SingleDiscontinuityDataset, {"num_samples": 150}),
        # ("B. Piecewise Linear", PiecewiseLinearDataset, {"num_samples": 150}),
        # ("C. High-Frequency (4 steps)", HighFrequencyDataset, {"num_samples": 150, "num_steps": 4}),
        # ("C. High-Frequency (8 steps)", HighFrequencyDataset, {"num_samples": 200, "num_steps": 8}),
        # ("D. Random Noise", RandomNoiseDataset, {"num_samples": 50, "noise_std": 0.005, "num_clusters": 10}),
        ("E. Multi-Valued", MultiValuedDataset, {"num_samples": 200}),
    ]
    
    # 运行所有实验
    for exp_name, dataset_class, dataset_kwargs in experiments:
        print(f"\n{'=' * 70}")
        print(f"运行实验: {exp_name}")
        print(f"{'=' * 70}")
        
        try:
            run_experiment(
                dataset_class, 
                dataset_kwargs=dataset_kwargs,
                num_epochs=1000,
                device=device
            )
        except Exception as e:
            print(f"实验 {exp_name} 失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'=' * 70}")
    print("所有实验完成！")
    print(f"结果已保存到: {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

