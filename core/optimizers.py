"""
无导数优化器（Derivative-Free Optimizer）
用于 EBM 模型的推理阶段，通过迭代采样和重要性采样来找到最优解
"""
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
from torch.optim.lr_scheduler import _LRScheduler
import math


class DerivativeFreeOptimizer:
    """
    无导数优化器：基于 IBC 的推理优化器
    
    使用迭代的重要性采样（Importance Sampling）和噪声衰减策略来优化能量函数。
    这种方法不需要计算梯度，适用于复杂的、可能不可微的能量函数。
    
    算法流程：
    1. 从均匀分布中采样初始候选解
    2. 对每个候选解计算能量值
    3. 将能量转换为概率分布（softmax(-energy)）
    4. 根据概率分布进行重要性采样
    5. 在重采样的解周围添加噪声
    6. 噪声逐渐衰减，重复步骤 2-5
    7. 返回能量最低的解
    
    参考: Implicit Behavioral Cloning (IBC)
    """
    
    def __init__(
        self, 
        bounds: np.ndarray,
        num_samples: int = 16384,
        num_iters: int = 3,
        noise_scale: float = 0.33,
        noise_shrink: float = 0.5,
        device: str = 'cpu'
    ):
        """
        Args:
            bounds: 搜索空间的边界，形状为 (2, action_dim)
                   bounds[0, :] 是下界，bounds[1, :] 是上界
                   例如: [[-1.0, -1.0], [1.0, 1.0]] 表示 2D 坐标范围 [-1, 1]^2
            num_samples: 每次迭代的候选样本数（IBC 默认 2^14 = 16384）
            num_iters: 优化迭代次数（IBC 默认 3）
            noise_scale: 初始噪声标准差（IBC 默认 0.33）
            noise_shrink: 噪声衰减系数，每次迭代噪声乘以此系数（IBC 默认 0.5）
            device: 计算设备 ('cpu' 或 'cuda')
        """
        self.bounds = np.array(bounds)
        self.num_samples = num_samples
        self.num_iters = num_iters
        self.noise_scale = noise_scale
        self.noise_shrink = noise_shrink
        self.device = torch.device(device)
        
        # 验证边界形状
        assert self.bounds.shape[0] == 2, "bounds 应该是形状为 (2, action_dim) 的数组"
        assert self.bounds.shape[1] >= 1, "action_dim 至少为 1"
    
    def _sample(self, num_samples: int) -> torch.Tensor:
        """
        从均匀分布中采样
        
        Args:
            num_samples: 采样数量
            
        Returns:
            采样结果，形状为 (num_samples, action_dim)
        """
        size = (num_samples, self.bounds.shape[1])
        samples = np.random.uniform(
            self.bounds[0, :], 
            self.bounds[1, :], 
            size=size
        )
        return torch.as_tensor(samples, dtype=torch.float32, device=self.device)
    
    @torch.no_grad()
    def infer(
        self, 
        x: torch.Tensor, 
        ebm: torch.nn.Module,
        return_history: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[Dict[str, Any]]]]:
        """
        执行优化推理
        
        Args:
            x: 输入观测（例如图像），形状为 (B, ...)
            ebm: 能量基模型，接受 (x, y) 并返回能量值
            return_history: 是否返回优化过程的历史记录（用于可视化）
            
        Returns:
            predictions: 预测的最优解，形状为 (B, action_dim)
            history: 如果 return_history=True，返回优化历史列表
                    每个元素是一个字典，包含：
                    - 'samples': 当前迭代的所有样本 (B, num_samples, action_dim)
                    - 'energies': 对应的能量值 (B, num_samples)
                    - 'best_samples': 当前最优样本 (B, action_dim)
                    - 'noise_scale': 当前噪声标准差
        """
        ebm.eval()
        B = x.size(0)
        noise_scale = self.noise_scale
        bounds_tensor = torch.as_tensor(
            self.bounds, 
            dtype=torch.float32
        ).to(self.device)
        
        # 初始化样本：从均匀分布采样
        samples = self._sample(B * self.num_samples)
        samples = samples.reshape(B, self.num_samples, -1)
        
        history = [] if return_history else None
        
        # 迭代优化
        for i in range(self.num_iters):
            # 计算能量
            energies = ebm(x, samples)  # (B, num_samples)
            
            # 转换为概率分布（能量越低，概率越高）
            probs = F.softmax(-1.0 * energies, dim=-1)  # (B, num_samples)
            
            # 重要性采样：按概率分布重采样
            idxs = torch.multinomial(
                probs, 
                self.num_samples, 
                replacement=True
            )  # (B, num_samples)
            
            # 获取重采样的样本
            samples = samples[
                torch.arange(B).unsqueeze(-1), 
                idxs
            ]  # (B, num_samples, action_dim)
            
            # 添加高斯噪声（探索）
            samples = samples + torch.randn_like(samples) * noise_scale
            
            # 限制在边界内
            samples = samples.clamp(
                min=bounds_tensor[0, :], 
                max=bounds_tensor[1, :]
            )
            
            # 记录历史（在添加噪声和裁剪之后）
            if return_history:
                # 重新计算能量以获取添加噪声后的能量
                energies_after_noise = ebm(x, samples)
                best_indices = energies_after_noise.argmin(dim=1)
                best_samples = samples[torch.arange(B), best_indices]
                history.append({
                    'samples': samples.clone().cpu().numpy(),
                    'energies': energies_after_noise.cpu().numpy(),
                    'best_samples': best_samples.clone().cpu().numpy(),
                    'noise_scale': noise_scale
                })
            
            # 噪声衰减
            noise_scale *= self.noise_shrink
        
        # 最终评估：返回能量最低的样本
        energies = ebm(x, samples)
        probs = F.softmax(-1.0 * energies, dim=-1)
        best_idxs = probs.argmax(dim=-1)  # (B,)
        predictions = samples[torch.arange(B), best_idxs, :]  # (B, action_dim)
        
        if return_history:
            return predictions, history
        return predictions, None


class DerivativeFreeOptimizer1D(DerivativeFreeOptimizer):
    """
    1D 版本的无导数优化器
    
    专门用于 1D 任务的优化器，继承自 DerivativeFreeOptimizer。
    主要用于教程和实验，提供更简洁的接口。
    """
    
    def __init__(
        self,
        bounds: Tuple[float, float] = (-1.0, 1.0),
        num_samples: int = 1024,
        num_iters: int = 3,
        noise_scale: float = 0.2,
        noise_shrink: float = 0.5,
        device: str = 'cpu'
    ):
        """
        Args:
            bounds: 1D 搜索空间的边界 (min, max)
            num_samples: 每次迭代的候选样本数
            num_iters: 优化迭代次数
            noise_scale: 初始噪声标准差
            noise_shrink: 噪声衰减系数
            device: 计算设备
        """
        # 将 1D bounds 转换为 (2, 1) 格式
        bounds_array = np.array([[bounds[0]], [bounds[1]]])
        super().__init__(
            bounds=bounds_array,
            num_samples=num_samples,
            num_iters=num_iters,
            noise_scale=noise_scale,
            noise_shrink=noise_shrink,
            device=device
        )


class SGLDSampler:
    """
    随机梯度朗之万动力学（Stochastic Gradient Langevin Dynamics, SGLD）采样器
    
    用于从能量基模型中采样负样本，通过添加噪声的梯度下降来探索能量景观。
    SGLD 可以生成更具挑战性的负样本，从而提高 EBM 的训练质量。
    
    更新规则:
    y_{t+1} = y_t - step_size * ∇_y E(x, y_t) + sqrt(2 * step_size) * ε
    其中 ε ~ N(0, I)
    
    参考: 
    - Welling & Teh, "Bayesian Learning via Stochastic Gradient Langevin Dynamics"
    - Du & Mordatch, "Implicit Generation and Modeling with Energy-Based Models"
    """
    
    def __init__(
        self,
        bounds: np.ndarray,
        step_size: float = 0.01,
        num_steps: int = 20,
        noise_scale: float = 0.01,
        device: str = 'cpu'
    ):
        """
        Args:
            bounds: 搜索空间的边界，形状为 (2, action_dim)
            step_size: SGLD 步长（学习率）
            num_steps: SGLD 迭代步数
            noise_scale: 噪声标准差
            device: 计算设备
        """
        self.bounds = np.array(bounds)
        self.step_size = step_size
        self.num_steps = num_steps
        self.noise_scale = noise_scale
        self.device = torch.device(device)
    
    def sample(
        self,
        x: torch.Tensor,
        ebm: torch.nn.Module,
        num_samples: int,
        return_trajectory: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        使用 SGLD 从 EBM 中采样
        
        Args:
            x: 输入观测，形状为 (B, ...)
            ebm: 能量基模型
            num_samples: 每个输入采样的样本数
            return_trajectory: 是否返回采样轨迹
            
        Returns:
            samples: 采样结果，形状为 (B, num_samples, action_dim)
            trajectory: 如果 return_trajectory=True，返回轨迹列表
        """
        ebm.eval()
        B = x.size(0)
        action_dim = self.bounds.shape[1]
        bounds_tensor = torch.as_tensor(
            self.bounds,
            dtype=torch.float32
        ).to(self.device)
        
        # 初始化样本（随机）
        samples = torch.rand(
            B, num_samples, action_dim,
            device=self.device
        ) * (bounds_tensor[1, :] - bounds_tensor[0, :]) + bounds_tensor[0, :]
        
        samples.requires_grad_(True)
        trajectory = [] if return_trajectory else None
        
        # SGLD 迭代
        for step in range(self.num_steps):
            # 计算能量
            energies = ebm(x, samples)  # (B, num_samples)
            
            # 计算梯度
            grad = torch.autograd.grad(
                energies.sum(),
                samples,
                create_graph=False
            )[0]
            
            # SGLD 更新
            with torch.no_grad():
                noise = torch.randn_like(samples) * self.noise_scale
                samples = samples - self.step_size * grad + noise
                
                # 限制在边界内
                samples = samples.clamp(
                    min=bounds_tensor[0, :],
                    max=bounds_tensor[1, :]
                )
            
            samples.requires_grad_(True)
            
            if return_trajectory:
                trajectory.append(samples.detach().clone())
        
        samples = samples.detach()
        
        if return_trajectory:
            return samples, trajectory
        return samples, None


class ULASampler:
    """
    未校正朗之万算法（Unadjusted Langevin Algorithm, ULA）采样器
    
    修改版：匹配 Implicit Behavioral Cloning (IBC) 官方实现的逻辑。
    包含 Polynomial 学习率衰减和特定的更新公式。
    
    IBC 更新规则:
    delta = step_size * (0.5 * grad + noise_scale * N(0, I))
    y_{t+1} = y_t - delta
    
    参考:
    - Implicit Behavioral Cloning (IBC) 官方代码 mcmc.py
    """
    
    def __init__(
        self,
        bounds: np.ndarray,
        step_size: float = 0.1,  # 初始步长
        num_steps: int = 100,
        noise_scale: float = 1.0,
        step_size_final: float = 1e-5,
        step_size_power: float = 2.0,
        delta_action_clip: float = 0.1,  # 每步最大变化量（匹配 IBC）
        device: str = 'cpu'
    ):
        """
        Args:
            bounds: 搜索空间的边界，形状为 (2, action_dim)
            step_size: 初始步长
            num_steps: 迭代步数
            noise_scale: 噪声缩放因子
            step_size_final: 最终步长（用于 Polynomial Schedule）
            step_size_power: Polynomial Schedule 的幂次
            delta_action_clip: 每步最大动作变化量（相对于归一化后的动作范围）
                              IBC 默认 0.1，确保采样稳定
            device: 计算设备
        """
        self.bounds = np.array(bounds)
        self.step_size = step_size
        self.num_steps = num_steps
        self.noise_scale = noise_scale
        self.step_size_final = step_size_final
        self.step_size_power = step_size_power
        self.delta_action_clip = delta_action_clip
        self.device = torch.device(device)
    
    def _get_step_size(self, step: int) -> float:
        """计算当前步的步长（Polynomial Schedule）"""
        if self.num_steps <= 1:
            return self.step_size
        
        # 匹配 ibc.agents.mcmc.PolynomialSchedule
        # rate = ((init - final) * ((1 - (step / (total_steps-1))) ** power)) + final
        # 注意：step 从 0 开始，到 num_steps-1
        
        progress = float(step) / float(self.num_steps - 1)
        rate = (self.step_size - self.step_size_final) * (
            (1.0 - progress) ** self.step_size_power
        ) + self.step_size_final
        return rate

    def sample(
        self,
        x: torch.Tensor,
        ebm: torch.nn.Module,
        num_samples: int,
        init_samples: Optional[torch.Tensor] = None,
        return_trajectory: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        使用 ULA 从 EBM 中采样
        
        Args:
            x: 输入观测，形状为 (B, ...)
            ebm: 能量基模型
            num_samples: 每个输入采样的样本数
            init_samples: 初始样本，形状为 (B, num_samples, action_dim)
                         如果为None，则随机初始化
            return_trajectory: 是否返回采样轨迹
            
        Returns:
            samples: 采样结果，形状为 (B, num_samples, action_dim)
            trajectory: 如果 return_trajectory=True，返回轨迹列表
        """
        # 保存模型的训练状态（匹配IBC的training=False）
        was_training = ebm.training
        ebm.eval()
        
        B = x.size(0)
        action_dim = self.bounds.shape[1]
        bounds_tensor = torch.as_tensor(
            self.bounds,
            dtype=torch.float32
        ).to(self.device)
        
        # 初始化样本
        if init_samples is None:
            samples = torch.rand(
                B, num_samples, action_dim,
                device=self.device
            ) * (bounds_tensor[1, :] - bounds_tensor[0, :]) + bounds_tensor[0, :]
        else:
            samples = init_samples.clone().to(self.device)
        
        samples.requires_grad_(True)
        trajectory = [] if return_trajectory else None
        
        # ULA 迭代
        current_step_size = self.step_size  # 初始步长
        
        for step in range(self.num_steps):
            # 更新步长 (下一轮使用，但这里我们按 step 更新)
            # IBC 代码逻辑：stepsize = schedule.get_rate(step_index + 1) 用于下一次
            # 这里我们直接计算当前 step 的 rate
            current_step_size = self._get_step_size(step)

            # 计算能量
            energies = ebm(x, samples)  # (B, num_samples)
            
            # 计算梯度
            grad = torch.autograd.grad(
                energies.sum(),
                samples,
                create_graph=False
            )[0]
            
            # ULA 更新：匹配 IBC 逻辑
            # delta = step_size * (0.5 * grad + noise)
            # 注意：IBC 中 grad 是 dE/da，我们这里也是 dE/da
            with torch.no_grad():
                noise = torch.randn_like(samples) * self.noise_scale
                
                # IBC 更新公式
                # gradient_scale = 0.5
                # de_dact = gradient_scale * grad + noise
                # delta = step_size * de_dact
                delta = current_step_size * (0.5 * grad + noise)
                
                # 关键修复：限制每步的变化量（匹配 IBC 的 delta_action_clip）
                # IBC 代码：delta_action_clip = delta_action_clip * 0.5 * (max_actions - min_actions)
                action_range = bounds_tensor[1, :] - bounds_tensor[0, :]
                delta_clip_scaled = self.delta_action_clip * 0.5 * action_range
                
                # 裁剪 delta 到 ±delta_clip_scaled 范围
                delta = torch.clamp(delta, min=-delta_clip_scaled, max=delta_clip_scaled)
                
                samples = samples - delta
                
                # 限制在边界内
                samples = samples.clamp(
                    min=bounds_tensor[0, :],
                    max=bounds_tensor[1, :]
                )
            
            samples.requires_grad_(True)
            
            if return_trajectory:
                trajectory.append(samples.detach().clone())
        
        samples = samples.detach()
        
        # 恢复模型的训练状态
        if was_training:
            ebm.train()
        
        if return_trajectory:
            return samples, trajectory
        return samples, None

class TFExponentialDecay(_LRScheduler):
    def __init__(self, optimizer, decay_rate, decay_steps, staircase=False, last_epoch=-1):
        """
        TensorFlow ExponentialDecay 的 PyTorch 实现
        
        Args:
            optimizer: PyTorch 优化器
            decay_rate: 衰减率（例如 0.99）
            decay_steps: 衰减步数
            staircase: 如果为 True，使用阶梯式衰减（每 decay_steps 步衰减一次）
            last_epoch: 最后一个 epoch 的索引，默认为 -1（从头开始）
        """
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.staircase = staircase
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch

        if self.staircase:
            # TF: staircase=True
            factor = self.decay_rate ** (step // self.decay_steps)
        else:
            # TF: staircase=False
            factor = self.decay_rate ** (step / self.decay_steps)

        return [base_lr * factor for base_lr in self.base_lrs]
