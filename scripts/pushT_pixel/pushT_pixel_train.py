"""
使用 PyTorch 复刻 IBC 的 PushT (RGB 图像) 环境训练
基于 diffusion_policy 的 pusht_cchi_v7_replay.zarr 数据集
参考 pushing_pixel_train.py 的网络框架
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import sys
import zarr
import wandb


# 添加项目路径
IBC_ROOT = Path(__file__).parent.parent.parent  # IBC_ebm_dp
sys.path.insert(0, str(IBC_ROOT))

from core.models import PixelEBM
from core.optimizers import ULASampler, TFExponentialDecay


class PushTImageDataset(Dataset):
    """
    PushT (RGB 图像) 环境数据集
    从 diffusion_policy 的 zarr 格式数据加载
    
    数据格式（来自 pusht_cchi_v7_replay.zarr）:
    - img: (25650, 96, 96, 3) uint8
    - action: (25650, 2) float32
    - state: (25650, 5) float32
    - episode_ends: (206,) int64
    """
    
    def __init__(
        self,
        zarr_path: str,
        obs_seq_len: int = 2,
        normalize_actions: bool = True,
        target_height: int = 96,  # PushT 图像原始大小
        target_width: int = 96,   # PushT 图像原始大小
    ):
        """
        Args:
            zarr_path: zarr 数据目录路径
            obs_seq_len: 观测序列长度（默认 2）
            normalize_actions: 是否归一化动作到 [-1, 1]
            target_height: 目标图像高度
            target_width: 目标图像宽度
        """
        self.zarr_path = Path(zarr_path)
        self.obs_seq_len = obs_seq_len
        self.normalize_actions = normalize_actions
        self.target_height = target_height
        self.target_width = target_width
        
        # 加载 zarr 数据
        print(f"加载 zarr 数据: {self.zarr_path}")
        root = zarr.open(str(self.zarr_path), mode='r')
        
        # 读取数据
        self.images = root['data']['img'][:]  # (N, 96, 96, 3) uint8
        self.actions = root['data']['action_delta'][:]  # (N, 2) float32
        self.states = root['data']['state'][:]  # (N, 5) float32
        self.episode_ends = root['meta']['episode_ends'][:]  # (num_episodes,) int64
        
        print(f"图像形状: {self.images.shape}")
        print(f"动作形状: {self.actions.shape}")
        print(f"状态形状: {self.states.shape}")
        print(f"回合数: {len(self.episode_ends)}")
        
        # 构建样本索引
        self.samples = []
        episode_start = 0
        
        for episode_idx, episode_end in enumerate(self.episode_ends):
            episode_length = episode_end - episode_start
            
            # 为每个回合构建滑动窗口样本
            # 首帧重复样本：[img0, img0] -> action0
            if episode_length >= 1:
                self.samples.append({
                    'indices': [episode_start] * self.obs_seq_len,
                    'action_idx': episode_start
                })
            
            # 滑动窗口样本
            for i in range(episode_length):
                if i + self.obs_seq_len <= episode_length:
                    # 构建图像序列索引
                    img_indices = [episode_start + i + j for j in range(self.obs_seq_len)]
                    # 动作索引为窗口最后一帧的动作
                    action_idx = episode_start + i + self.obs_seq_len - 1
                    
                    self.samples.append({
                        'indices': img_indices,
                        'action_idx': action_idx
                    })
            
            episode_start = episode_end
        
        print(f"共构建 {len(self.samples)} 个样本")
        
        # 计算动作归一化参数
        if self.normalize_actions:
            # Min-Max 归一化到 [-1, 1]
            self.action_min = self.actions.min(axis=0)
            self.action_max = self.actions.max(axis=0)
            self.action_range = self.action_max - self.action_min
            self.action_range[self.action_range < 1e-6] = 1.0  # 避免除零
            
            print(f"动作范围: min={self.action_min}, max={self.action_max}")
        else:
            self.action_min = None
            self.action_max = None
            self.action_range = None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_indices = sample['indices']
        action_idx = sample['action_idx']
        
        # 加载图像序列
        images = []
        for img_idx in img_indices:
            img = self.images[img_idx]  # (96, 96, 3) uint8
            # 转换到 [0, 1] 范围
            img_array = img.astype(np.float32) / 255.0  # (H, W, 3)
            images.append(img_array)
        
        # 转换为序列格式 (seq_len, H, W, 3)
        image_sequence = np.stack(images, axis=0)  # (seq_len, H, W, 3)
        
        # 加载动作
        action = self.actions[action_idx].copy()  # (2,)
        
        # 归一化动作（Min-Max 到 [-1, 1]）
        if self.normalize_actions:
            action = 2.0 * (action - self.action_min) / self.action_range - 1.0
        
        return {
            'images': torch.FloatTensor(image_sequence),  # (seq_len, H, W, 3)
            'action': torch.FloatTensor(action)    # (action_dim,)
        }


def compute_info_nce_loss(
    energy_positives: torch.Tensor,  # (B, 1) 正样本能量
    energy_negatives: torch.Tensor,  # (B, num_negatives) 负样本能量
    temperature: float = 1.0
) -> torch.Tensor:
    """
    计算 InfoNCE 损失
    """
    energies = torch.cat([energy_negatives, energy_positives], dim=1)  # (B, num_negatives + 1)
    probs = F.softmax(-energies / temperature, dim=-1)
    num_negatives = energy_negatives.size(1)
    labels = torch.zeros(energies.size(0), dtype=torch.long, device=energies.device)
    labels.fill_(num_negatives)
    loss = F.cross_entropy(-energies / temperature, labels, reduction='mean')
    return loss


def compute_gradient_penalty(
    model: nn.Module,
    images: torch.Tensor,  # (B, 3*seq_len, H, W)
    actions: torch.Tensor,  # (B, num_samples, action_dim) 或 (B*num_samples, action_dim)
    grad_margin: float = 1.0,
    grad_norm_type: str = 'inf',
    square_grad_penalty: bool = True,
    grad_loss_weight: float = 1.0
) -> torch.Tensor:
    """
    计算梯度惩罚损失（支持 Late Fusion）
    """
    # 确保 actions 需要梯度
    if actions.dim() == 3:
        B, num_samples, action_dim = actions.shape
        actions_flat = actions.view(-1, action_dim)
        obs_encoding = model.encode(images)  # (B, 256)
        obs_encoding_expanded = obs_encoding.unsqueeze(1).expand(B, num_samples, -1)
        obs_encoding_flat = obs_encoding_expanded.reshape(-1, obs_encoding.size(1))
    else:
        actions_flat = actions
        B = images.size(0)
        num_samples = actions_flat.size(0) // B
        obs_encoding = model.encode(images)  # (B, 256)
        obs_encoding_expanded = obs_encoding.unsqueeze(1).expand(B, num_samples, -1)
        obs_encoding_flat = obs_encoding_expanded.reshape(-1, obs_encoding.size(1))
    
    # 确保 actions 需要梯度
    actions_flat = actions_flat.detach().requires_grad_(True)
    
    # 计算能量对动作的梯度（使用 Late Fusion）
    energies = model.value_network(
        torch.cat([obs_encoding_flat, actions_flat], dim=-1)
    ).squeeze(-1)  # (B*num_samples,)
    
    # 计算梯度：dE/da
    grad_outputs = torch.ones_like(energies)
    de_dact = torch.autograd.grad(
        outputs=energies,
        inputs=actions_flat,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]  # (B*num_samples, action_dim)
    
    # 计算梯度范数
    if grad_norm_type == 'inf':
        grad_norms = torch.norm(de_dact, p=float('inf'), dim=1)
    elif grad_norm_type == '1':
        grad_norms = torch.norm(de_dact, p=1, dim=1)
    elif grad_norm_type == '2':
        grad_norms = torch.norm(de_dact, p=2, dim=1)
    else:
        raise ValueError(f"Unsupported grad_norm_type: {grad_norm_type}")
    
    # 重塑为 (B, num_samples)
    grad_norms = grad_norms.view(B, -1)
    
    # 应用 margin
    if grad_margin is not None:
        grad_norms = grad_norms - grad_margin
        grad_norms = torch.clamp(grad_norms, min=0.0, max=1e10)
    
    # 平方（如果启用）
    if square_grad_penalty:
        grad_norms = grad_norms ** 2
    
    # 平均
    grad_loss = grad_norms.mean()
    
    return grad_loss * grad_loss_weight


def train_pusht_ebm(
    zarr_path: str,
    output_dir: str,
    obs_seq_len: int = 2,
    target_height: int = 96,   # PushT 图像大小
    target_width: int = 96,    # PushT 图像大小
    value_width: int = 1024,
    value_num_blocks: int = 1,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    num_iterations: int = 100000,
    num_counter_examples: int = 8,
    ula_step_size: float = 0.1,
    ula_num_steps: int = 100,
    ula_noise_scale: float = 1.0,
    ula_step_size_final: float = 1e-5,
    ula_step_size_power: float = 2.0,
    temperature: float = 1.0,
    lr_decay_steps: int = 100,
    lr_decay_rate: float = 0.99,
    add_grad_penalty: bool = True,
    grad_margin: float = 1.0,
    grad_norm_type: str = 'inf',
    square_grad_penalty: bool = True,
    grad_loss_weight: float = 1.0,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    save_interval: int = 5000,
    eval_interval: int = 5000,
    eval_episodes: int = 20,
    use_wandb: bool = True,
    wandb_project: str = 'ibc-pusht_pixel',
    wandb_name: str = None,
    wandb_tags: list = None,
    wandb_mode: str = None
):
    """
    训练 PushT (RGB 图像) EBM 模型
    """
    print("=" * 60)
    print("训练 PushT (RGB 图像) EBM 模型")
    print("=" * 60)
    print(f"数据路径: {zarr_path}")
    print(f"输出目录: {output_dir}")
    print(f"设备: {device}")
    print()
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_path / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    # 创建 TensorBoard 日志目录
    log_dir = output_path / 'logs'
    log_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard 日志目录: {log_dir}")
    print(f"查看日志: tensorboard --logdir {log_dir}")
    print()
    
    # 初始化 wandb
    wandb_initialized = False
    if use_wandb:
        try:
            wandb_name = wandb_name or output_path.name
            wandb_mode = wandb_mode or os.getenv('WANDB_MODE', 'online')
            
            wandb.init(
                project=wandb_project,
                name=wandb_name,
                tags=wandb_tags or [],
                mode=wandb_mode,
                config={
                    'zarr_path': zarr_path,
                    'output_dir': str(output_dir),
                    'obs_seq_len': obs_seq_len,
                    'target_height': target_height,
                    'target_width': target_width,
                    'value_width': value_width,
                    'value_num_blocks': value_num_blocks,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'num_iterations': num_iterations,
                    'num_counter_examples': num_counter_examples,
                    'ula_step_size': ula_step_size,
                    'ula_num_steps': ula_num_steps,
                    'ula_noise_scale': ula_noise_scale,
                    'ula_step_size_final': ula_step_size_final,
                    'ula_step_size_power': ula_step_size_power,
                    'temperature': temperature,
                    'lr_decay_steps': lr_decay_steps,
                    'lr_decay_rate': lr_decay_rate,
                    'add_grad_penalty': add_grad_penalty,
                    'grad_margin': grad_margin,
                    'grad_norm_type': grad_norm_type,
                    'square_grad_penalty': square_grad_penalty,
                    'grad_loss_weight': grad_loss_weight,
                    'device': device,
                    'save_interval': save_interval,
                    'eval_interval': eval_interval,
                    'eval_episodes': eval_episodes,
                }
            )
            
            wandb_initialized = True
            print(f"✓ Wandb 初始化成功")
            print(f"  项目: {wandb_project}")
            print(f"  运行名称: {wandb_name}")
            print(f"  模式: {wandb_mode}")
            if wandb.run:
                print(f"  运行 URL: {wandb.run.url}")
            print()
        except Exception as e:
            print(f"⚠ Wandb 初始化失败: {e}")
            print("  继续训练，但不记录到 wandb")
            use_wandb = False
            wandb_initialized = False
            print()
    
    # 加载数据集
    print("加载数据集...")
    dataset = PushTImageDataset(
        zarr_path, 
        obs_seq_len=obs_seq_len,
        target_height=target_height,
        target_width=target_width
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device == 'cuda' else False
    )
    
    # 获取动作维度
    action_dim = 2  # PushT 动作维度
    image_channels = 3 * obs_seq_len  # 3 * sequence_length
    
    print(f"图像通道数: {image_channels} (3 * sequence_length={obs_seq_len})")
    print(f"动作维度: {action_dim}")
    print(f"观测序列长度: {obs_seq_len}")
    print()
    
    # 创建模型
    print("创建模型...")
    model = PixelEBM(
        image_channels=image_channels,
        action_dim=action_dim,
        target_height=target_height,
        target_width=target_width,
        value_width=value_width,
        value_num_blocks=value_num_blocks
    ).to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-7)
    
    # 学习率调度器
    scheduler = TFExponentialDecay(
        optimizer,
        decay_rate=lr_decay_rate,
        decay_steps=lr_decay_steps,
        staircase=False
    )
    
    # 创建 ULA 采样器
    uniform_boundary_buffer = 0.05
    norm_min, norm_max = -1.0, 1.0
    expanded_min = norm_min - uniform_boundary_buffer * (norm_max - norm_min)
    expanded_max = norm_max + uniform_boundary_buffer * (norm_max - norm_min)
    action_bounds = np.array([[expanded_min, expanded_min],
                              [expanded_max, expanded_max]], dtype=np.float32)
    ula_sampler = ULASampler(
        bounds=action_bounds,
        step_size=ula_step_size,
        num_steps=ula_num_steps,
        noise_scale=ula_noise_scale,
        step_size_final=ula_step_size_final,
        step_size_power=ula_step_size_power,
        delta_action_clip=0.1,
        device=device
    )
    
    # 训练循环
    print("开始训练...")
    print()
    
    model.train()
    global_step = 0
    epoch = 0
    
    # 保存归一化参数
    norm_params = {
        'action_min': dataset.action_min.tolist() if dataset.action_min is not None else None,
        'action_max': dataset.action_max.tolist() if dataset.action_max is not None else None,
        'action_range': dataset.action_range.tolist() if dataset.action_range is not None else None,
        'action_dim': action_dim,
        'obs_seq_len': obs_seq_len,
        'target_height': target_height,
        'target_width': target_width,
        'image_channels': image_channels,
    }
    
    with open(output_path / 'norm_params.json', 'w') as f:
        json.dump(norm_params, f, indent=2)
    
    pbar = tqdm(total=num_iterations, desc="训练进度")
    
    while global_step < num_iterations:
        epoch += 1
        
        for batch in dataloader:
            if global_step >= num_iterations:
                break
            
            images = batch['images'].to(device)  # (B, seq_len, H, W, 3)
            actions = batch['action'].to(device)  # (B, action_dim)
            
            B = images.size(0)
            
            # Late Fusion: 先编码图像
            obs_encoding = model.encode(images)  # (B, 256)
            
            # 计算正样本能量
            actions_expanded = actions.unsqueeze(1)  # (B, 1, action_dim)
            energy_positives = model(
                images=None,
                actions=actions_expanded,
                obs_encoding=obs_encoding
            )  # (B, 1)
            
            # 生成负样本（使用 ULA）
            neg_min = float(action_bounds[0, 0])
            neg_max = float(action_bounds[1, 0])
            init_negatives = torch.rand(
                B, num_counter_examples, action_dim,
                device=device
            ) * (neg_max - neg_min) + neg_min
            
            # ULA 采样
            obs_encoding_detached = obs_encoding.detach()
            negatives, _ = ula_sampler.sample(
                x=images,
                ebm=model,
                num_samples=num_counter_examples,
                init_samples=init_negatives,
                return_trajectory=False,
                obs_encoding=obs_encoding_detached
            )
            
            # 计算负样本能量
            energy_negatives = model(
                images=None,
                actions=negatives,
                obs_encoding=obs_encoding
            )  # (B, num_counter_examples)
            
            # 计算 InfoNCE 损失
            info_nce_loss = compute_info_nce_loss(
                energy_positives,
                energy_negatives,
                temperature=temperature
            )
            
            # 计算梯度惩罚损失（如果启用）
            if add_grad_penalty:
                combined_actions = torch.cat([
                    negatives,
                    actions.unsqueeze(1)
                ], dim=1)
                
                grad_loss = compute_gradient_penalty(
                    model=model,
                    images=images,
                    actions=combined_actions,
                    grad_margin=grad_margin,
                    grad_norm_type=grad_norm_type,
                    square_grad_penalty=square_grad_penalty,
                    grad_loss_weight=grad_loss_weight
                )
                
                loss = info_nce_loss + grad_loss
            else:
                loss = info_nce_loss
                grad_loss = None
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            global_step += 1
            
            # 更新进度条
            pbar.update(1)
            postfix_dict = {
                'loss': f'{loss.item():.4f}',
                'info_nce': f'{info_nce_loss.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}'
            }
            if add_grad_penalty and grad_loss is not None:
                postfix_dict['grad_penalty'] = f'{grad_loss.item():.4f}'
            pbar.set_postfix(postfix_dict)
            
            # 记录到 TensorBoard 和 wandb
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            writer.add_scalar('Train/InfoNCE_Loss', info_nce_loss.item(), global_step)
            writer.add_scalar('Train/Learning_Rate', scheduler.get_last_lr()[0], global_step)
            if add_grad_penalty and grad_loss is not None:
                writer.add_scalar('Train/Gradient_Penalty', grad_loss.item(), global_step)
            
            if use_wandb and wandb_initialized:
                log_dict = {
                    'Train/Loss': loss.item(),
                    'Train/InfoNCE_Loss': info_nce_loss.item(),
                    'Train/Learning_Rate': scheduler.get_last_lr()[0],
                }
                if add_grad_penalty and grad_loss is not None:
                    log_dict['Train/Gradient_Penalty'] = grad_loss.item()
                wandb.log(log_dict, step=global_step)
            
            # 记录能量值
            if global_step % 100 == 0:
                with torch.no_grad():
                    energy_pos_mean = energy_positives.mean().item()
                    energy_neg_mean = energy_negatives.mean().item()
                    energy_pos_std = energy_positives.std().item()
                    energy_neg_std = energy_negatives.std().item()
                    
                    writer.add_scalar('Train/Energy_Positive_Mean', energy_pos_mean, global_step)
                    writer.add_scalar('Train/Energy_Negative_Mean', energy_neg_mean, global_step)
                    writer.add_scalar('Train/Energy_Positive_Std', energy_pos_std, global_step)
                    writer.add_scalar('Train/Energy_Negative_Std', energy_neg_std, global_step)
                    
                    if use_wandb and wandb_initialized:
                        wandb.log({
                            'Train/Energy_Positive_Mean': energy_pos_mean,
                            'Train/Energy_Negative_Mean': energy_neg_mean,
                            'Train/Energy_Positive_Std': energy_pos_std,
                            'Train/Energy_Negative_Std': energy_neg_std,
                        }, step=global_step)
            
            # 保存检查点
            if global_step % save_interval == 0:
                checkpoint = {
                    'global_step': global_step,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss.item(),
                    'action_dim': action_dim,
                    'obs_seq_len': obs_seq_len,
                    'target_height': target_height,
                    'target_width': target_width,
                    'image_channels': image_channels,
                    'value_width': value_width,
                    'value_num_blocks': value_num_blocks,
                    'norm_params': norm_params
                }
                checkpoint_path = checkpoint_dir / f'checkpoint_{global_step:06d}.pth'
                torch.save(checkpoint, checkpoint_path)
                print(f"\n保存检查点: {checkpoint_path}")
            
            # 评估
            if global_step % eval_interval == 0:
                model.eval()
                eval_losses = []
                for i, eval_batch in enumerate(dataloader):
                    if i >= eval_episodes:
                        break
                    eval_images = eval_batch['images'].to(device)
                    eval_actions = eval_batch['action'].to(device)
                    eval_B = eval_images.size(0)
                    
                    with torch.no_grad():
                        eval_obs_encoding = model.encode(eval_images)
                        eval_actions_expanded = eval_actions.unsqueeze(1)
                        eval_energy_positives = model(
                            images=None,
                            actions=eval_actions_expanded,
                            obs_encoding=eval_obs_encoding
                        )
                    
                    eval_init_negatives = torch.rand(
                        eval_B, num_counter_examples, action_dim,
                        device=device
                    ) * (neg_max - neg_min) + neg_min
                    
                    eval_obs_encoding_detached = eval_obs_encoding.detach()
                    
                    with torch.enable_grad():
                        eval_negatives, _ = ula_sampler.sample(
                            x=eval_images,
                            ebm=model,
                            num_samples=num_counter_examples,
                            init_samples=eval_init_negatives,
                            return_trajectory=False,
                            obs_encoding=eval_obs_encoding_detached
                        )
                    
                    with torch.no_grad():
                        eval_negatives = eval_negatives.detach()
                        eval_energy_negatives = model(
                            images=None,
                            actions=eval_negatives,
                            obs_encoding=eval_obs_encoding
                        )
                        
                        eval_loss = compute_info_nce_loss(
                            eval_energy_positives,
                            eval_energy_negatives,
                            temperature=temperature
                        )
                        eval_losses.append(eval_loss.item())
                
                avg_eval_loss = np.mean(eval_losses)
                print(f"\n[评估] Step {global_step}, 平均损失: {avg_eval_loss:.4f}")
                
                writer.add_scalar('Eval/Loss', avg_eval_loss, global_step)
                if use_wandb and wandb_initialized:
                    wandb.log({'Eval/Loss': avg_eval_loss}, step=global_step)
                
                model.train()
    
    pbar.close()
    
    # 关闭 TensorBoard writer 和 wandb
    writer.close()
    print(f"\nTensorBoard 日志已保存到: {log_dir}")
    
    if use_wandb and wandb_initialized:
        if wandb.run:
            print(f"\nWandb 运行 URL: {wandb.run.url}")
        wandb.finish()
        print("Wandb 日志已上传")
    
    # 保存最终模型
    final_checkpoint = {
        'global_step': global_step,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'action_dim': action_dim,
        'obs_seq_len': obs_seq_len,
        'target_height': target_height,
        'target_width': target_width,
        'image_channels': image_channels,
        'value_width': value_width,
        'value_num_blocks': value_num_blocks,
        'norm_params': norm_params
    }
    final_path = checkpoint_dir / 'final_model.pth'
    torch.save(final_checkpoint, final_path)
    print(f"\n保存最终模型: {final_path}")
    print("训练完成！")


if __name__ == '__main__':
    import argparse
    
    IBC_ROOT = Path(__file__).parent.parent.parent  # IBC_ebm_dp
    
    parser = argparse.ArgumentParser(description='训练 PushT (RGB 图像) EBM 模型')
    parser.add_argument('--zarr_path', type=str, 
                       default=str(IBC_ROOT / 'data' / 'pusht_cchi_v7_replay.zarr'),
                       help='zarr 数据路径')
    parser.add_argument('--output_dir', type=str,
                       default=str(IBC_ROOT / 'models' / 'pusht_pixel'),
                       help='输出目录')
    parser.add_argument('--device', type=str, default=None,
                       help='计算设备 (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--num_iterations', type=int, default=20000,
                       help='训练迭代次数')
    parser.add_argument('--use_wandb', action='store_true', default=True,
                       help='使用 wandb 监控训练（默认启用）')
    parser.add_argument('--no_wandb', dest='use_wandb', action='store_false',
                       help='禁用 wandb 监控')
    parser.add_argument('--wandb_project', type=str, default='ibc-pusht_pixel',
                       help='wandb 项目名称')
    parser.add_argument('--wandb_name', type=str, default=None,
                       help='wandb 运行名称')
    parser.add_argument('--wandb_tags', type=str, nargs='+', default=None,
                       help='wandb 标签')
    parser.add_argument('--wandb_mode', type=str, default=None,
                       choices=['online', 'offline', 'disabled'],
                       help='wandb 模式')
    
    args = parser.parse_args()
    
    device = args.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train_pusht_ebm(
        zarr_path=args.zarr_path,
        output_dir=args.output_dir,
        obs_seq_len=2,
        target_height=96,
        target_width=96,
        value_width=1024,
        value_num_blocks=1,
        batch_size=args.batch_size,
        learning_rate=1e-3,
        num_iterations=args.num_iterations,
        num_counter_examples=8,
        ula_step_size=0.1,
        ula_num_steps=100,
        ula_noise_scale=1.0,
        ula_step_size_final=1e-5,
        ula_step_size_power=2.0,
        temperature=1.0,
        lr_decay_steps=100,
        lr_decay_rate=0.99,
        add_grad_penalty=True,
        grad_margin=1.0,
        grad_norm_type='inf',
        square_grad_penalty=True,
        grad_loss_weight=1.0,
        device=device,
        save_interval=5000,
        eval_interval=5000,
        eval_episodes=20,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        wandb_tags=args.wandb_tags,
        wandb_mode=args.wandb_mode
    )

