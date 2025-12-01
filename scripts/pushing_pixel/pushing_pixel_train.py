"""
使用 PyTorch 复刻 IBC 的 Block Pushing (pixel/RGB 图像) 环境训练
基于 run_pixel_ebm_langevin.sh 和 pixel_ebm_langevin.gin 的配置
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
from PIL import Image
import wandb


# 添加项目路径
IBC_ROOT = Path(__file__).parent.parent.parent  # IBC_ebm_dp
sys.path.insert(0, str(IBC_ROOT))

from core.models import PixelEBM
from core.optimizers import ULASampler, TFExponentialDecay


class PushingPixelDataset(Dataset):
    """
    Block Pushing (RGB 图像) 环境数据集
    从 JSON 文件和 PNG 图像加载轨迹数据
    """
    
    def __init__(
        self,
        data_dir: str,
        obs_seq_len: int = 2,
        normalize_actions: bool = True,
        target_height: int = 180,  # 匹配 gin: PixelEBM.target_height = 180
        target_width: int = 240,    # 匹配 gin: PixelEBM.target_width = 240
    ):
        """
        Args:
            data_dir: 数据目录，包含 traj_*/ 子目录
            obs_seq_len: 观测序列长度（默认 2，匹配 gin: train_eval.sequence_length = 2）
            normalize_actions: 是否归一化动作到 [-1, 1]
            target_height: 目标图像高度（resize 后）
            target_width: 目标图像宽度（resize 后）
        """
        self.data_dir = Path(data_dir)
        self.obs_seq_len = obs_seq_len
        self.normalize_actions = normalize_actions
        self.target_height = target_height
        self.target_width = target_width
        
        # 查找所有轨迹目录
        traj_dirs = sorted(self.data_dir.glob('traj_*'))
        self.traj_dirs = [d for d in traj_dirs if (d / 'traj.json').exists()]
        
        print(f"找到 {len(self.traj_dirs)} 个轨迹目录")
        
        # 加载所有数据并计算归一化参数
        self.episodes = []
        all_actions = []
        
        for traj_dir in tqdm(self.traj_dirs, desc="加载数据"):
            json_path = traj_dir / 'traj.json'
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # 获取图像路径和动作
            image_paths = data['trajectory']['image_paths']
            actions = np.array(data['trajectory']['actions'], dtype=np.float32)
            
            # 构建完整图像路径
            full_image_paths = [self.data_dir / path for path in image_paths]
            
            num_images = len(full_image_paths)
            num_actions = len(actions)
            
            # 数据格式说明：
            # - 图像序列：[img_0, img_1, ..., img_N] (N+1 个图像)
            # - 动作序列：[action_0, action_1, ..., action_N] (N+1 个动作)
            # - action_i 对应 img_i 的动作
            #   - action_0 到 action_{N-1} 是实际执行的动作（从 img_i 到 img_{i+1}）
            #   - action_N 是"停留"动作（保持在 img_N 的位置，通常是 [0, 0]）
            # - 观测序列 [img_i, img_{i+1}, ..., img_{i+obs_seq_len-1}] 对应动作 action_{i+obs_seq_len-1}
            #   由于每个观测都有对应的动作，action_idx 总是有效的
            
            # 确保 images 和 actions 数量一致
            if num_images != num_actions:
                print(f"警告: 轨迹 {traj_dir} 的图像数量 ({num_images}) 和动作数量 ({num_actions}) 不匹配，跳过")
                continue
            
            # 构建观测序列样本
            # ===== 首帧重复的样本：[img0, img0] -> action0 =====
            if num_images >= self.obs_seq_len and num_actions > 0:
                first_img_path = full_image_paths[0]
                first_action = actions[0]
                
                self.episodes.append({
                    'image_paths': [first_img_path] * self.obs_seq_len,
                    'action': first_action.astype(np.float32)
                })
                all_actions.append(first_action)
            
            # ===== 滑动窗口样本 =====
            # 对于窗口 [i, i+1, ..., i+obs_seq_len-1]，对应的动作是 action_{i+obs_seq_len-1}
            # 由于每个观测都有对应的动作，action_idx 总是 < num_actions
            # 注意：忽略最后一个动作（停留动作，通常是 [0, 0]），用于实验对比
            for i in range(num_images):
                # 确保窗口不超出图像范围
                if i + self.obs_seq_len <= num_images:
                    # 构建图像序列路径
                    img_seq_paths = []
                    for j in range(self.obs_seq_len):
                        idx = i + j
                        if idx < num_images:
                            img_seq_paths.append(full_image_paths[idx])
                    
                    if len(img_seq_paths) == self.obs_seq_len:
                        # 标签使用窗口最后一步对应的动作
                        # action_idx = i + obs_seq_len - 1 对应 img_{i+obs_seq_len-1} 的动作
                        action_idx = i + self.obs_seq_len - 1
                        
                        # # 跳过最后一个动作（停留动作），用于实验对比
                        # if action_idx >= num_actions - 1:
                        #     continue
                        
                        # 由于每个观测都有对应的动作，action_idx 总是有效的
                        assert action_idx < num_actions, \
                            f"动作索引 {action_idx} 超出范围 (num_actions={num_actions})"
                        
                        action = actions[action_idx]
                        
                        self.episodes.append({
                            'image_paths': img_seq_paths,
                            'action': action.astype(np.float32)
                        })
                        all_actions.append(action)
        
        print(f"共加载 {len(self.episodes)} 个样本")
        
        # 检查是否有数据
        if len(self.episodes) == 0:
            raise ValueError(
                f"未找到任何训练数据！请检查数据目录: {self.data_dir}\n"
                f"确保存在 traj_*/ 目录，每个目录包含 traj.json 和 frame_*.png 文件。\n"
                f"可以使用 pushing_pixel_data_generate.py 生成数据。"
            )
        
        # 计算动作归一化参数
        all_actions = np.array(all_actions)
        
        if self.normalize_actions:
            # Min-Max 归一化到 [-1, 1]
            self.action_min = all_actions.min(axis=0)
            self.action_max = all_actions.max(axis=0)
            self.action_range = self.action_max - self.action_min
            self.action_range[self.action_range < 1e-6] = 1.0  # 避免除零
            
            print(f"动作范围: min={self.action_min}, max={self.action_max}")
        else:
            self.action_min = None
            self.action_max = None
            self.action_range = None
    
    def __len__(self):
        return len(self.episodes)
    
    def __getitem__(self, idx):
        episode = self.episodes[idx]
        image_paths = episode['image_paths']
        action = episode['action'].copy()
        
        # ⚠️ 关键修复：匹配 IBC 官方流程
        # IBC 官方在数据加载时只归一化，不进行 resize 和堆叠
        # resize 和堆叠在模型内部的 encode 方法中动态执行（GPU 上）
        
        # 加载图像序列（只归一化，保持原始尺寸和序列格式）
        images = []
        for img_path in image_paths:
            img = Image.open(img_path).convert('RGB')
            # 转换到 [0, 1] 范围（匹配 IBC: tf.image.convert_image_dtype）
            img_array = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
            images.append(img_array)
        
        # 转换为序列格式 (seq_len, H, W, 3)，保持原始尺寸
        # 注意：不进行堆叠和 resize，这些操作在模型内部执行
        image_sequence = np.stack(images, axis=0)  # (seq_len, H_orig, W_orig, 3)
        
        # 归一化动作（Min-Max 到 [-1, 1]）
        if self.normalize_actions:
            action = 2.0 * (action - self.action_min) / self.action_range - 1.0
        
        return {
            'images': torch.FloatTensor(image_sequence),  # (seq_len, H_orig, W_orig, 3)
            'action': torch.FloatTensor(action)    # (action_dim,)
        }


def compute_info_nce_loss(
    energy_positives: torch.Tensor,  # (B, 1) 正样本能量
    energy_negatives: torch.Tensor,  # (B, num_negatives) 负样本能量
    temperature: float = 1.0
) -> torch.Tensor:
    """
    计算 InfoNCE 损失（与 pushing_states_train.py 相同）
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


def train_pushing_pixel_ebm(
    data_dir: str,
    output_dir: str,
    obs_seq_len: int = 2,
    target_height: int = 180,  # 匹配 gin: PixelEBM.target_height = 180
    target_width: int = 240,   # 匹配 gin: PixelEBM.target_width = 240
    value_width: int = 1024,   # 匹配 gin: DenseResnetValue.width = 1024
    value_num_blocks: int = 1, # 匹配 gin: DenseResnetValue.num_blocks = 1
    batch_size: int = 32,     # 匹配 gin: train_eval.batch_size = 32
    learning_rate: float = 1e-3,  # 匹配 gin: train_eval.learning_rate = 1e-3
    num_iterations: int = 100000,  # 匹配 gin: train_eval.num_iterations = 100000
    num_counter_examples: int = 8,  # 匹配 gin: ImplicitBCAgent.num_counter_examples = 8
    ula_step_size: float = 0.1,
    ula_num_steps: int = 100,  # 匹配 gin: langevin_actions_given_obs.num_iterations = 100
    ula_noise_scale: float = 1.0,
    ula_step_size_final: float = 1e-5,
    ula_step_size_power: float = 2.0,
    temperature: float = 1.0,
    lr_decay_steps: int = 100,
    lr_decay_rate: float = 0.99,
    add_grad_penalty: bool = True,  # 匹配 gin: grad_penalty.grad_margin = 1.0
    grad_margin: float = 1.0,
    grad_norm_type: str = 'inf',
    square_grad_penalty: bool = True,
    grad_loss_weight: float = 1.0,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    save_interval: int = 5000,
    eval_interval: int = 5000,  # 匹配 gin: train_eval.eval_interval = 5000
    eval_episodes: int = 20,  # 匹配 gin: train_eval.eval_episodes = 20
    use_wandb: bool = True,  # 是否使用 wandb
    wandb_project: str = 'ibc-pushing-pixel',  # wandb 项目名称
    wandb_name: str = None,  # wandb 运行名称（如果为 None，则使用输出目录名）
    wandb_tags: list = None,  # wandb 标签
    wandb_mode: str = None  # wandb 模式: 'online'(需要API key), 'offline'(本地保存), 'disabled'(禁用)
):
    """
    训练 Block Pushing (RGB 图像) EBM 模型
    
    参数与 IBC 的 pixel_ebm_langevin.gin 配置对齐
    """
    print("=" * 60)
    print("训练 Block Pushing (RGB 图像) EBM 模型")
    print("=" * 60)
    print(f"数据目录: {data_dir}")
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
            # 支持通过参数或环境变量设置 wandb 模式
            wandb_mode = wandb_mode or os.getenv('WANDB_MODE', 'online')
            
            wandb.init(
                project=wandb_project,
                name=wandb_name,
                tags=wandb_tags or [],
                mode=wandb_mode,  # 'online'(需要API key), 'offline'(本地保存), 'disabled'(禁用)
                config={
                    'data_dir': data_dir,
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
                print(f"  实体: {wandb.run.entity}")
            print()
        except Exception as e:
            print(f"⚠ Wandb 初始化失败: {e}")
            print("  继续训练，但不记录到 wandb")
            import traceback
            traceback.print_exc()
            use_wandb = False
            wandb_initialized = False
            print()
    
    # 加载数据集
    print("加载数据集...")
    dataset = PushingPixelDataset(
        data_dir, 
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
    action_dim = dataset.episodes[0]['action'].shape[-1]  # 动作维度（2）
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
    uniform_boundary_buffer = 0.05  # 匹配 gin: train_eval.uniform_boundary_buffer = 0.05
    norm_min, norm_max = -1.0, 1.0
    # 修复：匹配 IBC 官方公式，不要多乘 2.0
    # IBC: buffered = min/max ± uniform_boundary_buffer * range_size
    # 其中 range_size = max - min = 2.0
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
            
            images = batch['images'].to(device)  # (B, seq_len, H_orig, W_orig, 3)
            actions = batch['action'].to(device)  # (B, action_dim)
            
            B = images.size(0)
            
            # Late Fusion: 先编码图像（在模型内部进行堆叠和 resize，匹配 IBC 官方）
            obs_encoding = model.encode(images)  # (B, 256)
            
            # 计算正样本能量
            actions_expanded = actions.unsqueeze(1)  # (B, 1, action_dim)
            energy_positives = model(
                images=None,  # 不使用 images，使用 obs_encoding
                actions=actions_expanded,
                obs_encoding=obs_encoding
            )  # (B, 1)
            
            # 生成负样本（使用 ULA，支持 Late Fusion）
            neg_min = float(action_bounds[0, 0])
            neg_max = float(action_bounds[1, 0])
            init_negatives = torch.rand(
                B, num_counter_examples, action_dim,
                device=device
            ) * (neg_max - neg_min) + neg_min
            
            # ULA 采样（使用 ULASampler，支持 Late Fusion）
            # 断开 obs_encoding 的梯度，只优化 actions（匹配 IBC 的 stop_chain_grad=True）
            obs_encoding_detached = obs_encoding.detach()
            negatives, _ = ula_sampler.sample(
                x=images,  # 传入 images（Late Fusion 模式下会被忽略）
                ebm=model,
                num_samples=num_counter_examples,
                init_samples=init_negatives,
                return_trajectory=False,
                obs_encoding=obs_encoding_detached  # Late Fusion: 预编码的观测特征
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
                    negatives,  # (B, num_counter_examples, action_dim)
                    actions.unsqueeze(1)  # (B, 1, action_dim)
                ], dim=1)  # (B, num_counter_examples + 1, action_dim)
                
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
                wandb.log({
                    'Train/Loss': loss.item(),
                    'Train/InfoNCE_Loss': info_nce_loss.item(),
                    'Train/Learning_Rate': scheduler.get_last_lr()[0],
                }, step=global_step)
                if add_grad_penalty and grad_loss is not None:
                    wandb.log({'Train/Gradient_Penalty': grad_loss.item()}, step=global_step)
            
            # 记录能量值（可选，用于调试）
            if global_step % 100 == 0:  # 每 100 步记录一次，避免日志过大
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
                    
                    # 编码图像（不需要梯度）
                    with torch.no_grad():
                        eval_obs_encoding = model.encode(eval_images)
                        eval_actions_expanded = eval_actions.unsqueeze(1)
                        eval_energy_positives = model(
                            images=None,
                            actions=eval_actions_expanded,
                            obs_encoding=eval_obs_encoding
                        )
                    
                    # ULA 采样需要梯度（用于计算能量对动作的梯度）
                    # 但不需要更新模型参数，所以使用 torch.enable_grad() 但保持 model.eval()
                    eval_init_negatives = torch.rand(
                        eval_B, num_counter_examples, action_dim,
                        device=device
                    ) * (neg_max - neg_min) + neg_min
                    
                    eval_obs_encoding_detached = eval_obs_encoding.detach()
                    
                    # ULA 采样（使用 ULASampler，支持 Late Fusion）
                    with torch.enable_grad():
                        eval_negatives, _ = ula_sampler.sample(
                            x=eval_images,  # 传入 images（Late Fusion 模式下会被忽略）
                            ebm=model,
                            num_samples=num_counter_examples,
                            init_samples=eval_init_negatives,
                            return_trajectory=False,
                            obs_encoding=eval_obs_encoding_detached  # Late Fusion: 预编码的观测特征
                        )
                    
                    # 计算最终能量和损失（不需要梯度）
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
                
                # 记录评估指标到 TensorBoard 和 wandb
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
    
    parser = argparse.ArgumentParser(description='训练 Block Pushing (RGB 图像) EBM 模型')
    parser.add_argument('--data_dir', type=str, 
                       default=str(IBC_ROOT / 'data' / 'pushing_pixel'),
                       help='数据目录')
    parser.add_argument('--output_dir', type=str,
                       default=str(IBC_ROOT / 'models' / 'pushing_pixel'),
                       help='输出目录')
    parser.add_argument('--device', type=str, default=None,
                       help='计算设备 (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小（匹配 gin: train_eval.batch_size = 32）')
    parser.add_argument('--num_iterations', type=int, default=10000,
                       help='训练迭代次数')
    parser.add_argument('--use_wandb', action='store_true', default=True,
                       help='使用 wandb 监控训练（默认启用）')
    parser.add_argument('--no_wandb', dest='use_wandb', action='store_false',
                       help='禁用 wandb 监控')
    parser.add_argument('--wandb_project', type=str, default='ibc-pushing-pixel',
                       help='wandb 项目名称')
    parser.add_argument('--wandb_name', type=str, default=None,
                       help='wandb 运行名称（如果为 None，则使用输出目录名）')
    parser.add_argument('--wandb_tags', type=str, nargs='+', default=None,
                       help='wandb 标签')
    parser.add_argument('--wandb_mode', type=str, default=None,
                       choices=['online', 'offline', 'disabled'],
                       help='wandb 模式: online(需要API key), offline(本地保存), disabled(禁用)')
    
    args = parser.parse_args()
    
    device = args.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train_pushing_pixel_ebm(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        obs_seq_len=2,  # 匹配 gin: train_eval.sequence_length = 2
        target_height=180,  # 匹配 gin: PixelEBM.target_height = 180
        target_width=240,   # 匹配 gin: PixelEBM.target_width = 240
        value_width=1024,  # 匹配 gin: DenseResnetValue.width = 1024
        value_num_blocks=1, # 匹配 gin: DenseResnetValue.num_blocks = 1
        batch_size=args.batch_size,
        learning_rate=1e-3,  # 匹配 gin: train_eval.learning_rate = 1e-3
        num_iterations=args.num_iterations,
        num_counter_examples=8,  # 匹配 gin: ImplicitBCAgent.num_counter_examples = 8
        ula_step_size=0.1,
        ula_num_steps=100,  # 匹配 gin: langevin_actions_given_obs.num_iterations = 100
        ula_noise_scale=1.0,
        ula_step_size_final=1e-5,
        ula_step_size_power=2.0,
        temperature=1.0,
        lr_decay_steps=100,
        lr_decay_rate=0.99,
        add_grad_penalty=True,  # 匹配 gin: grad_penalty.grad_margin = 1.0
        grad_margin=1.0,
        grad_norm_type='inf',
        square_grad_penalty=True,
        grad_loss_weight=1.0,
        device=device,
        save_interval=5000,
        eval_interval=5000,  # 匹配 gin: train_eval.eval_interval = 5000
        eval_episodes=20,  # 匹配 gin: train_eval.eval_episodes = 20
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        wandb_tags=args.wandb_tags,
        wandb_mode=args.wandb_mode
    )

