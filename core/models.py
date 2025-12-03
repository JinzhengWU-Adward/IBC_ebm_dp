"""
IBC 风格的 EBM 模型组件
包含 CoordConv、SpatialSoftArgmax、ResidualBlock、IBC_CNN 和 IBC_EBM
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CoordConv(nn.Module):
    """
    坐标卷积层：在输入特征图上添加归一化的坐标通道
    
    这有助于网络学习空间位置信息，特别适用于需要位置感知的任务。
    坐标通道的值在 [-1, 1] 范围内归一化。
    
    输入形状: (B, C, H, W)
    输出形状: (B, C+2, H, W)  # 增加了 y_coords 和 x_coords 两个通道
    
    参考: An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution
    """
    
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：在输入张量上添加坐标通道
        
        Args:
            x: 输入张量，形状为 (B, C, H, W)
            
        Returns:
            添加了坐标通道的张量，形状为 (B, C+2, H, W)
        """
        batch_size, _, image_height, image_width = x.size()
        
        # 生成 y 坐标（垂直方向），范围 [-1, 1]
        y_coords = (
            2.0
            * torch.arange(image_height, device=x.device).unsqueeze(1).expand(image_height, image_width)
            / (image_height - 1.0)
            - 1.0
        )
        
        # 生成 x 坐标（水平方向），范围 [-1, 1]
        x_coords = (
            2.0
            * torch.arange(image_width, device=x.device).unsqueeze(0).expand(image_height, image_width)
            / (image_width - 1.0)
            - 1.0
        )
        
        # 堆叠坐标并扩展到 batch 维度
        coords = torch.stack((y_coords, x_coords), dim=0)
        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1)
        
        # 拼接坐标通道到输入特征
        x = torch.cat((coords, x), dim=1)
        return x


class SpatialSoftArgmax(nn.Module):
    """
    空间软最大值层：通过 softmax 权重保留空间位置信息
    
    将特征图的每个通道转换为该通道的"期望位置"坐标，
    使用 softmax 作为权重来计算加权平均位置。
    
    输入形状: (B, C, H, W)
    输出形状: (B, C*2)  # 每个通道输出 (x, y) 坐标
    
    参考: End-to-End Training of Deep Visuomotor Policies
    """
    
    def __init__(self, normalize: bool = True) -> None:
        """
        Args:
            normalize: 是否归一化坐标到 [-1, 1] 范围
                      如果为 False，则坐标范围是 [0, W-1] 和 [0, H-1]
        """
        super().__init__()
        self.normalize = normalize

    def _coord_grid(
        self,
        h: int,
        w: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        生成坐标网格
        
        Args:
            h: 特征图高度
            w: 特征图宽度
            device: 设备类型
            
        Returns:
            坐标网格，形状为 (2, H, W)
        """
        if self.normalize:
            # 归一化坐标到 [-1, 1]
            return torch.stack(
                torch.meshgrid(
                    torch.linspace(-1, 1, w, device=device),
                    torch.linspace(-1, 1, h, device=device),
                    indexing="ij",
                )
            )
        # 非归一化坐标 [0, W-1] 和 [0, H-1]
        return torch.stack(
            torch.meshgrid(
                torch.arange(0, w, device=device),
                torch.arange(0, h, device=device),
                indexing="ij",
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：计算每个通道的期望位置
        
        Args:
            x: 输入张量，形状为 (B, C, H, W)
            
        Returns:
            每个通道的期望位置坐标，形状为 (B, C*2)
        """
        assert x.ndim == 4, "Expecting a tensor of shape (B, C, H, W)."
        
        _, c, h, w = x.shape
        
        # 对每个通道的空间位置应用 softmax
        softmax = F.softmax(x.view(-1, h * w), dim=-1)  # (B*C, H*W)
        
        # 获取坐标网格
        xc, yc = self._coord_grid(h, w, x.device)  # (H, W), (H, W)
        
        # 计算期望位置（加权平均）
        x_mean = (softmax * xc.flatten()).sum(dim=1, keepdims=True)  # (B*C, 1)
        y_mean = (softmax * yc.flatten()).sum(dim=1, keepdims=True)  # (B*C, 1)
        
        # 拼接 x 和 y 坐标
        return torch.cat([x_mean, y_mean], dim=1).view(-1, c * 2)


class ResidualBlock(nn.Module):
    """
    残差块：带有跳跃连接的卷积块
    
    结构: x -> ReLU -> Conv -> ReLU -> Conv -> (+x) -> output
    
    参考: Deep Residual Learning for Image Recognition (ResNet)
    """
    
    def __init__(self, depth: int, activation_fn=nn.ReLU):
        """
        Args:
            depth: 通道数（输入和输出通道数相同）
            activation_fn: 激活函数类（默认为 ReLU）
        """
        super().__init__()
        self.conv1 = nn.Conv2d(depth, depth, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(depth, depth, 3, padding=1, bias=True)
        self.activation = activation_fn()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (B, depth, H, W)
            
        Returns:
            输出张量，形状为 (B, depth, H, W)
        """
        out = self.activation(x)
        out = self.conv1(out)
        out = self.activation(out)  # 修复：这里应该是 out 而不是 x
        out = self.conv2(out)
        return out + x


class IBC_CNN(nn.Module):
    """
    IBC 风格的卷积神经网络：使用残差块构建的 CNN 骨架
    
    结构: [Conv -> ResidualBlock] * len(blocks)
    每个块会改变通道数，但保持空间尺寸不变（padding=1）
    
    参考: Implicit Behavioral Cloning (IBC)
    """
    
    def __init__(
        self, 
        in_channels: int = 1, 
        blocks: list = None, 
        activation_fn=nn.ReLU
    ):
        """
        Args:
            in_channels: 输入通道数
            blocks: 每个残差块的输出通道数列表，例如 [16, 32, 32]
            activation_fn: 激活函数类（默认为 ReLU）
        """
        super().__init__()
        if blocks is None:
            blocks = [16, 32, 32]
        
        depth_in = in_channels
        layers = []
        for depth_out in blocks:
            layers.extend([
                nn.Conv2d(depth_in, depth_out, 3, padding=1),
                ResidualBlock(depth_out, activation_fn),
            ])
            depth_in = depth_out
        
        self.net = nn.Sequential(*layers)
        self.activation = activation_fn()
    
    def forward(self, x: torch.Tensor, activate: bool = False) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (B, in_channels, H, W)
            activate: 是否在输出时应用激活函数
            
        Returns:
            特征张量，形状为 (B, blocks[-1], H, W)
        """
        out = self.net(x)
        if activate:
            return self.activation(out)
        return out


class IBC_EBM(nn.Module):
    """
    IBC 风格的能量基模型（Energy-Based Model）
    
    整合了 CoordConv、IBC_CNN、SpatialSoftArgmax 和 MLP。
    用于条件能量建模：E(x, y)，其中 x 是图像，y 是候选坐标。
    
    架构流程:
    1. [可选] CoordConv: 添加坐标通道
    2. IBC_CNN: 提取图像特征
    3. 1x1 Conv: 降维到 feature_channels
    4. SpatialSoftArgmax: 保留空间位置信息
    5. MLP: 融合图像特征和候选坐标，输出能量值
    
    参考: Implicit Behavioral Cloning (IBC)
    """
    
    def __init__(
        self, 
        in_channels: int = 1, 
        cnn_blocks: list = None,
        feature_channels: int = 16, 
        hidden_dim: int = 256, 
        use_coord_conv: bool = True,
        activation_fn=nn.ReLU
    ):
        """
        Args:
            in_channels: 输入图像通道数（例如 1 表示灰度图，3 表示 RGB）
            cnn_blocks: CNN 每个块的通道数列表，例如 [16, 32, 32]
            feature_channels: 1x1 卷积后的特征通道数
            hidden_dim: MLP 的隐藏层维度
            use_coord_conv: 是否使用 CoordConv
            activation_fn: 激活函数类（默认为 ReLU）
        """
        super().__init__()
        if cnn_blocks is None:
            cnn_blocks = [16, 32, 32]
        
        self.use_coord_conv = use_coord_conv
        
        # CoordConv（可选）
        if use_coord_conv:
            self.coord_conv = CoordConv()
            # CoordConv 添加了 2 个通道（x 和 y 坐标）
            cnn_in_channels = in_channels + 2
        else:
            cnn_in_channels = in_channels
        
        # CNN 骨架
        self.cnn = IBC_CNN(cnn_in_channels, cnn_blocks, activation_fn)
        
        # 1x1 卷积降维
        self.conv_1x1 = nn.Conv2d(cnn_blocks[-1], feature_channels, 1)
        
        # SpatialSoftArgmax：输出 feature_channels * 2 维（每个通道的 x, y 坐标）
        self.spatial_softargmax = SpatialSoftArgmax(normalize=True)
        
        # MLP：输入是 (feature_channels * 2 + action_dim)，输出是能量值
        # feature_channels * 2 来自 SpatialSoftArgmax，+2 是候选坐标 y（假设是 2D）
        mlp_input_dim = feature_channels * 2 + 2  # 这里硬编码为 2D 坐标
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            activation_fn(),
            nn.Linear(hidden_dim, hidden_dim),
            activation_fn(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        前向传播：计算给定图像和候选坐标的能量值
        
        Args:
            x: 图像张量，形状为 (B, C, H, W)
            y: 候选坐标张量，形状为 (B, N, action_dim)
               其中 N 是候选数量，action_dim 是动作维度（例如 2D 坐标）
               
        Returns:
            能量值张量，形状为 (B, N)
            能量值越低表示该坐标越可能是正确的预测
        """
        # CoordConv（可选）
        if self.use_coord_conv:
            x = self.coord_conv(x)
        
        # CNN 特征提取
        out = self.cnn(x, activate=True)  # (B, cnn_blocks[-1], H, W)
        
        # 1x1 卷积降维
        out = F.relu(self.conv_1x1(out))  # (B, feature_channels, H, W)
        
        # SpatialSoftArgmax：保留空间位置信息
        out = self.spatial_softargmax(out)  # (B, feature_channels * 2)
        
        # 特征融合
        B, feature_dim = out.shape
        N = y.shape[1]
        
        # 扩展图像特征以匹配候选数量
        features_expanded = out.unsqueeze(1).expand(B, N, feature_dim)  # (B, N, feature_dim)
        
        # 拼接图像特征和候选坐标
        fused = torch.cat([features_expanded, y], dim=-1)  # (B, N, feature_dim+action_dim)
        fused = fused.reshape(B * N, feature_dim + y.shape[-1])
        
        # 计算能量
        energy = self.mlp(fused)  # (B*N, 1)
        return energy.view(B, N)


# 为了向后兼容，提供别名
EnergyBasedModel = IBC_EBM


class MLPResidualBlock(nn.Module):
    """
    MLP残差块（Pre-Activation ResNet）
    
    完全匹配 IBC 官方的 ResNetPreActivationLayer 中的单个残差块
    
    结构（当 norm_type=None 时）:
      x -> ReLU -> Dropout -> Dense -> ReLU -> Dropout -> Dense -> (+x) -> output
    
    结构（当 norm_type='layer' 时）:
      x -> Norm -> ReLU -> Dropout -> Dense -> Norm -> ReLU -> Dropout -> Dense -> (+x) -> output
    """
    
    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1,
        norm_type: str = 'layer'  # 'layer', 'batch' 或 None
    ):
        """
        Args:
            hidden_dim: 隐藏层维度
            dropout: Dropout概率
            norm_type: 归一化类型 ('layer', 'batch' 或 None)
                      当为 None 时，完全跳过归一化层（匹配 IBC 的 normalizer=None）
        """
        super().__init__()
        
        self.norm_type = norm_type
        
        # 选择归一化层（匹配 IBC 的实现）
        # IBC 中：if self.normalizer is None: pass (不创建 norm 层)
        # 当 normalizer=None 时，norm 层根本不存在，而不是 Identity
        if norm_type == 'layer':
            self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-6)  # 匹配 IBC 的 epsilon=1e-6
            self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-6)
        elif norm_type == 'batch':
            self.norm1 = nn.BatchNorm1d(hidden_dim)
            self.norm2 = nn.BatchNorm1d(hidden_dim)
        elif norm_type is None or norm_type.lower() == 'none':
            # 不创建 norm 层，在 forward 中直接跳过
            self.norm1 = None
            self.norm2 = None
        else:
            raise ValueError(f"不支持的归一化类型: {norm_type}")
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
        # 注意：权重初始化在 SequenceEBM._init_weights() 中统一完成
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播（Pre-Activation 顺序，完全匹配 IBC 官方）
        
        IBC 官方顺序（ResNetPreActivationLayer.call）:
          1. if normalizer is not None: Norm
          2. ReLU
          3. Dropout
          4. Dense
          5. if normalizer is not None: Norm
          6. ReLU
          7. Dropout
          8. Dense
          9. 残差连接
        
        Args:
            x: 输入张量，形状为 (B, hidden_dim)
            
        Returns:
            输出张量，形状为 (B, hidden_dim)
        """
        identity = x
        
        # 第一层：Pre-Activation 顺序
        # 1. Norm (如果存在)
        if self.norm1 is not None:
            out = self.norm1(x)
        else:
            out = x
        # 2. ReLU
        out = self.activation(out)
        # 3. Dropout
        out = self.dropout1(out)
        # 4. Dense
        out = self.fc1(out)
        
        # 第二层：Pre-Activation 顺序
        # 1. Norm (如果存在)
        if self.norm2 is not None:
            out = self.norm2(out)
        # 2. ReLU
        out = self.activation(out)
        # 3. Dropout
        out = self.dropout2(out)
        # 4. Dense
        out = self.fc2(out)
        
        # 残差连接
        return out + identity


class SequenceEBM(nn.Module):
    """
    基于序列观测的能量基模型（使用MLP + 残差块）
    
    完全匹配 IBC 官方的 MLPEBM 架构
    
    输入: observation序列 + action候选
    架构（匹配 IBC 的 MLPEBM）:
      1. 展平观测序列: [B, T, obs_dim] -> [B, T*obs_dim]
      2. 拼接 [obs, act]: [B, T*obs_dim + action_dim]
      3. ResNetPreActivationLayer（内部包含投影层和残差块）:
         - 投影层: 将输入投影到 hidden_dim
         - 残差块: Pre-Activation ResNet 块
      4. 能量投影层: Dense(1)
    
    输出: 标量能量值
    
    注意：
      - IBC 的 depth=2 意味着 hidden_sizes=[width, width]，对应 1 个残差块
      - 投影层在 ResNetPreActivationLayer 内部，不在外部
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        obs_seq_len: int = 2,
        hidden_dim: int = 256,
        num_residual_blocks: int = 1,  # depth=2 意味着 1 个残差块
        dropout: float = 0.1,
        norm_type: str = 'layer'  # 'layer', 'batch' 或 None
    ):
        """
        Args:
            obs_dim: 单个observation的维度
            action_dim: action的维度
            obs_seq_len: observation序列长度
            hidden_dim: 隐藏层维度（对应 IBC 的 width）
            num_residual_blocks: 残差块数量（对应 IBC 的 depth/2）
                                 IBC 的 depth=2 意味着 1 个残差块
            dropout: Dropout概率（对应 IBC 的 rate）
            norm_type: 归一化类型 ('layer', 'batch' 或 None)
                      对应 IBC 的 ResNetLayer.normalizer
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.obs_seq_len = obs_seq_len
        self.hidden_dim = hidden_dim
        
        # 输入维度: obs_seq_len * obs_dim + action_dim
        input_dim = obs_seq_len * obs_dim + action_dim
        
        # 投影层（匹配 IBC 的 ResNetPreActivationLayer._projection_layer）
        # IBC 中投影层在 ResNetPreActivationLayer 内部，将输入投影到 hidden_sizes[0]
        self.projection = nn.Linear(input_dim, hidden_dim)
        
        # 残差块（匹配 IBC 的 ResNetPreActivationLayer 中的残差块）
        # IBC 的 depth=2 意味着 hidden_sizes=[width, width]，对应 1 个残差块
        self.residual_blocks = nn.ModuleList([
            MLPResidualBlock(hidden_dim, dropout, norm_type)
            for _ in range(num_residual_blocks)
        ])
        
        # 能量投影层（匹配 IBC 的 MLPEBM._project_energy）
        # 注意：IBC 中输出维度是 action_dim，但最后会 squeeze 成标量
        # 这里直接输出 1 维，等价于 squeeze
        self.energy_projection = nn.Linear(hidden_dim, 1)
        
        # 初始化权重（匹配 IBC 的 kernel_initializer='normal', bias_initializer='normal'）
        # TensorFlow 的 'normal' 初始化: RandomNormal(mean=0.0, stddev=0.05)
        self._init_weights()
    
    def _init_weights(self):
        """
        初始化网络权重，匹配 IBC 的初始化方式
        
        IBC 使用:
          kernel_initializer='normal': tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)
          bias_initializer='normal': tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 匹配 TensorFlow 的 normal 初始化
                nn.init.normal_(module.weight, mean=0.0, std=0.05)
                if module.bias is not None:
                    nn.init.normal_(module.bias, mean=0.0, std=0.05)
    
    def forward(self, obs_seq: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            obs_seq: observation序列，形状为 (B, obs_seq_len, obs_dim)
            action: action候选，形状为 (B, N, action_dim)
                   其中 N 是候选数量
        
        Returns:
            能量值，形状为 (B, N)
        """
        B = obs_seq.size(0)
        N = action.size(1)
        
        # 展平observation序列
        obs_flat = obs_seq.reshape(B, -1)  # (B, obs_seq_len * obs_dim)
        
        # 扩展observation以匹配action候选数量
        obs_expanded = obs_flat.unsqueeze(1).expand(B, N, -1)  # (B, N, obs_seq_len * obs_dim)
        
        # 拼接 [obs_seq, action]
        x = torch.cat([obs_expanded, action], dim=-1)  # (B, N, input_dim)
        x = x.reshape(B * N, -1)  # (B*N, input_dim)
        
        # 投影层
        x = self.projection(x)  # (B*N, hidden_dim)
        
        # 残差块
        for block in self.residual_blocks:
            x = block(x)  # (B*N, hidden_dim)
        
        # 能量投影
        energy = self.energy_projection(x)  # (B*N, 1)
        
        return energy.view(B, N)


class ConvMaxpoolEncoder(nn.Module):
    """
    ConvMaxpoolEncoder: 简单的 4 层 Conv2D + MaxPool2D CNN
    
    匹配 IBC 官方的 conv_maxpool.get_conv_maxpool 架构：
    - Conv2D(32) -> MaxPool2D -> Conv2D(64) -> MaxPool2D -> 
      Conv2D(128) -> MaxPool2D -> Conv2D(256) -> MaxPool2D -> GlobalAveragePooling2D
    
    输出: (B, 256) 特征向量
    """
    
    def __init__(self, in_channels: int = 3):
        """
        Args:
            in_channels: 输入通道数（例如 3*sequence_length 用于堆叠的图像序列）
        """
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # GlobalAveragePooling2D
        self.activation = nn.ReLU()
        
        # 初始化权重（匹配 IBC 的 'normal' 初始化）
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重，匹配 IBC 的 'normal' 初始化 (std=0.05)"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0.0, std=0.05)
                if module.bias is not None:
                    nn.init.normal_(module.bias, mean=0.0, std=0.05)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像，形状为 (B, C, H, W)
            
        Returns:
            特征向量，形状为 (B, 256)
        """
        x = self.activation(self.conv1(x))
        x = self.pool1(x)
        
        x = self.activation(self.conv2(x))
        x = self.pool2(x)
        
        x = self.activation(self.conv3(x))
        x = self.pool3(x)
        
        x = self.activation(self.conv4(x))
        x = self.pool4(x)
        
        # GlobalAveragePooling2D
        x = self.global_pool(x)  # (B, 256, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 256)
        
        return x


class ResNetDenseBlock(nn.Module):
    """
    Dense ResNet 块（匹配 IBC 的 ResNetDenseBlock）
    
    结构:
      y = ReLU -> Dense(width//4) -> ReLU -> Dense(width//4) -> 
          ReLU -> Dense(width) -> ReLU -> Dense(width)
      if x.shape != y.shape:
        x = ReLU -> Dense(width)
      return x + y
    """
    
    def __init__(self, width: int):
        """
        Args:
            width: 隐藏层维度
        """
        super().__init__()
        
        # 匹配 IBC 官方的 dense_resnet_value.py 中的 ResNetDenseBlock
        # dense(width // 4) 表示输出维度为 width // 4，输入维度自动推断
        # 在 TensorFlow 中，如果输入是 (B, width)，Dense(width // 4) 会创建 (width, width // 4) 的权重
        # 在 PyTorch 中，需要明确指定输入和输出维度
        self.dense0 = nn.Linear(width, width // 4)  # 从 width 降到 width // 4
        self.dense1 = nn.Linear(width // 4, width // 4)  # 保持 width // 4
        self.dense2 = nn.Linear(width // 4, width)  # 从 width // 4 升到 width
        self.dense3 = nn.Linear(width, width)  # 用于形状不匹配时的投影
        
        self.activation = nn.ReLU()
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重，匹配 IBC 的 'normal' 初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.05)
                if module.bias is not None:
                    nn.init.normal_(module.bias, mean=0.0, std=0.05)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播（匹配 IBC 官方实现）
        
        Args:
            x: 输入张量，形状为 (B, width)
            
        Returns:
            输出张量，形状为 (B, width)
        
        匹配 IBC 官方的 dense_resnet_value.py：
        y = dense0(activation0(x))
        y = dense1(activation1(y))
        y = dense2(activation2(y))
        if x.shape != y.shape:
            x = dense3(activation3(x))
        return x + y
        """
        identity = x
        
        # 匹配 IBC 官方：y = dense0(activation0(x))
        y = self.dense0(self.activation(x))
        # 匹配 IBC 官方：y = dense1(activation1(y))
        y = self.dense1(self.activation(y))
        # 匹配 IBC 官方：y = dense2(activation2(y))
        y = self.dense2(self.activation(y))
        
        # 如果形状不匹配，投影 identity（匹配 IBC 官方）
        if identity.shape != y.shape:
            identity = self.dense3(self.activation(identity))
        
        return identity + y


class DenseResnetValue(nn.Module):
    """
    Dense ResNet Value Network（匹配 IBC 的 DenseResnetValue）
    
    结构: Dense(width) -> ResNetDenseBlock * num_blocks -> Dense(1)
    
    用于计算能量值：E(obs_encoding, action)
    """
    
    def __init__(self, input_dim: int, width: int = 1024, num_blocks: int = 1):
        """
        Args:
            input_dim: 输入维度（obs_encoding_dim + action_dim）
            width: 隐藏层维度（默认 1024，匹配 gin 配置）
            num_blocks: ResNet 块数量（默认 1，匹配 gin 配置）
        """
        super().__init__()
        
        self.dense0 = nn.Linear(input_dim, width)
        self.blocks = nn.ModuleList([
            ResNetDenseBlock(width) for _ in range(num_blocks)
        ])
        self.dense1 = nn.Linear(width, 1)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重，匹配 IBC 的 'normal' 初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.05)
                if module.bias is not None:
                    nn.init.normal_(module.bias, mean=0.0, std=0.05)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (B*N, obs_encoding_dim + action_dim)
               其中 N 是 action 候选数量
               
        Returns:
            能量值，形状为 (B*N, 1)
        """
        x = self.dense0(x)  # 无激活（匹配 IBC）
        
        for block in self.blocks:
            x = block(x)
        
        x = self.dense1(x)  # 无激活（匹配 IBC）
        
        return x


class PixelEBM(nn.Module):
    """
    PixelEBM: 基于图像的 Energy-Based Model（匹配 IBC 官方架构）
    
    支持 Late Fusion：
    1. 先编码图像序列得到 obs_encoding
    2. 对每个 action 候选，拼接 [obs_encoding, action] 输入 value network
    3. 输出能量值 E(obs, action)
    
    架构:
    - Encoder: ConvMaxpoolEncoder (将图像序列编码为特征向量)
    - Value Network: DenseResnetValue (计算能量值)
    
    参考: IBC 官方的 PixelEBM 和 pixel_ebm_langevin.gin 配置
    """
    
    def __init__(
        self,
        image_channels: int = 6,  # 3 * sequence_length (默认 sequence_length=2)
        action_dim: int = 2,
        target_height: int = 180,  # 匹配 gin: PixelEBM.target_height = 180
        target_width: int = 240,   # 匹配 gin: PixelEBM.target_width = 240
        value_width: int = 1024,   # 匹配 gin: DenseResnetValue.width = 1024
        value_num_blocks: int = 1, # 匹配 gin: DenseResnetValue.num_blocks = 1
    ):
        """
        Args:
            image_channels: 输入图像通道数（3 * sequence_length，图像序列堆叠在通道维度）
            action_dim: 动作维度
            target_height: 目标图像高度（编码前会 resize 到此尺寸）
            target_width: 目标图像宽度
            value_width: Value Network 的隐藏层维度
            value_num_blocks: Value Network 的 ResNet 块数量
        """
        super().__init__()
        
        self.target_height = target_height
        self.target_width = target_width
        self.action_dim = action_dim  # 保存动作维度，供测试代码使用
        
        # Encoder: 将图像编码为特征向量
        self.encoder = ConvMaxpoolEncoder(in_channels=image_channels)
        obs_encoding_dim = 256  # ConvMaxpoolEncoder 输出维度
        
        # Value Network: 计算能量值 E(obs_encoding, action)
        value_input_dim = obs_encoding_dim + action_dim
        self.value_network = DenseResnetValue(
            input_dim=value_input_dim,
            width=value_width,
            num_blocks=value_num_blocks
        )
    
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        编码图像序列为特征向量（匹配 IBC 官方流程）
        
        Args:
            images: 输入图像，形状为 (B, seq_len, H, W, 3) 或 (seq_len, H, W, 3)
                   注意：这里是原始尺寸，未 resize 和堆叠
                   匹配 IBC: obs['rgb'] 的形状 (B, seq_len, H_orig, W_orig, 3)
                   
        Returns:
            特征向量，形状为 (B, 256)
        """
        # 处理单样本情况
        if images.dim() == 4:
            images = images.unsqueeze(0)  # (1, seq_len, H, W, 3)
        
        B, seq_len, H, W, C = images.shape
        
        # ⚠️ 关键修复：匹配 IBC 官方的 image_prepro.preprocess 流程
        # 1. 堆叠图像序列在通道维度（匹配 IBC: stack_images_channelwise）
        # TensorFlow 的 tf.reshape 是直接 reshape，不 permute
        # 从 (B, seq_len, H, W, C) 直接 reshape 为 (B, H, W, C*seq_len)
        # 这与 permute + reshape 的结果不同！
        images = images.reshape(B, H, W, seq_len * C)  # (B, H, W, 3*seq_len)
        
        # 2. 转换为 NCHW 格式以便使用 F.interpolate
        images = images.permute(0, 3, 1, 2)  # (B, 3*seq_len, H, W)
        
        # 3. Resize 到目标尺寸（在 GPU 上动态执行，匹配 IBC）
        # IBC 在模型内部进行 resize，每次前向传播时执行
        if images.shape[2] != self.target_height or images.shape[3] != self.target_width:
            images = F.interpolate(
                images,
                size=(self.target_height, self.target_width),
                mode='bilinear',
                align_corners=False
            )
        
        # 4. 编码
        obs_encoding = self.encoder(images)  # (B, 256)
        return obs_encoding
    
    def forward(
        self, 
        images: torch.Tensor, 
        actions: torch.Tensor,
        obs_encoding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播：计算能量值（支持 Late Fusion）
        
        Args:
            images: 输入图像，形状为 (B, C, H, W)
                   如果提供了 obs_encoding，此参数可忽略
            actions: 动作候选，形状为 (B, N, action_dim)
                    其中 N 是候选数量
            obs_encoding: 可选的预编码观测特征，形状为 (B, 256)
                         如果为 None，则从 images 编码
                         
        Returns:
            能量值，形状为 (B, N)
            能量值越低表示该动作越可能是正确的预测
        """
        # Late Fusion: 如果未提供 obs_encoding，先编码图像
        if obs_encoding is None:
            obs_encoding = self.encode(images)  # (B, 256)
        
        B = obs_encoding.size(0)
        N = actions.size(1)
        
        # 扩展 obs_encoding 以匹配 action 候选数量
        obs_encoding_expanded = obs_encoding.unsqueeze(1).expand(B, N, -1)  # (B, N, 256)
        
        # 拼接 [obs_encoding, action]
        fused = torch.cat([obs_encoding_expanded, actions], dim=-1)  # (B, N, 256 + action_dim)
        fused = fused.reshape(B * N, -1)  # (B*N, 256 + action_dim)
        
        # 计算能量值
        energy = self.value_network(fused)  # (B*N, 1)
        energy = energy.view(B, N)  # (B, N)
        
        return energy

