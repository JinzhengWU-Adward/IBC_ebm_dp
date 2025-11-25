"""
IBC_ebm_dp.core - 核心模块
包含可复用的模型组件、优化器和训练工具
"""

# 模型组件
from .models import (
    CoordConv,
    SpatialSoftArgmax,
    ResidualBlock,
    IBC_CNN,
    IBC_EBM,
    EnergyBasedModel,  # 别名
)

# 优化器
from .optimizers import (
    DerivativeFreeOptimizer,
    DerivativeFreeOptimizer1D,
    SGLDSampler,
)

__all__ = [
    # 模型组件
    'CoordConv',
    'SpatialSoftArgmax',
    'ResidualBlock',
    'IBC_CNN',
    'IBC_EBM',
    'EnergyBasedModel',
    # 优化器
    'DerivativeFreeOptimizer',
    'DerivativeFreeOptimizer1D',
    'SGLDSampler',
]

__version__ = '0.1.0'

