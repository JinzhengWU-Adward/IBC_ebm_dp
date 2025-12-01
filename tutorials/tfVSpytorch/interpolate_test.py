"""
验证 TensorFlow tf.image.resize 和 PyTorch F.interpolate 的差异

从数学角度 + 实际库实现 + 默认参数 + 边界行为 + 对齐策略 5 个层面进行验证
使用 IBC_ebm_dp/data/pushing_pixel 下的实际图像数据
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import os

# 抑制 TensorFlow 的 INFO 和 WARNING 日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warnings, 3=errors

# 尝试导入 TensorFlow（如果可用）
try:
    import tensorflow as tf
    TF_AVAILABLE = True
    TF_VERSION = tf.__version__
    print(f"✓ TensorFlow 可用 (版本: {TF_VERSION})")
    
    # TensorFlow 2.x 的 tf.image.resize 不支持 align_corners 参数
    # 需要使用 tf.compat.v1.image.resize_images 来测试 align_corners
    # 或者使用 tf.image.resize（默认相当于 align_corners=False）
    def tf_resize(images, size, method='bilinear', align_corners=False):
        """
        TensorFlow resize 的兼容函数
        在 TF 2.x 中，tf.image.resize 默认相当于 align_corners=False
        如果要测试 align_corners=True，需要使用 compat.v1 API
        """
        if align_corners:
            # 使用 compat.v1 API 支持 align_corners=True
            return tf.compat.v1.image.resize_images(
                images, size, method=tf.compat.v1.image.ResizeMethod.BILINEAR,
                align_corners=True
            )
        else:
            # TF 2.x 的默认行为（half-pixel center，相当于 align_corners=False）
            return tf.image.resize(images, size, method=method)
            
except ImportError:
    TF_AVAILABLE = False
    TF_VERSION = None
    print("⚠ TensorFlow 不可用，将跳过 TF 相关测试")

# 添加项目路径
IBC_ROOT = Path(__file__).parent.parent
DATA_DIR = IBC_ROOT / "data" / "pushing_pixel"


def load_test_image():
    """从 pushing_pixel 数据目录加载一张测试图像"""
    # 查找第一个轨迹目录
    traj_dirs = sorted(DATA_DIR.glob("traj_*"))
    if not traj_dirs:
        raise FileNotFoundError(f"未找到轨迹数据目录: {DATA_DIR}")
    
    traj_dir = traj_dirs[0]
    image_path = traj_dir / "frame_0000.png"
    
    if not image_path.exists():
        raise FileNotFoundError(f"未找到图像文件: {image_path}")
    
    # 加载图像
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img, dtype=np.float32)  # (H, W, 3), [0, 255]
    
    print(f"✓ 加载测试图像: {image_path}")
    print(f"  原始尺寸: {img_array.shape}")
    
    return img_array, image_path


def test_align_corners_effect():
    """测试 align_corners 参数的影响"""
    print("\n" + "="*80)
    print("测试 1: align_corners 参数的影响")
    print("="*80)
    
    # 创建一个简单的测试图像（4x4）
    test_img = np.arange(16, dtype=np.float32).reshape(4, 4, 1)  # (4, 4, 1)
    test_img = np.repeat(test_img, 3, axis=2)  # (4, 4, 3)
    test_img = test_img / 16.0  # 归一化到 [0, 1]
    
    print(f"输入图像尺寸: {test_img.shape}")
    print(f"输入图像值范围: [{test_img.min():.3f}, {test_img.max():.3f}]")
    
    target_size = (8, 8)  # 放大 2 倍
    
    # PyTorch: align_corners=True
    img_torch = torch.FloatTensor(test_img).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 4, 4)
    resized_torch_true = F.interpolate(
        img_torch, size=target_size, mode='bilinear', align_corners=True
    ).squeeze(0).permute(1, 2, 0).numpy()
    
    # PyTorch: align_corners=False
    resized_torch_false = F.interpolate(
        img_torch, size=target_size, mode='bilinear', align_corners=False
    ).squeeze(0).permute(1, 2, 0).numpy()
    
    print(f"\nPyTorch align_corners=True  输出尺寸: {resized_torch_true.shape}")
    print(f"PyTorch align_corners=False 输出尺寸: {resized_torch_false.shape}")
    
    # 检查边界像素是否对齐
    print("\n边界像素对比（取第一个通道）:")
    print(f"输入边界: [{test_img[0, 0, 0]:.3f}, {test_img[0, -1, 0]:.3f}, "
          f"{test_img[-1, 0, 0]:.3f}, {test_img[-1, -1, 0]:.3f}]")
    print(f"Torch True  边界: [{resized_torch_true[0, 0, 0]:.3f}, "
          f"{resized_torch_true[0, -1, 0]:.3f}, {resized_torch_true[-1, 0, 0]:.3f}, "
          f"{resized_torch_true[-1, -1, 0]:.3f}]")
    print(f"Torch False 边界: [{resized_torch_false[0, 0, 0]:.3f}, "
          f"{resized_torch_false[0, -1, 0]:.3f}, {resized_torch_false[-1, 0, 0]:.3f}, "
          f"{resized_torch_false[-1, -1, 0]:.3f}]")
    
    if TF_AVAILABLE:
        # TensorFlow: align_corners=True
        img_tf = tf.constant(test_img, dtype=tf.float32)  # (4, 4, 3)
        resized_tf_true = tf_resize(
            img_tf, target_size, method='bilinear', align_corners=True
        ).numpy()
        
        # TensorFlow: align_corners=False（默认行为）
        resized_tf_false = tf_resize(
            img_tf, target_size, method='bilinear', align_corners=False
        ).numpy()
        
        print(f"\nTF True  边界: [{resized_tf_true[0, 0, 0]:.3f}, "
              f"{resized_tf_true[0, -1, 0]:.3f}, {resized_tf_true[-1, 0, 0]:.3f}, "
              f"{resized_tf_true[-1, -1, 0]:.3f}]")
        print(f"TF False 边界: [{resized_tf_false[0, 0, 0]:.3f}, "
              f"{resized_tf_false[0, -1, 0]:.3f}, {resized_tf_false[-1, 0, 0]:.3f}, "
              f"{resized_tf_false[-1, -1, 0]:.3f}]")
        
        # 计算差异
        diff_tf_torch_false = np.abs(resized_tf_false - resized_torch_false)
        print(f"\nTF vs PyTorch (align_corners=False) 最大差异: {diff_tf_torch_false.max():.6f}")
        print(f"TF vs PyTorch (align_corners=False) 平均差异: {diff_tf_torch_false.mean():.6f}")
    
    return {
        'input': test_img,
        'torch_true': resized_torch_true,
        'torch_false': resized_torch_false,
        'tf_true': resized_tf_true if TF_AVAILABLE else None,
        'tf_false': resized_tf_false if TF_AVAILABLE else None,
    }


def test_coordinate_mapping():
    """测试坐标映射的数学差异"""
    print("\n" + "="*80)
    print("测试 2: 坐标映射的数学差异")
    print("="*80)
    
    # 创建一个简单的测试图像（4x4 -> 8x8）
    H_in, W_in = 4, 4
    H_out, W_out = 8, 8
    
    # 创建一个有特征的图像（中心有一个亮点）
    test_img = np.zeros((H_in, W_in, 3), dtype=np.float32)
    test_img[2, 2, :] = 1.0  # 中心像素为 1
    
    print(f"输入图像尺寸: {test_img.shape}")
    print(f"输出图像尺寸: ({H_out}, {W_out})")
    
    # PyTorch: align_corners=False
    img_torch = torch.FloatTensor(test_img).permute(2, 0, 1).unsqueeze(0)
    resized_torch = F.interpolate(
        img_torch, size=(H_out, W_out), mode='bilinear', align_corners=False
    ).squeeze(0).permute(1, 2, 0).numpy()
    
    # 找到输出图像中最大值的位置
    max_pos_torch = np.unravel_index(np.argmax(resized_torch[:, :, 0]), resized_torch.shape[:2])
    print(f"\nPyTorch (align_corners=False) 最大值位置: {max_pos_torch}")
    print(f"  理论中心位置应该是 (4, 4)，实际: {max_pos_torch}")
    
    if TF_AVAILABLE:
        # TensorFlow: align_corners=False（默认行为）
        img_tf = tf.constant(test_img, dtype=tf.float32)
        resized_tf = tf_resize(
            img_tf, (H_out, W_out), method='bilinear', align_corners=False
        ).numpy()
        
        max_pos_tf = np.unravel_index(np.argmax(resized_tf[:, :, 0]), resized_tf.shape[:2])
        print(f"\nTensorFlow (align_corners=False) 最大值位置: {max_pos_tf}")
        print(f"  理论中心位置应该是 (4, 4)，实际: {max_pos_tf}")
        
        # 计算位置差异
        pos_diff = np.array(max_pos_torch) - np.array(max_pos_tf)
        print(f"\n位置差异: {pos_diff}")
        print(f"  这反映了坐标映射的差异（half-pixel center vs scale-factor）")
    
    return {
        'input': test_img,
        'torch': resized_torch,
        'tf': resized_tf if TF_AVAILABLE else None,
    }


def test_real_image_resize():
    """使用真实图像测试 resize 差异"""
    print("\n" + "="*80)
    print("测试 3: 使用真实图像测试 resize 差异")
    print("="*80)
    
    # 加载真实图像
    img_array, img_path = load_test_image()
    H_orig, W_orig = img_array.shape[:2]
    
    # 归一化到 [0, 1]
    img_norm = img_array / 255.0
    
    # 目标尺寸（匹配 IBC 的配置）
    target_height, target_width = 180, 240
    
    print(f"原始尺寸: ({H_orig}, {W_orig})")
    print(f"目标尺寸: ({target_height}, {target_width})")
    
    # PyTorch: align_corners=False（默认）
    img_torch = torch.FloatTensor(img_norm).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    resized_torch = F.interpolate(
        img_torch, size=(target_height, target_width), mode='bilinear', align_corners=False
    ).squeeze(0).permute(1, 2, 0).numpy()
    
    print(f"PyTorch 输出尺寸: {resized_torch.shape}")
    print(f"PyTorch 输出值范围: [{resized_torch.min():.3f}, {resized_torch.max():.3f}]")
    
    if TF_AVAILABLE:
        # TensorFlow: 使用与 IBC 官方完全相同的调用方式
        # IBC 官方代码: tf.image.resize(images, [target_height, target_width])
        # 不指定 method 参数，使用默认的 'bilinear'
        img_tf = tf.constant(img_norm, dtype=tf.float32)  # (H, W, 3)
        
        # 方式1: 完全匹配 IBC 官方的调用（不指定 method）
        resized_tf_ibc = tf.image.resize(img_tf, [target_height, target_width]).numpy()
        
        # 方式2: 使用我们的 tf_resize 函数（显式指定 method='bilinear'）
        resized_tf_explicit = tf_resize(
            img_tf, (target_height, target_width), method='bilinear', align_corners=False
        ).numpy()
        
        # 验证两种方式是否一致
        diff_methods = np.abs(resized_tf_ibc - resized_tf_explicit)
        print(f"\nTensorFlow 两种调用方式对比:")
        print(f"  IBC 官方方式（不指定 method）输出尺寸: {resized_tf_ibc.shape}")
        print(f"  显式指定 method='bilinear' 输出尺寸: {resized_tf_explicit.shape}")
        print(f"  两种方式的最大差异: {diff_methods.max():.6f}")
        print(f"  两种方式是否完全一致: {np.allclose(resized_tf_ibc, resized_tf_explicit, atol=1e-6)}")
        
        # 使用 IBC 官方方式的结果进行后续对比
        resized_tf = resized_tf_ibc
        
        print(f"\nTensorFlow (IBC 官方方式) 输出尺寸: {resized_tf.shape}")
        print(f"TensorFlow 输出值范围: [{resized_tf.min():.3f}, {resized_tf.max():.3f}]")
        
        # 计算差异
        diff = np.abs(resized_tf - resized_torch)
        print(f"\nTF (IBC 官方方式) vs PyTorch 差异统计:")
        print(f"  最大差异: {diff.max():.6f}")
        print(f"  平均差异: {diff.mean():.6f}")
        print(f"  标准差: {diff.std():.6f}")
        print(f"  差异 > 0.01 的像素比例: {(diff > 0.01).sum() / diff.size * 100:.2f}%")
        print(f"  差异 > 0.05 的像素比例: {(diff > 0.05).sum() / diff.size * 100:.2f}%")
        
        # 计算相对误差
        relative_error = diff / (np.abs(resized_tf) + 1e-8)
        print(f"\n相对误差统计:")
        print(f"  最大相对误差: {relative_error.max():.6f}")
        print(f"  平均相对误差: {relative_error.mean():.6f}")
    
    return {
        'original': img_norm,
        'torch': resized_torch,
        'tf': resized_tf if TF_AVAILABLE else None,
        'diff': diff if TF_AVAILABLE else None,
    }


def test_boundary_behavior():
    """测试边界外行为（extrapolation）"""
    print("\n" + "="*80)
    print("测试 4: 边界外行为（extrapolation）")
    print("="*80)
    
    # 创建一个测试图像，边缘有明显的值
    test_img = np.zeros((10, 10, 3), dtype=np.float32)
    test_img[0, :, :] = 1.0  # 上边缘
    test_img[-1, :, :] = 1.0  # 下边缘
    test_img[:, 0, :] = 1.0  # 左边缘
    test_img[:, -1, :] = 1.0  # 右边缘
    
    print(f"输入图像尺寸: {test_img.shape}")
    print(f"边缘像素值: 1.0，中心像素值: 0.0")
    
    # 放大到 20x20
    target_size = (20, 20)
    
    # PyTorch
    img_torch = torch.FloatTensor(test_img).permute(2, 0, 1).unsqueeze(0)
    resized_torch = F.interpolate(
        img_torch, size=target_size, mode='bilinear', align_corners=False
    ).squeeze(0).permute(1, 2, 0).numpy()
    
    print(f"\nPyTorch 输出尺寸: {resized_torch.shape}")
    print(f"PyTorch 边缘像素值（取第一个通道）:")
    print(f"  上边缘: {resized_torch[0, :5, 0]}")
    print(f"  下边缘: {resized_torch[-1, :5, 0]}")
    print(f"  左边缘: {resized_torch[:5, 0, 0]}")
    print(f"  右边缘: {resized_torch[:5, -1, 0]}")
    
    if TF_AVAILABLE:
        # TensorFlow（默认行为，相当于 align_corners=False）
        img_tf = tf.constant(test_img, dtype=tf.float32)
        resized_tf = tf_resize(
            img_tf, target_size, method='bilinear', align_corners=False
        ).numpy()
        
        print(f"\nTensorFlow 输出尺寸: {resized_tf.shape}")
        print(f"TensorFlow 边缘像素值（取第一个通道）:")
        print(f"  上边缘: {resized_tf[0, :5, 0]}")
        print(f"  下边缘: {resized_tf[-1, :5, 0]}")
        print(f"  左边缘: {resized_tf[:5, 0, 0]}")
        print(f"  右边缘: {resized_tf[:5, -1, 0]}")
        
        # 计算边缘差异
        edge_diff = np.abs(resized_tf - resized_torch)
        print(f"\n边缘差异统计:")
        print(f"  最大差异: {edge_diff.max():.6f}")
        print(f"  边缘区域平均差异: {edge_diff[[0, -1], :, :].mean():.6f}")
    
    return {
        'input': test_img,
        'torch': resized_torch,
        'tf': resized_tf if TF_AVAILABLE else None,
    }


def test_scale_factor_vs_size():
    """测试 scale_factor vs size 的行为差异"""
    print("\n" + "="*80)
    print("测试 5: scale_factor vs size 的行为差异")
    print("="*80)
    
    # 创建测试图像
    test_img = np.random.rand(100, 100, 3).astype(np.float32)
    
    # 使用 size
    target_size = (200, 200)
    img_torch = torch.FloatTensor(test_img).permute(2, 0, 1).unsqueeze(0)
    resized_size = F.interpolate(
        img_torch, size=target_size, mode='bilinear', align_corners=False
    ).squeeze(0).permute(1, 2, 0).numpy()
    
    # 使用 scale_factor
    scale_factor = 2.0
    resized_scale = F.interpolate(
        img_torch, scale_factor=scale_factor, mode='bilinear', align_corners=False
    ).squeeze(0).permute(1, 2, 0).numpy()
    
    print(f"输入尺寸: {test_img.shape}")
    print(f"使用 size={target_size} 输出: {resized_size.shape}")
    print(f"使用 scale_factor={scale_factor} 输出: {resized_scale.shape}")
    
    # 检查是否一致
    diff = np.abs(resized_size - resized_scale)
    print(f"\n两种方式差异:")
    print(f"  最大差异: {diff.max():.6f}")
    print(f"  平均差异: {diff.mean():.6f}")
    print(f"  是否完全一致: {np.allclose(resized_size, resized_scale, atol=1e-6)}")
    
    return {
        'input': test_img,
        'size': resized_size,
        'scale': resized_scale,
        'diff': diff,
    }


def visualize_comparison(results_real):
    """可视化真实图像的 resize 结果对比"""
    if not TF_AVAILABLE or results_real['tf'] is None:
        print("\n⚠ 跳过可视化（TensorFlow 不可用）")
        return
    
    print("\n" + "="*80)
    print("可视化对比")
    print("="*80)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 原始图像（缩小显示）
    orig = results_real['original']
    if orig.shape[0] > 300 or orig.shape[1] > 400:
        # 如果太大，先缩小显示
        from PIL import Image
        orig_display = np.array(Image.fromarray((orig * 255).astype(np.uint8)).resize(
            (400, 300), Image.BILINEAR)) / 255.0
    else:
        orig_display = orig
    
    axes[0, 0].imshow(orig_display)
    axes[0, 0].set_title(f'Original Image\n{orig.shape[:2]}')
    axes[0, 0].axis('off')
    
    # PyTorch 结果
    axes[0, 1].imshow(results_real['torch'])
    axes[0, 1].set_title(f'PyTorch F.interpolate\nalign_corners=False')
    axes[0, 1].axis('off')
    
    # TensorFlow 结果
    axes[0, 2].imshow(results_real['tf'])
    axes[0, 2].set_title(f'TensorFlow tf.image.resize\nalign_corners=False')
    axes[0, 2].axis('off')
    
    # 差异图
    diff = results_real['diff']
    im = axes[1, 0].imshow(diff.max(axis=2), cmap='hot', vmin=0, vmax=diff.max())
    axes[1, 0].set_title('Absolute Difference')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0])
    
    # 差异直方图
    axes[1, 1].hist(diff.flatten(), bins=50, edgecolor='black')
    axes[1, 1].set_title('Difference Distribution')
    axes[1, 1].set_xlabel('Difference Value')
    axes[1, 1].set_ylabel('Frequency')
    
    # 相对误差图
    relative_error = diff / (np.abs(results_real['tf']) + 1e-8)
    im = axes[1, 2].imshow(relative_error.max(axis=2), cmap='hot', vmin=0, vmax=0.1)
    axes[1, 2].set_title('Relative Error')
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2])
    
    plt.tight_layout()
    
    # 保存图像
    output_path = IBC_ROOT / "tutorials" / "interpolate_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ 可视化结果已保存到: {output_path}")
    
    plt.close()


def main():
    """主函数：运行所有测试"""
    print("="*80)
    print("TensorFlow vs PyTorch Resize 差异验证")
    print("="*80)
    print(f"数据目录: {DATA_DIR}")
    print(f"TensorFlow 可用: {TF_AVAILABLE}")
    print(f"PyTorch 版本: {torch.__version__}")
    if TF_AVAILABLE:
        print(f"TensorFlow 版本: {TF_VERSION}")
        print(f"注意: TensorFlow 2.x 的 tf.image.resize 默认相当于 align_corners=False")
        print(f"      使用 tf.compat.v1.image.resize_images 来测试 align_corners=True")
        print(f"\nIBC 官方调用方式:")
        print(f"  tf.image.resize(images, [target_height, target_width])")
        print(f"  不指定 method 参数，使用默认的 'bilinear'")
        print(f"  默认行为相当于 align_corners=False (half-pixel center)")
    
    # 运行所有测试
    results = {}
    
    try:
        results['align_corners'] = test_align_corners_effect()
    except Exception as e:
        print(f"❌ 测试 1 失败: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        results['coordinate_mapping'] = test_coordinate_mapping()
    except Exception as e:
        print(f"❌ 测试 2 失败: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        results['real_image'] = test_real_image_resize()
        if TF_AVAILABLE:
            visualize_comparison(results['real_image'])
    except Exception as e:
        print(f"❌ 测试 3 失败: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        results['boundary'] = test_boundary_behavior()
    except Exception as e:
        print(f"❌ 测试 4 失败: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        results['scale_factor'] = test_scale_factor_vs_size()
    except Exception as e:
        print(f"❌ 测试 5 失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("所有测试完成！")
    print("="*80)
    
    # # 总结
    # print("\n📊 总结:")
    # print("1. align_corners 参数影响边界像素的对齐方式")
    # print("2. TensorFlow 和 PyTorch 在 align_corners=False 时仍有坐标映射差异")
    # print("3. TensorFlow 使用 half-pixel center，PyTorch 使用 scale-factor based")
    # print("4. 这种差异在视觉任务中可能导致像素级偏移")
    # print("5. 对于 CNN 特征图，PyTorch 的插值更符合卷积语义")
    # print("6. 对于真实图像处理，TensorFlow 更接近 OpenCV 标准")


if __name__ == "__main__":
    main()

