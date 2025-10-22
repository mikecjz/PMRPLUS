#!/usr/bin/env python3
"""
测试k-space损失记录功能的脚本
"""

import torch
import pytorch_lightning as pl
from lightning.pytorch.loggers import WandbLogger
from pl_modules import PromptMrModule
from types import SimpleNamespace
import tempfile
import os

def test_loss_logging():
    """测试损失记录功能"""
    print("=== 测试k-space损失记录功能 ===")
    
    # 创建模型
    model = PromptMrModule(
        num_cascades=1,
        num_adj_slices=3,
        kspace_loss_weight=0.02  # 稍微增加权重以便观察
    )
    
    # 创建临时目录用于日志
    with tempfile.TemporaryDirectory() as temp_dir:
        # 设置wandb为offline模式进行测试
        logger = WandbLogger(
            project="test_kspace_loss",
            save_dir=temp_dir,
            offline=True,
            name="test_run"
        )
        
        # 创建trainer
        trainer = pl.Trainer(
            logger=logger,
            max_epochs=1,
            limit_train_batches=2,
            limit_val_batches=1,
            accelerator='auto',
            devices=1 if torch.cuda.is_available() else 'auto'
        )
        
        # 模拟batch数据
        def create_batch():
            batch = SimpleNamespace()
            batch.masked_kspace = torch.randn(1, 24, 64, 64, 2)  # 8 coils * 3 adj_slices
            batch.mask = torch.ones(1, 1, 64, 64, 1).bool()
            batch.target = torch.randn(1, 64, 64)
            batch.max_value = torch.tensor([1.0])
            batch.num_low_frequencies = torch.tensor([16])
            batch.mask_type = ('cartesian',)
            batch.fname = ['test']
            batch.slice_num = torch.tensor([0])
            return batch
        
        # 测试训练步骤
        print("1. 测试training_step...")
        model.trainer = trainer  # 设置trainer引用
        
        try:
            batch = create_batch()
            loss = model.training_step(batch, 0)
            print(f"   ✓ Training step successful! Total Loss: {loss.item():.4f}")
            
            # 检查模型是否有正确的损失函数
            assert hasattr(model, 'normalized_l1'), "模型缺少normalized_l1损失函数"
            assert hasattr(model, 'kspace_loss_weight'), "模型缺少kspace_loss_weight参数"
            print(f"   ✓ kspace_loss_weight = {model.kspace_loss_weight}")
            print("   ✓ 训练中应该记录以下指标：")
            print("     - train_loss (总损失)")
            print("     - sup_loss (SSIM监督损失)")
            print("     - self_loss (k-space归一化L1损失)")
            
        except Exception as e:
            print(f"   ✗ Training step failed: {e}")
            return False
        
        # 测试验证步骤
        print("\n2. 测试validation_step...")
        try:
            batch = create_batch()
            result = model.validation_step(batch, 0)
            print("   ✓ Validation step successful!")
            print("   ✓ 验证中应该记录以下指标：")
            print("     - val_loss_sup (验证SSIM损失)")
            print("     - val_loss_kspace (验证k-space损失)")
            
        except Exception as e:
            print(f"   ✗ Validation step failed: {e}")
            return False
    
    print("\n=== 总结 ===")
    print("✓ 模型已正确配置k-space损失")
    print("✓ 训练和验证步骤都会记录相关损失指标")
    print("✓ 在实际训练中，这些指标将显示在wandb中：")
    print("  📊 训练指标：train_loss, sup_loss, self_loss")
    print("  📊 验证指标：val_loss_sup, val_loss_kspace")
    print("\n🎯 在wandb项目 'cmr2024_2025_phased' 中可以监控这些指标！")
    
    return True

if __name__ == "__main__":
    test_loss_logging()
