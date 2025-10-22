import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from .promptmr_v2 import PromptMR, PromptMRBlock, NormPromptUnet, SensitivityModel
from mri_utils import sens_reduce, sens_expand, rss, complex_abs, ifft2c, complex_mul


class ParallelPromptMR(PromptMR):
    """
    并行Cascade架构的PromptMR实现
    
    架构设计:
    Branch A: Cascade1_A → Cascade2_A → Cascade3_A → ... → CascadeN_A
    Branch B: Cascade1_B → Cascade2_B → Cascade3_B → ... → CascadeN_B
    
    Tensor传递路径:
    Cascade1_A → Cascade1_B → Cascade2_A → Cascade2_B → Cascade3_A → Cascade3_B → ...
    """
    
    def __init__(
        self,
        num_cascades: int,
        num_adj_slices: int,
        n_feat0: int,
        feature_dim: List[int],
        prompt_dim: List[int],
        sens_n_feat0: int,
        sens_feature_dim: List[int],
        sens_prompt_dim: List[int],
        len_prompt: List[int],
        prompt_size: List[int],
        n_enc_cab: List[int],
        n_dec_cab: List[int],
        n_skip_cab: List[int],
        n_bottleneck_cab: int,
        no_use_ca: bool = False,
        sens_len_prompt: Optional[List[int]] = None,
        sens_prompt_size: Optional[List[int]] = None,
        sens_n_enc_cab: Optional[List[int]] = None,
        sens_n_dec_cab: Optional[List[int]] = None,
        sens_n_skip_cab: Optional[List[int]] = None,
        sens_n_bottleneck_cab: Optional[List[int]] = None,
        sens_no_use_ca: Optional[bool] = None,
        mask_center: bool = True,
        learnable_prompt: bool = False,
        adaptive_input: bool = False,
        n_buffer: int = 4,
        n_history: int = 0,
        use_sens_adj: bool = True,
        parallel_mode: bool = True,  # 新增参数控制是否使用并行模式
        chain_mode: str = "AB",     # "AB" 或 "A_only"
        b_apply_dc: bool = True,
        b_model_scale: float = 1.0,
        b_use_buffer: bool = True,
        b_use_history: bool = True,
        ab_indices: Optional[List[int]] = None,
    ):
        # 调用父类初始化，但不创建cascades
        super(PromptMR, self).__init__()
        
        self.num_cascades = num_cascades
        self.num_adj_slices = num_adj_slices
        self.center_slice = num_adj_slices // 2
        self.n_buffer = n_buffer
        self.parallel_mode = parallel_mode
        self.chain_mode = chain_mode
        # normalize ab_indices: only allow 0..num_cascades-2 (can't run B after last A)
        raw_indices = ab_indices if ab_indices is not None else []
        norm_indices: List[int] = []
        for idx in raw_indices:
            try:
                iv = int(idx)
                if self.chain_mode == "AB":
                    allow_max = self.num_cascades - 2   # 0..N-2，保证以A收尾，步数=2N-1
                else:  # "A_only"
                    allow_max = self.num_cascades - 1   # 0..N-1，允许最后一个A后再跑B

                if 0 <= iv <= allow_max:
                    norm_indices.append(iv)
            except Exception:
                continue
        self.ab_indices = sorted(set(norm_indices))
        self._ab_index_set = set(self.ab_indices)
        if not getattr(self, "_ab_debug_printed", False):
            print(f"[ParallelPromptMR] chain_mode={self.chain_mode}, ab_indices={self.ab_indices}")
            self._ab_debug_printed = True
        self._debug_printed = False
        # 灵敏度图估计网络
        self.sens_net = SensitivityModel(
            num_adj_slices=num_adj_slices,
            n_feat0=sens_n_feat0,
            feature_dim=sens_feature_dim,
            prompt_dim=sens_prompt_dim,
            len_prompt=sens_len_prompt if sens_len_prompt is not None else len_prompt,
            prompt_size=sens_prompt_size if sens_prompt_size is not None else prompt_size,
            n_enc_cab=sens_n_enc_cab if sens_n_enc_cab is not None else n_enc_cab,
            n_dec_cab=sens_n_dec_cab if sens_n_dec_cab is not None else n_dec_cab,
            n_skip_cab=sens_n_skip_cab if sens_n_skip_cab is not None else n_skip_cab,
            n_bottleneck_cab=sens_n_bottleneck_cab if sens_n_bottleneck_cab is not None else n_bottleneck_cab,
            no_use_ca=sens_no_use_ca if sens_no_use_ca is not None else no_use_ca,
            mask_center=mask_center,
            learnable_prompt=learnable_prompt,
            use_sens_adj=use_sens_adj
        )
        
        if parallel_mode:
            # 创建两个并行的cascade分支
            self.cascades_a = nn.ModuleList([
                PromptMRBlock(
                    NormPromptUnet(
                        in_chans=2 * num_adj_slices,
                        out_chans=2 * num_adj_slices,
                        n_feat0=n_feat0,
                        feature_dim=feature_dim,
                        prompt_dim=prompt_dim,
                        len_prompt=len_prompt,
                        prompt_size=prompt_size,
                        n_enc_cab=n_enc_cab,
                        n_dec_cab=n_dec_cab,
                        n_skip_cab=n_skip_cab,
                        n_bottleneck_cab=n_bottleneck_cab,
                        no_use_ca=no_use_ca,
                        learnable_prompt=learnable_prompt,
                        adaptive_input=adaptive_input,
                        n_buffer=n_buffer,
                        n_history=n_history
                    ),
                    num_adj_slices=num_adj_slices
                ) for _ in range(num_cascades)
            ])
            
            self.cascades_b = nn.ModuleList([
                PromptMRBlock(
                    NormPromptUnet(
                        in_chans=2 * num_adj_slices,
                        out_chans=2 * num_adj_slices,
                        n_feat0=n_feat0,
                        feature_dim=feature_dim,
                        prompt_dim=prompt_dim,
                        len_prompt=len_prompt,
                        prompt_size=prompt_size,
                        n_enc_cab=n_enc_cab,
                        n_dec_cab=n_dec_cab,
                        n_skip_cab=n_skip_cab,
                        n_bottleneck_cab=n_bottleneck_cab,
                        no_use_ca=no_use_ca,
                        learnable_prompt=learnable_prompt,
                        adaptive_input=adaptive_input,
                        n_buffer=n_buffer,
                        n_history=n_history
                    ),
                    num_adj_slices=num_adj_slices
                ) for _ in range(num_cascades)
            ])
            
            # 配置B分支的DC与模型强度
            for m in self.cascades_b:
                m.apply_dc = b_apply_dc
                m.model_scale = float(b_model_scale)
                m.use_buffer = bool(b_use_buffer)
                m.use_history = bool(b_use_history)
        else:
            # 原始的单分支模式
            self.cascades = nn.ModuleList([
                PromptMRBlock(
                    NormPromptUnet(
                        in_chans=2 * num_adj_slices,
                        out_chans=2 * num_adj_slices,
                        n_feat0=n_feat0,
                        feature_dim=feature_dim,
                        prompt_dim=prompt_dim,
                        len_prompt=len_prompt,
                        prompt_size=prompt_size,
                        n_enc_cab=n_enc_cab,
                        n_dec_cab=n_dec_cab,
                        n_skip_cab=n_skip_cab,
                        n_bottleneck_cab=n_bottleneck_cab,
                        no_use_ca=no_use_ca,
                        learnable_prompt=learnable_prompt,
                        adaptive_input=adaptive_input,
                        n_buffer=n_buffer,
                        n_history=n_history
                    ),
                    num_adj_slices=num_adj_slices
                ) for _ in range(num_cascades)
            ])
    
    def copy_weights_from_original(self, original_model):
        """
        从原始PromptMR模型复制权重到并行架构
        
        Args:
            original_model: 原始的PromptMR模型实例
        """
        if not self.parallel_mode:
            raise ValueError("只有在parallel_mode=True时才能复制权重")
        
        print("正在复制权重到并行架构...")
        
        # 复制灵敏度图网络权重
        self.sens_net.load_state_dict(original_model.sens_net.state_dict())
        
        # 复制cascade权重到两个分支
        for i in range(self.num_cascades):
            # 复制到分支A
            self.cascades_a[i].load_state_dict(original_model.cascades[i].state_dict())
            # 复制到分支B  
            self.cascades_b[i].load_state_dict(original_model.cascades[i].state_dict())
        
        print(f"成功复制了{self.num_cascades}个cascade的权重到两个并行分支")
    
    def forward_parallel(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: torch.Tensor,
        mask_type: Tuple[str] = ("cartesian",),
        use_checkpoint: bool = False,
        compute_sens_per_coil: bool = False,
    ) -> torch.Tensor:
        """
        并行cascade的前向传播 (兼容promptmr_v2接口)
        
        Tensor传递路径:
        Cascade1_A → Cascade1_B → Cascade2_A → Cascade2_B → Cascade3_A → Cascade3_B → ...
        """
        # 计算灵敏度图
        if use_checkpoint:
            sens_maps = torch.utils.checkpoint.checkpoint(
                self.sens_net, masked_kspace, mask, num_low_frequencies, mask_type, compute_sens_per_coil,
                use_reentrant=False)
        else:
            sens_maps = self.sens_net(masked_kspace, mask, num_low_frequencies, mask_type, compute_sens_per_coil)

        img_zf = sens_reduce(masked_kspace, sens_maps, self.num_adj_slices)
        img_pred = img_zf.clone()
        latent = img_zf.clone()
        history_feat = None

        # 并行cascade处理
        total_steps = 0
        for ith in range(self.num_cascades):
            is_last = ith == self.num_cascades - 1
            
            # 分支A处理
            if use_checkpoint and self.training:
                img_pred, latent, history_feat = torch.utils.checkpoint.checkpoint(
                    self.cascades_a[ith], img_pred, img_zf, latent, mask, sens_maps, history_feat, 
                    use_reentrant=False)
            else:
                img_pred, latent, history_feat = self.cascades_a[ith](
                    img_pred, img_zf, latent, mask, sens_maps, history_feat)
            total_steps += 1
            
            # 分支B处理 (如果不是最后一个cascade)
            run_b_this_step = (
                (self.chain_mode == "AB" and not is_last) or
                (self.chain_mode == "A_only" and ith in self._ab_index_set)
            )
            if run_b_this_step:
                if use_checkpoint and self.training:
                    img_pred, latent, history_feat = torch.utils.checkpoint.checkpoint(
                        self.cascades_b[ith], img_pred, img_zf, latent, mask, sens_maps, history_feat, 
                        use_reentrant=False)
                else:
                    img_pred, latent, history_feat = self.cascades_b[ith](
                        img_pred, img_zf, latent, mask, sens_maps, history_feat)
                total_steps += 1
        
        # 调试信息：确认执行步骤数
        if hasattr(self, '_debug_printed') and not self._debug_printed:
            print(f"🔍 并行架构执行了 {total_steps} 个cascade步骤 (预期: {self.num_cascades * 2 - 1})")
            self._debug_printed = True

        # 获取最终输出
        current_kspace = sens_expand(img_pred, sens_maps, self.num_adj_slices)
        img_pred = torch.chunk(img_pred, self.num_adj_slices, dim=1)[self.center_slice]
        pred_kspace = torch.chunk(current_kspace, self.num_adj_slices, dim=1)[self.center_slice]
        
        # 处理img_pred为单线圈图像 (与promptmr_v2.py保持一致)
        sens_maps = torch.chunk(sens_maps, self.num_adj_slices, dim=1)[self.center_slice]
        img_pred = rss(complex_abs(complex_mul(img_pred, sens_maps)), dim=1)
        
        # 准备额外输出 (与promptmr_v2.py保持一致)
        kspace_zf = masked_kspace
        img_zf = torch.chunk(masked_kspace, self.num_adj_slices, dim=1)[self.center_slice]
        kspace_zf = torch.chunk(masked_kspace, self.num_adj_slices, dim=1)[self.center_slice]
        img_zf = rss(complex_abs(ifft2c(img_zf)), dim=1)
        
        sens_maps = torch.view_as_complex(sens_maps)
        
        return {
            'img_pred': img_pred,
            'img_zf': img_zf,
            'sens_maps': sens_maps,
            'pred_kspace': pred_kspace,
            'original_kspace': kspace_zf  # 修复：与promptmr_v2.py保持一致
        }
    
    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: torch.Tensor,
        mask_type: Tuple[str] = ("cartesian",),
        use_checkpoint: bool = False,
        compute_sens_per_coil: bool = False,
    ) -> torch.Tensor:
        """
        统一的前向传播接口
        """
        if self.parallel_mode:
            return self.forward_parallel(masked_kspace, mask, num_low_frequencies, mask_type, use_checkpoint, compute_sens_per_coil)
        else:
            # 调用父类的原始forward方法
            return super().forward(masked_kspace, mask, num_low_frequencies, mask_type, use_checkpoint, compute_sens_per_coil)


def create_parallel_promptmr_from_checkpoint(checkpoint_path: str, parallel_mode: bool = True):
    """
    从checkpoint创建并行PromptMR模型
    
    Args:
        checkpoint_path: 原始模型checkpoint路径
        parallel_mode: 是否使用并行模式
    
    Returns:
        ParallelPromptMR: 并行架构的模型
    """
    # 加载原始checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 提取模型配置
    if 'hyper_parameters' in checkpoint:
        config = checkpoint['hyper_parameters']
    else:
        # 如果checkpoint中没有配置，使用默认值
        config = {
            'num_cascades': 12,
            'num_adj_slices': 5,
            'n_feat0': 48,
            'feature_dim': [72, 96, 120],
            'prompt_dim': [24, 48, 72],
            'sens_n_feat0': 24,
            'sens_feature_dim': [36, 48, 60],
            'sens_prompt_dim': [12, 24, 36],
            'len_prompt': [5, 5, 5],
            'prompt_size': [64, 32, 16],
            'n_enc_cab': [2, 3, 3],
            'n_dec_cab': [2, 2, 3],
            'n_skip_cab': [1, 1, 1],
            'n_bottleneck_cab': 3,
            'no_use_ca': False,
            'learnable_prompt': False,
            'adaptive_input': True,
            'n_buffer': 4,
            'n_history': 0,
            'use_sens_adj': True,
        }
    
    # 创建并行模型
    parallel_model = ParallelPromptMR(
        parallel_mode=parallel_mode,
        **config
    )
    
    # 如果使用并行模式，复制权重
    if parallel_mode and 'state_dict' in checkpoint:
        # 先创建原始模型来提取权重
        original_model = PromptMR(**config)
        original_model.load_state_dict(checkpoint['state_dict'], strict=False)
        
        # 复制权重到并行架构
        parallel_model.copy_weights_from_original(original_model)
    
    return parallel_model

