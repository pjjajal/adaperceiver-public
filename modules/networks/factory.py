from omegaconf import DictConfig
from .adaperceiver import AdaPercevierConfig
from .classification import ClassificationAdaPerceiver, ClassificationConfig
from .dense_pred import DensePredictionAdaPerceiver, DensePredictionConfig
from .distillation import DistillationAdaPerceiver, DistillationConfig


def create_model(cfg: DictConfig, **kwargs):
    encoder_cfg = cfg.encoder_config
    encoder_cfg.ffn_ratio = eval(encoder_cfg.ffn_ratio) if isinstance(encoder_cfg.ffn_ratio, str) else encoder_cfg.ffn_ratio
    class_cfg = ClassificationConfig(
        img_size=encoder_cfg.img_size,
        in_channels=encoder_cfg.in_channels,
        patch_size=encoder_cfg.patch_size,
        num_classes=encoder_cfg.num_classes,
        use_embed_ffn=encoder_cfg.use_embed_ffn,
        use_output_ffn=encoder_cfg.use_output_ffn,
        use_fourier=encoder_cfg.use_fourier,
        fourier_bands=encoder_cfg.fourier_bands,
        adapter_type=encoder_cfg.adapter_type,
        conv_adapter_model=encoder_cfg.conv_adapter_model,
    )
    perceiver_cfg = AdaPercevierConfig(
        embed_dim=encoder_cfg.embed_dim,
        num_heads=encoder_cfg.num_heads,
        depth=encoder_cfg.depth,
        max_latent_tokens=encoder_cfg.max_latent_tokens,
        max_latent_tokens_mult=encoder_cfg.max_latent_tokens_mult,
        rope_theta=encoder_cfg.rope_theta,
        ffn_ratio=encoder_cfg.ffn_ratio,
        qkv_bias=encoder_cfg.qkv_bias,
        proj_bias=encoder_cfg.proj_bias,
        ls_init_values=encoder_cfg.ls_init_values,
        proj_drop=encoder_cfg.proj_drop,
        attn_drop=encoder_cfg.attn_drop,
        drop_path=encoder_cfg.drop_path,
        head_drop=encoder_cfg.head_drop,
        act_layer=encoder_cfg.act_layer,
        norm_layer=encoder_cfg.norm_layer,
        ffn_layer=encoder_cfg.ffn_layer,
        attn_layer=encoder_cfg.attn_layer,
        process_token_init=encoder_cfg.process_token_init,
        block_mask=cfg.mask_type,
        mask_token_grans=cfg.token_grans,
    )
    return ClassificationAdaPerceiver(
        config=perceiver_cfg,
        cls_config=class_cfg,
    )


def create_distillation_model(cfg: DictConfig, **kwargs):
    encoder_cfg = cfg.encoder_config
    encoder_cfg.ffn_ratio = eval(encoder_cfg.ffn_ratio) if isinstance(encoder_cfg.ffn_ratio, str) else encoder_cfg.ffn_ratio
    distillation_cfg = DistillationConfig(
        img_size=encoder_cfg.img_size,
        in_channels=encoder_cfg.in_channels,
        patch_size=encoder_cfg.patch_size,
        use_fourier=encoder_cfg.use_fourier,
        fourier_bands=encoder_cfg.fourier_bands,
        num_classes=encoder_cfg.num_classes,
        output_feat_dim=kwargs["output_feat_dim"],
        use_embed_ffn=encoder_cfg.use_embed_ffn,
        use_output_ffn=encoder_cfg.use_output_ffn,
        adapter_type=encoder_cfg.adapter_type,
        conv_adapter_model=encoder_cfg.conv_adapter_model,
    )
    perceiver_cfg = AdaPercevierConfig(
        embed_dim=encoder_cfg.embed_dim,
        num_heads=encoder_cfg.num_heads,
        depth=encoder_cfg.depth,
        max_latent_tokens=encoder_cfg.max_latent_tokens,
        max_latent_tokens_mult=encoder_cfg.max_latent_tokens_mult,
        rope_theta=encoder_cfg.rope_theta,
        ffn_ratio=encoder_cfg.ffn_ratio,
        qkv_bias=encoder_cfg.qkv_bias,
        proj_bias=encoder_cfg.proj_bias,
        ls_init_values=encoder_cfg.ls_init_values,
        proj_drop=encoder_cfg.proj_drop,
        attn_drop=encoder_cfg.attn_drop,
        drop_path=encoder_cfg.drop_path,
        head_drop=encoder_cfg.head_drop,
        act_layer=encoder_cfg.act_layer,
        norm_layer=encoder_cfg.norm_layer,
        ffn_layer=encoder_cfg.ffn_layer,
        attn_layer=encoder_cfg.attn_layer,
        process_token_init=encoder_cfg.process_token_init,
        block_mask=cfg.mask_type,
        mask_token_grans=cfg.token_grans,
    )
    return DistillationAdaPerceiver(
        config=perceiver_cfg,
        distillation_config=distillation_cfg,
    )


def create_dense_prediction_model(cfg: DictConfig, **kwargs):
    encoder_cfg = cfg.encoder_config
    encoder_cfg.ffn_ratio = eval(encoder_cfg.ffn_ratio) if isinstance(encoder_cfg.ffn_ratio, str) else encoder_cfg.ffn_ratio 
    dense_cfg = DensePredictionConfig(
        img_size=encoder_cfg.img_size,
        in_channels=encoder_cfg.in_channels,
        patch_size=encoder_cfg.patch_size,
        use_fourier=encoder_cfg.use_fourier,
        fourier_bands=encoder_cfg.fourier_bands,
        use_embed_ffn=encoder_cfg.use_embed_ffn,
        use_output_ffn=encoder_cfg.use_output_ffn,
        adapter_type=encoder_cfg.adapter_type,
        conv_adapter_model=encoder_cfg.conv_adapter_model,
        output_feat_dim=encoder_cfg.output_feat_dim
    )
    perceiver_cfg = AdaPercevierConfig(
        embed_dim=encoder_cfg.embed_dim,
        num_heads=encoder_cfg.num_heads,
        depth=encoder_cfg.depth,
        max_latent_tokens=encoder_cfg.max_latent_tokens,
        max_latent_tokens_mult=encoder_cfg.max_latent_tokens_mult,
        rope_theta=encoder_cfg.rope_theta,
        ffn_ratio=encoder_cfg.ffn_ratio,
        qkv_bias=encoder_cfg.qkv_bias,
        proj_bias=encoder_cfg.proj_bias,
        ls_init_values=encoder_cfg.ls_init_values,
        proj_drop=encoder_cfg.proj_drop,
        attn_drop=encoder_cfg.attn_drop,
        drop_path=encoder_cfg.drop_path,
        head_drop=encoder_cfg.head_drop,
        act_layer=encoder_cfg.act_layer,
        norm_layer=encoder_cfg.norm_layer,
        ffn_layer=encoder_cfg.ffn_layer,
        attn_layer=encoder_cfg.attn_layer,
        process_token_init=encoder_cfg.process_token_init,
        block_mask=cfg.mask_type,
        mask_token_grans=cfg.token_grans,
    )
    return DensePredictionAdaPerceiver(
        config=perceiver_cfg,
        dense_config=dense_cfg,
    )
