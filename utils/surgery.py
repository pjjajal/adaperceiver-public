import torch
import torch.nn as nn


def init_with_pretrained_weights(model: nn.Module, teacher: nn.Module):
    assert len(model.blocks) == len(
        teacher.blocks
    ), "Model and teacher must have same number of blocks"
    assert (
        model.embed_dim == teacher.embed_dim
    ), "Model and teacher must have same embed_dim"
    embed_dim = model.embed_dim
    for student_block, teacher_block in zip(model.blocks, teacher.blocks):
        student_block.norm1.load_state_dict(teacher_block.norm1.state_dict())
        student_block.norm2.load_state_dict(teacher_block.norm2.state_dict())
    
        if not isinstance(teacher_block.ls1, nn.Identity):
            student_block.ls1.load_state_dict(teacher_block.ls1.state_dict())
        if not isinstance(teacher_block.ls2, nn.Identity):
            student_block.ls2.load_state_dict(teacher_block.ls2.state_dict())

        # Load attention weights
        student_block.attn.q.weight.data = teacher_block.attn.qkv.weight[
            :embed_dim, :
        ].clone()
        if teacher_block.attn.qkv.bias is not None:
            student_block.attn.q.bias.data = teacher_block.attn.qkv.bias[
                :embed_dim
            ].clone()
        student_block.attn.k.weight.data = teacher_block.attn.qkv.weight[
            embed_dim : 2 * embed_dim, :
        ].clone()
        if teacher_block.attn.qkv.bias is not None:
            student_block.attn.k.bias.data = teacher_block.attn.qkv.bias[
                embed_dim : 2 * embed_dim
            ].clone()
        student_block.attn.v.weight.data = teacher_block.attn.qkv.weight[
            2 * embed_dim :, :
        ].clone()
        if teacher_block.attn.qkv.bias is not None:
            student_block.attn.v.bias.data = teacher_block.attn.qkv.bias[
                2 * embed_dim :
            ].clone()

        student_block.attn.proj.load_state_dict(teacher_block.attn.proj.state_dict())

        # Load feed-forward network weights
        student_block.ffn.load_state_dict(teacher_block.mlp.state_dict())

        # Load patch embed weights
        # model.patch_embed.embed.load_state_dict(teacher.patch_embed.state_dict())
        # model.patch_embed.pos_embed.data = teacher.pos_embed.squeeze().clone()