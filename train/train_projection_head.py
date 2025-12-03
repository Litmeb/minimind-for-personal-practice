import argparse
import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.utils.rnn as rnn_utils
import torch.optim.lr_scheduler as lr_scheduler
import time
import json
import random
import numpy as np
import logging
import warnings
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm
from contextlib import nullcontext
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model, get_lr
from peft import PromptTuningConfig, get_peft_model

warnings.filterwarnings('ignore')

# 修改模型的 forward 来注入 cls embedding（使用 wrapper）
class ClsTunedModel(nn.Module):
    def __init__(self, model, cls_embedding):
        super().__init__()
        self.model = model
        self.cls_embedding = cls_embedding

    def forward(self, input_ids, **kwargs):
        embeds = self.model.model.embed_tokens(input_ids)
        batch_size = embeds.size(0)
        # 替换序列开头的 <cls> embedding
        embeds[:, 0, :] = self.cls_embedding.expand(batch_size, -1)
        return self.model(inputs_embeds=embeds, **kwargs)
        
class projectionhead(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.linear = nn.Linear(hidden_size, 128)
        self.linear2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(hidden_size)  # 输入层归一化
        self.norm2 = nn.LayerNorm(128)  # 中间层归一化
    
    def forward(self, x):
        x = self.norm1(x)  # 归一化输入 (batch_size, hidden_size)
        x = self.linear(x)  # (batch_size, 128)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.norm2(x)  # 归一化中间层 (batch_size, 128)
        x = self.linear2(x)  # (batch_size, num_classes)
        return x


class ClassificationDataset(Dataset):
    """文本分类数据集，使用<cls> token进行分类"""
    def __init__(self, jsonl_path, tokenizer, max_length=1024, label2id=None, placeholder_id=0):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(jsonl_path)
        self.placeholder_id = placeholder_id
        # 获取所有类别标签
        all_labels = set()
        for sample in self.samples:
            label = sample['label']
            all_labels.add(label)
        all_labels = sorted(list(all_labels))
        
        # 建立标签映射
        if label2id is None:
            self.label2id = {label: idx for idx, label in enumerate(all_labels)}
        else:
            self.label2id = label2id
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        self.num_classes = len(self.label2id)
        
        Logger(f'数据集类别数: {self.num_classes}, 类别: {list(self.label2id.keys())}')

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                # 提取文本和标签
                conversations = data.get('conversations', [])
                if len(conversations) >= 2:
                    user_content = conversations[0].get('content', '')
                    label = conversations[1].get('content', '').strip()
                    samples.append({
                        'text': user_content,
                        'label': label
                    })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        text = sample['text']
        label = sample['label']
        label_id = self.label2id[label]
        
        # Tokenize文本（不添加特殊token，因为我们要手动添加<cls>）
        encoding = self.tokenizer(
            text,
            max_length=self.max_length - 1,  # 留一个位置给<cls>
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            add_special_tokens=False
        )
        
        input_ids = encoding.input_ids.squeeze(0)  # (max_length-1,)
        
        # 在开头添加<cls> placeholder (使用 original_vocab_size or any unused ID; embedding will be replaced in forward)
        input_ids = torch.cat([torch.tensor([self.placeholder_id]), input_ids[:-1]])  # Placeholder ID
        
        # 创建attention mask
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        return input_ids, attention_mask, label_id


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    global tokenizer, train_ds
    loss_fct = nn.CrossEntropyLoss()
    start_time = time.time()
    
    for step, (input_ids, attention_mask, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        attention_mask = attention_mask.to(args.device)
        labels = labels.to(args.device)
        
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            # 前向传播，获取hidden states
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                outputs = model.module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False
                )
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False
                )
            hidden_states = outputs.hidden_states  # (batch_size, seq_len, hidden_size)
            # print(hidden_states.shape)
            # 获取<cls> token的hidden state（第一个位置）
            cls_hidden = hidden_states[:, 0, :]  # (batch_size, hidden_size)
            
            # 通过projection head进行分类
            if isinstance(projection_head, torch.nn.parallel.DistributedDataParallel):
                logits = projection_head.module(cls_hidden)  # (batch_size, num_classes)
            else:
                logits = projection_head(cls_hidden)  # (batch_size, num_classes)
            
            # 计算损失
            loss = loss_fct(logits, labels)
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            # 只对可训练参数进行梯度裁剪（optimizer中的所有参数都是可训练的）
            torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            
            # 计算准确率
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                accuracy = (preds == labels).float().mean().item()
            
            Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}) loss:{current_loss:.6f} acc:{accuracy:.4f} lr:{current_lr:.12f} epoch_Time:{eta_min}min')
            
            if wandb: 
                wandb.log({
                    "loss": current_loss, 
                    "accuracy": accuracy,
                    "lr": current_lr, 
                    "epoch_Time": eta_min
                })

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            if isinstance(projection_head, torch.nn.parallel.DistributedDataParallel):
                projection_head.module.eval()
            else:
                projection_head.eval()
            
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp_head = f'{args.save_dir}/{args.save_weight}_head_{lm_config.hidden_size}{moe_suffix}.pth'
            ckp_cls = f'{args.save_dir}/{args.save_weight}_cls_{lm_config.hidden_size}{moe_suffix}.pth'
            
            # 保存projection head
            if isinstance(projection_head, torch.nn.parallel.DistributedDataParallel):
                torch.save(projection_head.module.state_dict(), ckp_head)
            else:
                torch.save(projection_head.state_dict(), ckp_head)
            
            # 保存 cls_embedding (assuming it exists from previous fix)
            torch.save(model.module.cls_embedding.data.half() if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.cls_embedding.data.half(), ckp_cls)
            
            # 保存label2id映射
            import json
            label2id_path = f'{args.save_dir}/{args.save_weight}_label2id_{lm_config.hidden_size}{moe_suffix}.json'
            with open(label2id_path, 'w', encoding='utf-8') as f:
                json.dump(train_ds.label2id, f, ensure_ascii=False, indent=2)
            Logger(f'已保存label2id映射到: {label2id_path}')
            
            if isinstance(projection_head, torch.nn.parallel.DistributedDataParallel):
                projection_head.module.train()
            else:
                projection_head.train()

        del input_ids, attention_mask, labels, outputs, hidden_states, cls_hidden, logits, loss


def main():
    global args, model, projection_head, optimizer, scaler, autocast_ctx, lm_config, train_ds
    
    parser = argparse.ArgumentParser(description="MiniMind Projection Head Classification")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='cls_tuning_projection_head_classifier', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=500, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=1024, type=int, help="训练的最大截断长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/bbc_news_train.jsonl", help="训练数据路径")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练，为none则不基于任何权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="minimind-cls-tuning-projection-head", help="wandb项目名")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): 
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size, 
        num_hidden_layers=args.num_hidden_layers, 
        use_moe=bool(args.use_moe)
    )
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配置wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import wandb as wandb_module
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-ProjHead-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb = wandb_module.init(
            entity='xi-an-jiaotong-university-ltimbe',
            project=args.wandb_project, 
            name=wandb_run_name, 
            id=wandb_id, 
            resume=resume
        )
    
    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    
    # 添加<cls> token到tokenizer（临时添加，用于训练）
    # 注意：不需要保存新tokenizer，推理时直接使用原始vocab_size作为<cls>的id
    # 不添加 token 或 resize embedding；使用 placeholder ID for <cls> (e.g., original_vocab_size)
    placeholder_id = 0  # Placeholder; will manually handle in forward
    Logger(f'Using placeholder placeholder_id: {placeholder_id} (no tokenizer modification)')

    # 冻结所有模型参数
    for param in model.parameters():
        param.requires_grad = False
    Logger('已冻结所有模型参数')

    # 创建数据集（先创建一个临时数据集以获取类别数）
    temp_ds = ClassificationDataset(args.data_path, tokenizer, max_length=args.max_seq_len, placeholder_id=placeholder_id)
    num_classes = temp_ds.num_classes

    # 创建projection head
    projection_head = projectionhead(lm_config.hidden_size, num_classes).to(args.device)
    # projection head的所有参数都是可训练的

    # 新增：独立的 trainable cls embedding
    cls_embedding = nn.Parameter(torch.randn(1, lm_config.hidden_size))  # shape: (1, hidden_size)

    model = ClsTunedModel(model, cls_embedding).to(args.device)

    # 重新创建数据集（使用正确的label2id）
    train_ds = ClassificationDataset(args.data_path, tokenizer, max_length=args.max_seq_len, label2id=temp_ds.label2id, placeholder_id=placeholder_id)
    
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    
    # 收集所有可训练参数
    trainable_params = []
    # cls_embedding参数
    trainable_params.append(cls_embedding)
    # projection head的所有参数
    trainable_params.extend(projection_head.parameters())
    
    # 统计可训练参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_count = sum(p.numel() for p in trainable_params)
    Logger(f'模型总参数量: {total_params / 1e6:.3f} M')
    Logger(f'可训练参数量: {trainable_params_count / 1e6:.3f} M (<cls> embedding + projection head)')
    Logger(f'可训练参数占比: {trainable_params_count / total_params * 100:.4f}%')
    
    # 优化器：只优化可训练参数
    optimizer = optim.AdamW(trainable_params, lr=args.learning_rate)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.model.load_state_dict(ckp_data['model'], strict=False)  # Load original model
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
        
        # Load projection head
        moe_suffix = '_moe' if lm_config.use_moe else ''
        ckp_head = f'{args.save_dir}/{args.save_weight}_head_{lm_config.hidden_size}{moe_suffix}.pth'
        if os.path.exists(ckp_head):
            if isinstance(projection_head, torch.nn.parallel.DistributedDataParallel):
                projection_head.module.load_state_dict(torch.load(ckp_head, map_location=args.device))
            else:
                projection_head.load_state_dict(torch.load(ckp_head, map_location=args.device))
            Logger(f'已加载projection head: {ckp_head}')
        
        # Load cls_embedding
        ckp_cls = f'{args.save_dir}/{args.save_weight}_cls_{lm_config.hidden_size}{moe_suffix}.pth'
        if os.path.exists(ckp_cls):
            cls_embedding.data = torch.load(ckp_cls, map_location=args.device)
            Logger(f'已加载cls_embedding: {ckp_cls}')
        
        # Rebuild wrapper with loaded cls_embedding
        model = ClsTunedModel(model.model, cls_embedding).to(args.device)
        
        # Freeze original model params
        for param in model.model.parameters():
            param.requires_grad = False
        
        Logger(f'已恢复参数冻结状态：只有cls_embedding和projection head可训练')

        # 注意：优化器状态已经在上面加载了，不需要重新创建
    
    # ========== 7. DDP包模型 ==========
    if dist.is_initialized():
        model.model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}  # 调整为 wrapper 中的 model
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        projection_head = torch.nn.parallel.DistributedDataParallel(projection_head, device_ids=[local_rank])
    
    # ========== 8. 设置训练模式 ==========
    model.train()
    projection_head.train()
    
    # ========== 9. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        
        if epoch == start_epoch and start_step > 0:  # 第一个epoch且存在检查点
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), args.batch_size, start_step + 1)
            loader = DataLoader(
                train_ds, 
                batch_sampler=batch_sampler, 
                num_workers=args.num_workers, 
                pin_memory=True
            )
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + start_step + 1, start_step, wandb)
        else:  # 默认从头开始
            loader = DataLoader(
                train_ds, 
                batch_size=args.batch_size, 
                shuffle=(train_sampler is None), 
                sampler=train_sampler, 
                num_workers=args.num_workers, 
                pin_memory=True
            )
            train_epoch(epoch, loader, len(loader), 0, wandb)
    # Save cls_embedding
    ckp_cls = f'{args.save_dir}/{args.save_weight}_cls_{lm_config.hidden_size}.pth'
    torch.save(cls_embedding.data.half(), ckp_cls)  # Save as half precision

    Logger('训练完成！')
    if wandb:
        wandb.finish()

    


if __name__ == "__main__":
    main()