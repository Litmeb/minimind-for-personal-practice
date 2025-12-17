import os
import sys
import random
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from torch.utils.data import Dataset
from model.model_lora import save_lora, apply_lora,save_rank_map
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')
def _iter_target_modules(model):
    for name, module in model.named_modules():
        if 'lora' not in name and ('q_proj' in name or 'k_proj' in name or 'v_proj' in name or 'o_proj' in name):
            if hasattr(module, 'weight'):
                yield name, module


def _randomized_svd_rank(matrix: torch.Tensor, k: int, q: int, tau: float) -> int:
    matrix_2d = matrix.detach().float()
    effective_k = min(k, matrix_2d.shape[0], matrix_2d.shape[1])
    if effective_k == 0:
        return 0,None

    omega = torch.randn(matrix_2d.shape[1], effective_k, device=matrix_2d.device, dtype=matrix_2d.dtype)
    y = matrix_2d @ omega
    for _ in range(max(1, q)):
        y = matrix_2d @ (matrix_2d.transpose(0, 1) @ y)

    q_matrix, _ = torch.linalg.qr(y, mode='reduced')
    b_matrix = q_matrix.transpose(0, 1) @ matrix_2d
    _, singular_values, _ = torch.linalg.svd(b_matrix, full_matrices=False)
    sum_singular_values = torch.pow(singular_values, 2).sum()
    p=torch.pow(singular_values, 2)/sum_singular_values
    log_p=-(p*torch.log(p)).sum()
    r_eff=torch.exp(log_p)


    singular_values = singular_values[:effective_k]

    if singular_values.numel() == 0:
        return 0,r_eff

    energy = singular_values.pow(2)
    energy_sum = energy.sum()
    if energy_sum == 0:
        return 0,r_eff

    coverage = torch.cumsum(energy, dim=0) / energy_sum
    rank_idx = torch.nonzero(coverage >= tau, as_tuple=False)
    rank = rank_idx[0, 0].item() + 1 if rank_idx.numel() > 0 else singular_values.numel()
    return min(rank, effective_k),r_eff


def estimate_ranks(model, loader, steps, autocast_ctx, tau, k, q, device):
    grad_sums = {name: torch.zeros_like(module.weight, device=device) for name, module in _iter_target_modules(model)}
    loss_fct = nn.CrossEntropyLoss(reduction='none')

    model.train()
    for idx, (X, Y, loss_mask) in enumerate(loader):
        if idx >= steps:
            break

        X = X.to(device)
        Y = Y.to(device)
        loss_mask = loss_mask.to(device)

        with autocast_ctx:
            res = model(X)
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss += res.aux_loss

        loss.backward()

        for name, module in _iter_target_modules(model):
            if module.weight.grad is not None:
                grad_sums[name] += module.weight.grad.detach()

        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

    rank_map = {}
    for name, grad in grad_sums.items():
        rank,r_eff = _randomized_svd_rank(grad, k=k, q=q, tau=tau)
        rank_map[name] = rank if rank > 0 else 1
        Logger(f"LoRA rank for {name}: {rank_map[name]} r_eff: {r_eff}")

    return rank_map
class LoraDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples, self.validation_samples = self.load_data(jsonl_path)
        self.bos_id = tokenizer('answer: <|im_start|>', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def load_data(self, path):
        samples = []
        validation_samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                if line_num % 10 == 0:
                    validation_samples.append(data)
                else:
                    samples.append(data)
        return samples, validation_samples

    def _create_chat_prompt(self, cs):
        messages = cs.copy()
        tools = cs[0]["functions"] if (cs and cs[0]["role"] == "system" and cs[0].get("functions")) else None
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start , min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index):
        sample = self.samples[index]
        # 构建对话提示
        prompt = sample['text']+'<|im_start|>'+sample['label']+self.tokenizer.eos_token
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        # 生成动态损失掩码
        loss_mask = self._generate_loss_mask(input_ids)
        # print(input_ids)
        # print(self.tokenizer.decode(input_ids))
        # print(self.bos_id)
        # print(loss_mask)
        # raise Exception('stop')
        # 构建训练数据
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 对齐预测位置
        # # === 打印每个token的掩码情况 ===
        # print(f"\n--- Sample {index} Token Loss Mask (length: {len(input_ids)}) ---")
        # for i, (token_id, mask) in enumerate(zip(input_ids, loss_mask)):
        #     token_str = self.tokenizer.decode([token_id], skip_special_tokens=False)
        #     token_str = token_str.replace('\n', '\\n').replace('\t', '\\t')  # 处理换行等不可见字符
        #     print(f"Token {i:3d}: {token_id:5d} -> '{token_str:10s}' | mask: {mask}")
        # print(f"--- End of Sample {index} ---")
        # # ================================
        return X, Y, loss_mask
    
    def validation(self, model):
        correct = 0
        total = 0
        category_token_ids = {
            'sport': self.tokenizer.encode('sport', add_special_tokens=False)[0],
            'politics': self.tokenizer.encode('politics', add_special_tokens=False)[0],
            'entertainment': self.tokenizer.encode('entertainment', add_special_tokens=False)[0],
            'business': self.tokenizer.encode('business', add_special_tokens=False)[0],  # FIX: was 'science'
            'technology': self.tokenizer.encode('technology', add_special_tokens=False)[0]
            }
        
        # Handle DDP model
        actual_model = model.module if hasattr(model, 'module') else model
        device = next(actual_model.parameters()).device
        
        for sample in self.validation_samples:
            prompt = sample['text']+'<|im_start|>'
            input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
            position = len(input_ids)
            # FIX: use position-1 to get logits that predict the answer token
            logit_position = position - 1
            input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
            input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
            with torch.no_grad():
                output_logits = actual_model(input_ids).logits.squeeze(0)
            category_scores = {}
            
            for cat, token_id in category_token_ids.items():
                if token_id is not None:
                    category_scores[cat] = output_logits[logit_position][token_id].item()
            
            # 选择分数最高的类别作为预测
            predicted_category = max(category_scores, key=category_scores.get)
            true_category = sample['label']
            if true_category:
                if predicted_category == true_category:
                    correct += 1
                total += 1
        return correct, total
def train_epoch(epoch, loader, iters, lora_params, start_step=0, wandb=None, train_ds=None):
    # if train_ds:
    #     correct, total = train_ds.validation(model)
    #     Logger(f'Epoch [{epoch + 1}/{args.epochs}]: Validation Accuracy: {correct}/{total} = {correct/total:.2f}')
    #     if wandb: wandb.log({"validation_accuracy": correct/total, "epoch": epoch})
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(loader, start=start_step + 1):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(X)
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())

            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss += res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            
            Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}) loss:{current_loss:.6f} lr:{current_lr:.12f} epoch_Time:{eta_min}min:')
            
            if wandb: wandb.log({"loss": current_loss, "lr": current_lr, "epoch_Time": eta_min})

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            lora_save_path = f'{args.save_dir}/{args.lora_name}_{lm_config.hidden_size}.pth'
            # LoRA只保存LoRA权重
            save_lora(model, lora_save_path)
            lm_checkpoint(lm_config, weight=args.lora_name, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
        
        del X, Y, loss_mask, res, loss
    if train_ds:
        model.eval()
        correct, total = train_ds.validation(model)
        Logger(f'Validation Accuracy: {correct}/{total} = {correct/total:.2f}')
        if wandb: wandb.log({"validation_accuracy": correct/total, "epoch": epoch})
        model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind LoRA Fine-tuning")
    parser.add_argument("--save_dir", type=str, default="../out/lora", help="模型保存目录")
    parser.add_argument("--lora_name", type=str, default="SVDlora_classifier_en", help="LoRA权重名称(如lora_identity/lora_medical等)")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=10, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=1024, type=int, help="训练的最大截断长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/bbc_news_train_chunked.jsonl", help="LoRA训练数据路径")
    parser.add_argument('--from_weight', default='pretrain_en', type=str, help="基于哪个权重训练，默认full_sft")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-SVDLoRA-en", help="wandb项目名")
    parser.add_argument('--rank', default=8, type=int, help="lora的秩")
    parser.add_argument('--rank_eval_steps', default=30, type=int, help="用于rSVD估计秩的batch数量（0表示跳过）")
    parser.add_argument('--svd_tau', default=0.95, type=float, help="rSVD能量覆盖率阈值τ")
    parser.add_argument('--svd_k', default=64, type=int, help="rSVD截断奇异值数k")
    parser.add_argument('--svd_q', default=2, type=int, help="rSVD幂迭代次数q (建议1-2)")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.lora_name, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        # import swanlab as wandb
        import wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-LoRA-{args.lora_name}-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、应用LoRA、冻结非LoRA参数 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    # ========== 6. rSVD估计各投影层rank ==========
    train_ds = LoraDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    if args.rank_eval_steps!=0:
        rank_sampler = DistributedSampler(train_ds, shuffle=False) if dist.is_initialized() else None
        rank_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, sampler=rank_sampler, num_workers=args.num_workers, pin_memory=True)
        svd_q = max(1, min(args.svd_q, 2))
        rank_map = None
        rank_map_path = os.path.join(args.save_dir, f"{args.lora_name}_{lm_config.hidden_size}_rank_map.json")
        if args.rank_eval_steps > 0:
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    rank_map = estimate_ranks(model, rank_loader, args.rank_eval_steps, autocast_ctx, args.svd_tau, args.svd_k, svd_q, args.device)
                obj_list = [rank_map]
                dist.broadcast_object_list(obj_list, src=0)
                rank_map = obj_list[0]
            else:
                rank_map = estimate_ranks(model, rank_loader, args.rank_eval_steps, autocast_ctx, args.svd_tau, args.svd_k, svd_q, args.device)

        # if rank_map is not None and is_main_process():
        #     save_rank_map(rank_map, rank_map_path)
        # apply_lora(model, rank_map=rank_map)
    # else:
    apply_lora(model, rank=args.rank)
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    lora_params_count = sum(p.numel() for name, p in model.named_parameters() if 'lora' in name)
    Logger(f"LLM 总参数量: {total_params / 1e6:.3f} M")
    Logger(f"LoRA 参数量: {lora_params_count / 1e6:.3f} M")
    Logger(f"LoRA 参数占比: {lora_params_count / total_params * 100:.2f}%")
    
    # 冻结非LoRA参数，收集LoRA参数
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
            lora_params.append(param)
        else:
            param.requires_grad = False
    
    # ========== 7. 定义数据和优化器 ==========
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(lora_params, lr=args.learning_rate)
    
    # ========== 8. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 9. DDP包模型 ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 10. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        if epoch == start_epoch and start_step > 0: # 第一个epoch且存在检查点
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), args.batch_size, start_step + 1)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + start_step + 1, lora_params, start_step, wandb, train_ds)
        else: # 默认从头开始
            loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
            train_epoch(epoch, loader, len(loader), lora_params, 0, wandb, train_ds)
            for name, param in model.named_parameters():
                param.requires_grad = True
            if args.rank_eval_steps!=0:
                rank_sampler = DistributedSampler(train_ds, shuffle=False) if dist.is_initialized() else None
                rank_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, sampler=rank_sampler, num_workers=args.num_workers, pin_memory=True)
                svd_q = max(1, min(args.svd_q, 2))
                rank_map = None
                rank_map_path = os.path.join(args.save_dir, f"{args.lora_name}_{lm_config.hidden_size}_rank_map.json")
                if args.rank_eval_steps > 0:
                    if dist.is_initialized():
                        if dist.get_rank() == 0:
                            rank_map = estimate_ranks(model, rank_loader, args.rank_eval_steps, autocast_ctx, args.svd_tau, args.svd_k, svd_q, args.device)
                        obj_list = [rank_map]
                        dist.broadcast_object_list(obj_list, src=0)
                        rank_map = obj_list[0]
                    else:
                        rank_map = estimate_ranks(model, rank_loader, args.rank_eval_steps, autocast_ctx, args.svd_tau, args.svd_k, svd_q, args.device)
                for name, param in model.named_parameters():
                    if 'lora' not in name:
                        param.requires_grad = False