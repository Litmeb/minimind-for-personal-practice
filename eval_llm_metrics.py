import argparse
import random
import pandas as pd
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import os
import json
from llm_as_a_judge import judge
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import *
from model.model_adapter import *
from trainer.train_projection_head import projectionhead, ClsTunedModel
from trainer.train_prompt_tuning import PromptTuningModel
from trainer.trainer_utils import setup_seed
from torch.utils.data import Dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
warnings.filterwarnings('ignore')

def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from) 
    if 'model' in args.load_from:
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling
        ))
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        
        # 如果使用projection head，需要加载带<cls> embedding的模型（vocab_size + 1）
        if args.projectionhead:
            # 扩展embedding层以容纳<cls> token
            original_vocab_size = len(tokenizer)
            model.resize_token_embeddings(original_vocab_size + 1)
            print(f'模型embedding已扩展以容纳<cls> token: {original_vocab_size} -> {original_vocab_size + 1}')
        
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
        
        if args.lora_weight != 'None':
            apply_lora(model, rank=args.rank)
            load_lora(model, f'./{args.save_dir}/lora/{args.lora_weight}_{args.hidden_size}.pth')
        
        if args.adapter_weight != 'None':
            apply_adapter(model, middle_features=args.middle_features)
            load_adapter(model, f'./{args.save_dir}/adapter/{args.adapter_weight}_{args.hidden_size}.pth')
        
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    
    print(f'MiniMind模型参数: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M(illion)')
    return model.eval().to(args.device), tokenizer

def main():
    parser = argparse.ArgumentParser(description="MiniMind模型推理与对话")
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）")
    parser.add_argument('--adapter_weight', default='None', type=str, help="Adapter权重名称（None表示不使用，可选：adapter_identity, adapter_medical）")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度（512=Small-26M, 640=MoE-145M, 768=Base-104M）")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量（Small/MoE=8, Base=16）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--max_new_tokens', default=8192, type=int, help="最大生成长度（注意：并非模型实际长文本能力）")
    parser.add_argument('--temperature', default=0.85, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.85, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument('--historys', default=0, type=int, help="携带历史对话轮数（需为偶数，0表示不携带历史）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    parser.add_argument('--prompt_tuning', default=0, type=int, help="使用prompt tuning进行分类")
    parser.add_argument('--projectionhead', default=False, action='store_true', help="使用projection head进行分类")
    parser.add_argument('--projectionhead_cls_tuning', default=False, action='store_true', help="使用tuned cls embedding+projection head进行分类")
    parser.add_argument('--cls_tuning_weight', default='cls_tuning_projection_head_classifier', type=str, help="cls tuning权重名称（None表示不使用，可选：full_sft, rlhf, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--rank', default=8, type=int, help="lora的秩")
    parser.add_argument("--middle_features", type=int, default=8, help="中间层特征维度")
    parser.add_argument('--llm_as_a_judge', default=False, action='store_true', help="使用llm作为法官进行分类准确性评估")
    parser.add_argument('--deepseek_api_key', default='None', type=str, help="deepseek api key（None表示不使用）")
    # TODO: add other llms for llm_as_a_judge
    args = parser.parse_args()
    if args.llm_as_a_judge:
        if args.deepseek_api_key != 'None':
            os.environ["DEEPSEEK_API_KEY"] = args.deepseek_api_key
        else:
            raise ValueError('deepseek_api_key is required when llm_as_a_judge is True')
    # prompts = [
    #     '你有什么特长？',
    #     '为什么天空是蓝色的',
    #     '请用Python写一个计算斐波那契数列的函数',
    #     '解释一下"光合作用"的基本过程',
    #     '如果明天下雨，我应该如何出门',
    #     '比较一下猫和狗作为宠物的优缺点',
    #     '解释什么是机器学习',
    #     '推荐一些中国的美食'
    # ]
    # 用于计算perplexity（仅在不使用projection head时）
    total_nll = 0.0  # 总负对数似然
    total_tokens = 0  # 总token数
    conversation = []
    model, tokenizer = init_model(args)
    moe_suffix = '_moe' if args.use_moe else ''

    # 如果使用projection head，需要加载它并构建类别映射
    label2id = None
    id2label = None

    # input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
    input_mode=0
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    # 读取jsonl测试文件
    import json
    test_prompts = []
    test_labels = []
    categories = set()
    
    with open('dataset/bbc_news_test.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            conversations = data['conversations']
            # 提取user的content作为prompt，assistant的content作为label
            if len(conversations) >= 2:
                prompt = conversations[0]['content']  # user的content
                label = conversations[1]['content']   # assistant的content
                test_prompts.append(prompt)
                test_labels.append(label)
                categories.add(label)
    
    categories = sorted(list(categories))
    if args.projectionhead:
        # 加载训练时保存的label2id映射（确保与训练时一致）
        
        label2id_path = f'./{args.save_dir}/{args.weight}_label2id_{args.hidden_size}{moe_suffix}.json'
        
        if os.path.exists(label2id_path):
            with open(label2id_path, 'r', encoding='utf-8') as f:
                label2id = json.load(f)
            # 将字符串键转换为整数（JSON保存时键会被转换为字符串）
            label2id = {label: int(idx) for label, idx in label2id.items()}
            id2label = {idx: label for label, idx in label2id.items()}
            num_classes = len(label2id)
            print(f'已加载训练时的label2id映射: {label2id_path}')
            print(f'类别映射: {label2id}')
        else:
            raise ValueError(f'未找到训练时的label2id映射: {label2id_path}')
        # 创建并加载projection head
        projection_head = projectionhead(args.hidden_size, num_classes).to(args.device)
        ckp_head = f'./{args.save_dir}/{args.weight}_head_{args.hidden_size}{moe_suffix}.pth'
        projection_head.load_state_dict(torch.load(ckp_head, map_location=args.device))
        projection_head.eval()
        print(f'已加载projection head，类别数: {num_classes}')
    if args.projectionhead_cls_tuning:
        # 加载训练时保存的cls embedding
        cls_embedding_path = f'./{args.save_dir}/{args.cls_tuning_weight}_cls_{args.hidden_size}{moe_suffix}.pth'
        cls_embedding = torch.load(cls_embedding_path, map_location=args.device)
        cls_embedding = cls_embedding.to(args.device)
        print(f'已加载cls embedding')
        model = ClsTunedModel(model, cls_embedding).to(args.device)
                # 加载训练时保存的label2id映射（确保与训练时一致）
        label2id_path = f'./{args.save_dir}/{args.cls_tuning_weight}_label2id_{args.hidden_size}{moe_suffix}.json'
        
        if os.path.exists(label2id_path):
            with open(label2id_path, 'r', encoding='utf-8') as f:
                label2id = json.load(f)
            # 将字符串键转换为整数（JSON保存时键会被转换为字符串）
            label2id = {label: int(idx) for label, idx in label2id.items()}
            id2label = {idx: label for label, idx in label2id.items()}
            num_classes = len(label2id)
            print(f'已加载训练时的label2id映射: {label2id_path}')
            print(f'类别映射: {label2id}')
        else:
            raise ValueError(f'未找到训练时的label2id映射: {label2id_path}')
        # 创建并加载projection head
        projection_head = projectionhead(args.hidden_size, num_classes).to(args.device)
        ckp_head = f'./{args.save_dir}/{args.cls_tuning_weight}_head_{args.hidden_size}{moe_suffix}.pth'
        projection_head.load_state_dict(torch.load(ckp_head, map_location=args.device))
        projection_head.eval()
        print(f'已加载projection head，类别数: {num_classes}')
    if args.prompt_tuning:
        # 加载 virtual_embedding（训练时保存为 half precision）
        virtual_embedding = torch.load(f'./{args.save_dir}/{args.weight}_virtual_embedding_{args.prompt_tuning}{moe_suffix}.pth', map_location=args.device)
        # 确保 virtual_embedding 的形状正确
        if len(virtual_embedding.shape) != 2 or virtual_embedding.shape[0] != args.prompt_tuning:
            raise ValueError(f'virtual_embedding 的形状错误：期望 ({args.prompt_tuning}, {args.hidden_size})，得到 {virtual_embedding.shape}')
        # 转换为 float32 以匹配模型权重（如果需要）
        virtual_embedding = virtual_embedding.float().to(args.device)
        # 如果用PromptTuningModel包装的话，下面的代码就不兼容了，而且还得自己写generate方法
        # HACK: 把几个virtual embedding假装成真实存在的token(<placeholder>)的embedding，然后给语料最前面加上placeholder，这样就能让模型在生成时使用placeholder token，找到对应的virtual embedding
        tokenizer.add_tokens([f'<placeholder_{i}>' for i in range(args.prompt_tuning)])
        model.resize_token_embeddings(len(tokenizer))
        
        # 将 virtual_embedding 赋值给新添加的 placeholder tokens
        # 注意：需要确保权重类型匹配（model 权重可能是 float32 或 bfloat16）
        with torch.no_grad():
            model.model.embed_tokens.weight[-args.prompt_tuning:, :] = virtual_embedding.to(model.model.embed_tokens.weight.dtype)
        
        print(f'已加载prompt tuning，虚拟token数: {args.prompt_tuning}')
    class dataset(Dataset):
        def __init__(self, prompts, labels):
            self.prompt = prompts
            self.label = labels
        def __len__(self):
            return len(self.prompt)
    # for prompt in ['你有什么特长？']:
    #     setup_seed(2026) # or setup_seed(random.randint(0, 2048))
    #     if input_mode == 0: print(f'👶: {prompt}')
    #     conversation = []
    #     conversation.append({"role": "user", "content": prompt})

    #     templates = {"conversation": conversation, "tokenize": False, "add_generation_prompt": True}
    #     inputs = tokenizer.apply_chat_template(**templates)
    #     inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

    #     print('🤖️: ', end='')
    #     generated_ids = model.generate(
    #         inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
    #         max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
    #         pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
    #         top_p=args.top_p, temperature=args.temperature, repetition_penalty=1.0
    #     )
    #     response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    test_dataset = dataset(test_prompts, test_labels)
    prompts = test_dataset.prompt
    labels = test_dataset.label
    
    # 获取所有类别并转换为token_id
    category_token_ids = {}
    for cat in categories:
        # 将类别名称转换为token_id（取第一个token）
        tokens = tokenizer.encode(cat, add_special_tokens=False)
        category_token_ids[cat] = tokens[0] if tokens else None
    llm_as_a_judge_correct = 0
    correct = 0  # 基于logits的准确率
    total = 0
    correct_gen = 0  # 基于生成的准确率
    total_gen = 0
    bleu_scores = []  # 存储所有BLEU分数
    if input_mode == 0:
        # 自动测试模式
        for idx, prompt in enumerate(prompts):
            # prompt='你有什么特长？'
            # print(f'👶: {prompt}')
            # print(f'👶: {prompt[:100]}...' if len(prompt) > 100 else f'👶: {prompt}')
            conversation = conversation[-args.historys:] if args.historys else []
            conversation.append({"role": "user", "content": prompt})
            true_category = labels[idx] if idx < len(labels) else None
            if args.projectionhead:
                cls_token_id=len(tokenizer)
                input_ids = torch.cat([torch.tensor([cls_token_id],device=args.device), torch.tensor(tokenizer.encode(prompt,add_special_tokens=False),dtype=torch.long,device=args.device)]).unsqueeze(0)
                attention_mask = (input_ids != tokenizer.pad_token_id).long()
                outputs = model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False
                )
                logits=outputs[0][:,0,:]
                cls=projection_head(logits)
                predicted_category = torch.argmax(cls, dim=-1).item()
                if predicted_category == label2id[true_category]:
                    correct += 1
                total += 1
                continue
            if args.projectionhead_cls_tuning:
                placeholder_id=1
                input_ids = torch.cat([torch.tensor([placeholder_id],device=args.device), torch.tensor(tokenizer.encode(prompt,add_special_tokens=False),dtype=torch.long,device=args.device)]).unsqueeze(0)
                attention_mask = (input_ids != tokenizer.pad_token_id).long()
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False
                )
                hidden_states = outputs.hidden_states
                logits=hidden_states[:,0,:]
                cls=projection_head(logits)
                predicted_category = torch.argmax(cls, dim=-1).item()
                if predicted_category == label2id[true_category]:
                    correct += 1
                total += 1
                continue

            # 计算perplexity：使用ground truth（真实答案）作为上下文
            if true_category:
                # 构建完整对话序列（prompt + 真实的assistant回复）
                full_conversation = conversation + [{"role": "assistant", "content": true_category}]
                full_templates = {"conversation": full_conversation, "tokenize": False, "add_generation_prompt": False}
                full_text = tokenizer.apply_chat_template(**full_templates)
                if args.prompt_tuning:
                    placeholder=[f'<placeholder_{i}>' for i in range(args.prompt_tuning)]
                    full_text = ''.join(placeholder)+full_text
                full_inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=8192).to(args.device)
                full_input_ids = full_inputs['input_ids'][0]  # [seq_len]
                full_attention_mask = full_inputs['attention_mask'][0]  # [seq_len]
                
                # 查找assistant回复的起始位置（通过查找<|im_start|>assistant标记）
                bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
                assistant_start_pos = None
                for i in range(len(full_input_ids) - len(bos_id) + 1):
                    if full_input_ids[i:i+len(bos_id)].tolist() == bos_id:
                        assistant_start_pos = i + len(bos_id)  # assistant内容的起始位置
                        break
                
                if assistant_start_pos is not None and assistant_start_pos < len(full_input_ids):
                    # 前向传播获取所有位置的logits
                    with torch.no_grad():
                        full_logits = model(input_ids=full_input_ids.unsqueeze(0), 
                                           attention_mask=full_attention_mask.unsqueeze(0)).logits[0]  # [seq_len, vocab_size]
                    
                    # 提取assistant回复部分的token（从assistant内容开始到序列结束）
                    assistant_tokens = full_input_ids[assistant_start_pos:]  # [assistant_len]
                    assistant_mask = full_attention_mask[assistant_start_pos:]  # [assistant_len]
                    
                    # 如果遇到eos_token，只计算到eos_token之前（包含eos_token）
                    if tokenizer.eos_token_id in assistant_tokens:
                        eos_idx = (assistant_tokens == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                        if len(eos_idx) > 0:
                            eos_pos = eos_idx[0].item() + 1  # 包含eos_token
                            assistant_tokens = assistant_tokens[:eos_pos]
                            assistant_mask = assistant_mask[:eos_pos]
                    
                    if len(assistant_tokens) > 0:
                        # 计算assistant回复部分的perplexity
                        # logits[i] 预测 input_ids[i+1]
                        # 所以 full_logits[assistant_start_pos-1] 预测 assistant_tokens[0]
                        #     full_logits[assistant_start_pos] 预测 assistant_tokens[1]
                        #     ...
                        logits_start_idx = assistant_start_pos - 1  # 用于预测assistant第一个token的logits位置
                        logits_end_idx = logits_start_idx + len(assistant_tokens)  # 最后一个预测位置
                        
                        # 获取assistant回复部分的logits
                        assistant_logits = full_logits[logits_start_idx:logits_end_idx, :]  # [assistant_len, vocab_size]
                        assistant_labels = assistant_tokens  # [assistant_len]
                        
                        # 计算每个token的负对数似然
                        log_probs = torch.log_softmax(assistant_logits[:,:len(tokenizer)-args.prompt_tuning], dim=-1)  # [assistant_len, vocab_size]
                        nll = -log_probs.gather(1, assistant_labels.unsqueeze(1)).squeeze(1)  # [assistant_len]
                        
                        # 只计算有效位置（非padding）
                        valid_nll = nll * assistant_mask.float()
                        total_nll += valid_nll.sum().item()
                        total_tokens += assistant_mask.sum().item()
            
            # 用于分类预测：只需要prompt部分
            templates = {"conversation": conversation, "tokenize": False, "add_generation_prompt": True}
            inputs = tokenizer.apply_chat_template(**templates)
            if args.prompt_tuning:
                placeholder=[f'<placeholder_{i}>' for i in range(args.prompt_tuning)]
                inputs = ''.join(placeholder)+inputs
            # print(f'inputs: {inputs}')
            # print(f'inputs: {inputs}')
            # print(true_category)
            # raise Exception('stop')
            inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)
            logits = model(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask']).logits
            
            # 取最后一个位置的logits用于分类
            last_logits = logits[0, -1, :len(tokenizer)-args.prompt_tuning]  # [vocab_size]
            last_prob = torch.softmax(last_logits, dim=-1)
            # 获取每个类别对应的分数
            category_scores = {}
            for cat, token_id in category_token_ids.items():
                if token_id is not None:
                    category_scores[cat] = last_prob[token_id].item()
            
            # 选择分数最高的类别作为预测
            predicted_category = max(category_scores, key=category_scores.get)
            
            # print(f'预测类别: {predicted_category} (分数: {category_scores[predicted_category]:.2f})')
            if true_category:
                # print(f'真实类别: {true_category}')
                if predicted_category == true_category:
                    correct += 1
                #     print('✓ 正确')
                # else:
                #     print('✗ 错误')
                total += 1
            
            # 方法2：基于生成的准确率（使用正则匹配）
            if true_category:
                # print(f'attention_mask: {inputs['attention_mask']}')
                # 让模型生成完整答案
                with torch.no_grad():
                    generated_ids = model.generate(
                        repetition_penalty=1.0,
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_new_tokens=50,  # 限制生成长度
                        do_sample=True,  # 使用采样
                        temperature=args.temperature,
                        top_p=args.top_p,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        bad_words_ids=[tokenizer.convert_tokens_to_ids([f'<placeholder_{i}>']) for i in range(args.prompt_tuning)] if args.prompt_tuning else None
                    )
                
                # 提取生成的部分（不包括prompt）
                prompt_length = inputs['input_ids'].shape[1]
                generated_tokens = generated_ids[0, prompt_length:]
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                # print(f'generated_text: {generated_text}')
                # raise Exception('stop')
                # 使用正则表达式匹配，检查生成的文本中是否包含真实类别
                # 构建正则表达式：匹配类别名称（不区分大小写，单词边界）
                if args.llm_as_a_judge:
                    result = judge(prompt, generated_text, true_category, categories, true_category)
                    if result['correctness']:
                        llm_as_a_judge_correct += 1
                pattern = r'\b' + re.escape(true_category) + r'\b'
                if re.search(pattern, generated_text, re.IGNORECASE):
                    correct_gen += 1
                total_gen += 1
                
                # 计算BLEU分数
                # 将参考答案和生成答案转换为token列表（按单词分割）
                reference = [true_category.lower().split()]  # BLEU需要列表的列表
                candidate = generated_text.lower().split()
                
                # 使用平滑函数避免0分（当n-gram不匹配时）
                smoothing = SmoothingFunction().method1
                bleu_score = sentence_bleu(reference, candidate, smoothing_function=smoothing)
                bleu_scores.append(bleu_score)
            if idx%100==0:
                print(idx)
    # 计算perplexity
    # print(f'total_nll: {total_nll}, total_tokens: {total_tokens}')
    if args.projectionhead or args.projectionhead_cls_tuning:
        print(f'准确率（基于projection head）: {correct}/{total} = {correct/total*100:.2f}%')
        exit()
    if total_tokens > 0:
        avg_nll = total_nll / total_tokens
        perplexity = torch.exp(torch.tensor(avg_nll)).item()
        print(f'Perplexity: {perplexity:.4f} (基于 {total_tokens} 个tokens)')
    
    if total > 0:
        accuracy = correct / total * 100
        print(f'准确率（基于logits）: {correct}/{total} = {accuracy:.2f}%')
    
    if total_gen > 0:
        accuracy_gen = correct_gen / total_gen * 100
        print(f'准确率（基于生成+正则匹配）: {correct_gen}/{total_gen} = {accuracy_gen:.2f}%')
    
    if len(bleu_scores) > 0:
        avg_bleu = np.mean(bleu_scores)
        print(f'BLEU分数: {avg_bleu:.4f} (基于 {len(bleu_scores)} 个样本)')
    if args.llm_as_a_judge:
        print(f'LLM作为法官准确率: {llm_as_a_judge_correct}/{total} = {llm_as_a_judge_correct/total*100:.2f}%')
    # for prompt in prompt_iter:
    #     setup_seed(2026) # or setup_seed(random.randint(0, 2048))
    #     if input_mode == 0: print(f'👶: {prompt}')
    #     conversation = conversation[-args.historys:] if args.historys else []
    #     conversation.append({"role": "user", "content": prompt})

    #     templates = {"conversation": conversation, "tokenize": False, "add_generation_prompt": True}
    #     if args.weight == 'reason': templates["enable_thinking"] = True # 仅Reason模型使用
    #     inputs = tokenizer.apply_chat_template(**templates) if args.weight != 'pretrain' else (tokenizer.bos_token + prompt)
    #     inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

    #     print('🤖️: ', end='')
    #     generated_ids = model.generate(
    #         inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
    #         max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
    #         pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
    #         top_p=args.top_p, temperature=args.temperature, repetition_penalty=1.0
    #     )
    #     response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    #     conversation.append({"role": "assistant", "content": response})
    #     print('\n\n')

if __name__ == "__main__":
    main()