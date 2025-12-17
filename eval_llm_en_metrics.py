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
        
        if args.lora_weight == 'SVDlora_classifier_en':
            lora_weight_path = f'./{args.save_dir}/lora/{args.lora_weight}_{args.hidden_size}.pth'
            rank_map_path = lora_weight_path.replace('.pth', '_rank_map.json')
            rank_map = load_rank_map(rank_map_path)
            print(f'rank_map: {rank_map}')
            apply_lora(model, rank_map=rank_map)
            load_lora(model, lora_weight_path)
        else:
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
    parser.add_argument('--weight', default='pretrain_en', type=str, help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）")
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
    model, tokenizer = init_model(args)
    test_prompts = []
    test_labels = []
    categories = set()
    
    with open('dataset/bbc_news_test_no_instruction.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            prompt = data['text']  # user的content
            label = data['label']   # assistant的content
            test_prompts.append(prompt)
            test_labels.append(label)
            categories.add(label)
    
    categories = sorted(list(categories))

    def split_text_into_chunks(text, target_words=700, overlap_words=50):
        """
        将文本切分成指定单词数的chunks，尽量在句子边界处切分
        
        Args:
            text: 要切分的文本
            target_words: 目标单词数（约）
            overlap_words: chunk之间的重叠单词数
        
        Returns:
            chunks列表
        """
        if not text or not text.strip():
            return []
        
        # 按空格分割单词
        words = text.split()
        if len(words) <= target_words:
            return [text]
        
        chunks = []
        start_idx = 0
        
        while start_idx < len(words):
            # 计算当前chunk的结束位置
            end_idx = min(start_idx + target_words, len(words))
            
            # 如果还没到文本末尾，尝试在句子边界处切分
            if end_idx < len(words):
                # 向后查找句子结束符（. ! ? 后跟空格或换行）
                # 在目标位置前后100个单词范围内查找
                search_start = max(start_idx + target_words - 100, start_idx)
                search_end = min(end_idx + 100, len(words))
                
                best_split = end_idx
                # 从目标位置向前查找句子边界
                for i in range(end_idx, search_start, -1):
                    if i < len(words):
                        # 检查前一个单词是否以句子结束符结尾
                        prev_word = words[i-1] if i > 0 else ""
                        if prev_word and re.search(r'[.!?]$', prev_word):
                            best_split = i
                            break
                
                end_idx = best_split
            
            # 提取chunk
            chunk_words = words[start_idx:end_idx]
            chunk_text = " ".join(chunk_words)
            
            if chunk_text.strip():
                chunks.append(chunk_text.strip())
            
            # 移动到下一个chunk的起始位置（考虑重叠）
            if end_idx >= len(words):
                break
            start_idx = max(end_idx - overlap_words, start_idx + 1)
        
        return chunks
    def predict(model, tokenizer, max_length, test_prompts, test_labels):
        category_token_ids = {
            'sport': tokenizer.encode('sport', add_special_tokens=False)[0],
            'politics': tokenizer.encode('politics', add_special_tokens=False)[0],
            'entertainment': tokenizer.encode('entertainment', add_special_tokens=False)[0],
            'business': tokenizer.encode('business', add_special_tokens=False)[0],
            'tech': tokenizer.encode('tech', add_special_tokens=False)[0]
            }
        # print({category: tokenizer.decode([token_id]) for category, token_id in category_token_ids.items()})
        # raise Exception('stop')
        # Handle DDP model
        actual_model = model.module if hasattr(model, 'module') else model
        device = next(actual_model.parameters()).device
        accumulation = {category: 0 for category in categories}
        # print(accumulation)
        input_batch=torch.zeros(10,max_length,dtype=torch.long,device=device)
        logit_position_batch=torch.zeros(10,dtype=torch.long,device=device)
        for i,prompt in enumerate(test_prompts):
            input_ids = tokenizer(prompt+'<|im_start|>').input_ids[:max_length]
            position = len(input_ids)
            # FIX: use position-1 to get logits that predict the answer token
            logit_position_batch[i] = position - 1
            input_ids += [tokenizer.pad_token_id] * (max_length - len(input_ids))
            input_batch[i] = torch.tensor(input_ids, dtype=torch.long, device=device)
        with torch.no_grad():
            output_logits = actual_model(input_batch).logits
        
        for i in range(10):
            category_scores = {}
            for cat, token_id in category_token_ids.items():
                if token_id is not None:
                    category_scores[cat] = output_logits[i][logit_position_batch[i]][token_id].item()
        
            # 选择分数最高的类别作为预测
        
            predicted_category = max(category_scores, key=category_scores.get)
            accumulation[predicted_category] += 1
        return max(accumulation, key=accumulation.get)
    correct = 0
    total = 0
    wrong_samples = []  # 存储错误的题目
    for idx in range(len(test_prompts)):
        
        title=test_prompts[idx].split('\nbody: ')[0]
        body=test_prompts[idx].split('\nbody: ')[1]
        total_length=len(body)
        chunks = split_text_into_chunks(body, target_words=200, overlap_words=max(10,(2000-total_length)//9))
        temp_test_prompts = []
        temp_test_labels = []
        for i in range(10):
            chunk=chunks[i%len(chunks)]
            temp_test_prompts.append(f'{title}\nbody: {chunk}<|im_start|>')
            temp_test_labels.append(test_labels[idx])
        predicted_category=predict(model, tokenizer, args.max_new_tokens, temp_test_prompts, temp_test_labels)
        if predicted_category == test_labels[idx]:
            correct += 1
        else:
            # 保存错误的题目
            wrong_sample = {
                'text': test_prompts[idx],
                'label': test_labels[idx],
                'predicted': predicted_category,
                'index': idx
            }
            wrong_samples.append(wrong_sample)
        total += 1
        if idx%100:
            print(f'Accuracy: {correct}/{total} = {correct/total}')
    print(f'Accuracy: {correct}/{total} = {correct/total}')
    
    # 保存错误的题目到jsonl文件
    if wrong_samples:
        output_file = 'wrong_predictions.jsonl'
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in wrong_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        print(f'已保存 {len(wrong_samples)} 个错误题目到 {output_file}')
if __name__ == '__main__':
    main()