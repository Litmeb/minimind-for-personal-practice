# minimind-for-personal-practice

这个项目是个人基于MiniMind进行的一些探索

## 目录

- [minimind-for-personal-practice](#minimind-for-personal-practice)
  - [目录](#目录)
  - [运行之前的预处理](#运行之前的预处理)
  - [核心目标是做分类任务](#核心目标是做分类任务)
    - [1、LoRA训练](#1lora训练)
    - [2、Prompt Tuning](#2prompt-tuning)
    - [3、Projection Head分类器](#3projection-head分类器)
    - [4、Adapter训练](#4adapter训练)
  - [其他功能](#其他功能)

---

## 运行之前的预处理

把这个仓库的文件替换原文件夹中的同名文件，新的文件就直接放进对应位置，然后把数据集下载下来放进dataset文件夹，进入data_process.ipynb处理数据

## 核心目标是做分类任务

尝试的方法包括：

### 1、LoRA训练

运行命令：`python train/train_lora.py --rank 16 --epochs 15`

**实现方式：**

- 没改动什么，直接用MiniMind的LoRA训练代码
- 改了一个bug：原来是只对q_proj、o_proj应用LoRA，现在改成对所有q_proj、k_proj、v_proj、o_proj应用LoRA
- 增加了train/train_lora.py中的传入参数rank，使其能够训练不同rank的LoRA
- 调整了model/model_lora.py中的apply_lora函数，使其能够正确地应用LoRA的rank

### 2、Prompt Tuning

运行命令：`python train/train_prompt_tuning.py`

**实现方式：**

- 在序列开头添加可训练的virtual token embeddings（虚拟token嵌入）
- 默认添加20个virtual tokens（可通过--num_virtual_tokens参数调整）
- 使用自定义的PromptTuningModel包装器，在forward时将序列开头的placeholder tokens替换为可训练的virtual embeddings
- 数据预处理时在序列开头添加num_virtual_tokens个placeholder token IDs，训练时在embedding层替换为virtual embeddings
- 修改了model_minimind.py的forward函数，增加了inputs_embeds参数支持，以便直接传入替换后的embeddings

### 3、Projection Head分类器

运行命令：`python train/train_projection_head.py`

**实现方式：**

- 在序列开头添加一个可训练的`<cls>` token embedding（分类标记嵌入）
- 使用ClsTunedModel包装器，在forward时将序列第一个位置的embedding替换为可训练的cls_embedding
- 创建一个两层的projection head分类器：hidden_size -> 128 -> num_classes
- projection head包含LayerNorm层归一化和Dropout（0.1）正则化
- 冻结所有原始模型参数，只训练cls_embedding和projection head参数
- 使用分类任务训练（CrossEntropyLoss），而不是生成式训练
- 前向传播流程：获取序列第一个位置的hidden state（即`<cls>` token的表示） -> 通过projection head -> 得到分类logits
- 训练完成后保存cls_embedding、projection head权重和label2id映射文件
- 修改了model/model_minimind.py中的forward函数，使其能够适配prompt tuning里面的替换开头embedding的操作（增加了新的参数inputs_embeds）

### 4、Adapter训练

运行命令：`python train/train_adapter.py`

**实现方式：**

- 在每个Transformer层的MiniMindBlock中插入两个Adapter模块
- 每个Adapter是一个两层的MLP：hidden_size -> middle_features -> hidden_size（默认middle_features=8，可通过--middle_features调整）
- Adapter使用GELU激活函数
- 在每一层Transformer block中添加adapter1和adapter2两个Adapter
- 冻结所有原始模型参数，只训练Adapter参数
- 使用生成式训练（语言模型损失），支持动态损失掩码
- Adapter参数数量：每个block有2个Adapter，每个Adapter包含两个线性层，总参数量约为 2 × num_layers × (2 × hidden_size × middle_features)
- 训练完成后只保存Adapter权重

## 其他功能

1. 写了eval_llm_metrics.py可以评估模型的分类能力
    
    运行命令：`python eval_llm_metrics.py (--weight 权重名称) (--lora_weight LoRA权重名称) (--adapter_weight Adapter权重名称) (--prompt_tuning 20) (--projectionhead) (--projectionhead_cls_tuning) (--cls_tuning_weight cls_tuning_projection_head_classifier) (--rank 8) (--middle_features 8)`

2. 增加了llm_as_a_judge.py，用于评估模型的分类能力（虽然最后因为太贵了没用上）

    运行命令：`python eval_llm_metrics.py --llm_as_a_judge --deepseek_api_key`（在这里填入你的api key）

3. 实现了注意力的读取，可以进attentionvisualize.ipynb中试试，不过功能不完善