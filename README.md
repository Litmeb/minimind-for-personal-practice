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
    - [1. 模型评估脚本（eval\_llm\_metrics.py）](#1-模型评估脚本eval_llm_metricspy)
    - [2. LLM as a Judge评估（llm\_as\_a\_judge.py）](#2-llm-as-a-judge评估llm_as_a_judgepy)
    - [3. 注意力可视化（attentionvisualize.ipynb）](#3-注意力可视化attentionvisualizeipynb)

---

## 运行之前的预处理

把这个仓库的文件替换原文件夹中的同名文件，新的文件就直接放进对应位置，然后把数据集下载下来放进dataset文件夹，进入data_process.ipynb处理数据

## 核心目标是做分类任务

尝试的方法包括：

### 1、LoRA训练

运行命令：`python train/train_lora.py --rank 16 --epochs 15 --rank_eval_steps 0`

**实现方式：**

- 没改动什么，直接用MiniMind的LoRA训练代码
- 改了一个bug：原来是只对q_proj、o_proj应用LoRA，现在改成对所有q_proj、k_proj、v_proj、o_proj应用LoRA
- 增加了train/train_lora.py中的传入参数rank，使其能够训练不同rank的LoRA
- 调整了model/model_lora.py中的apply_lora函数，使其能够正确地应用LoRA的rank

**基于SVD分解的自适应rank调整（新增功能）：**

实现了基于随机SVD（Randomized SVD）的自适应rank选择机制，可以根据每层的梯度特征自动确定最优的LoRA rank，避免对所有层使用统一rank造成的参数浪费。

**核心思路：**

1. **梯度收集阶段**：在训练开始前，先进行少量batch的前向和反向传播，收集各目标层（q_proj、k_proj、v_proj、o_proj）的梯度矩阵

2. **随机SVD分解**：对每个层的梯度矩阵进行随机SVD分解
   - 使用随机投影矩阵Ω进行降维加速
   - 进行q次幂迭代提升精度（通常q=1-2）
   - 计算截断后的奇异值和能量分布

3. **rank自动确定**：基于能量覆盖率（energy coverage）阈值τ确定最优rank
   - 累积能量覆盖率 = Σ(σ²ᵢ) / Σ(σ²ⱼ)，其中σ为奇异值
   - 找到覆盖率达到τ的最小rank（默认τ=0.95，即保留95%的能量）
   - 为每层生成个性化的rank值

4. **rank_map应用**：将生成的rank映射应用到LoRA初始化
   - 保存rank_map到JSON文件，便于后续使用和复现
   - 在apply_lora时使用rank_map为不同层分配不同rank

**使用方法：**

```bash
# 使用SVD自适应rank（推荐）
python train/train_lora.py --rank_eval_steps 30 --svd_tau 0.95 --svd_k 64 --svd_q 2

# 参数说明：
# --rank_eval_steps: 用于估计rank的batch数量（0表示跳过，使用固定rank）
# --svd_tau: 能量覆盖率阈值（默认0.95，即保留95%梯度能量）
# --svd_k: 随机SVD截断奇异值数（默认64，影响计算效率和精度）
# --svd_q: 幂迭代次数（默认2，建议1-2，次数越多精度越高但速度越慢）
```

**优势：**

- **参数效率**：不同层使用不同rank，避免参数浪费
- **性能优化**：在保持性能的前提下最小化可训练参数量
- **自动化**：无需手动调参，根据数据特征自动选择最优配置
- **可复现**：rank_map保存为JSON，便于复现和对比实验

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

### 1. 模型评估脚本（eval_llm_metrics.py）

实现了统一的模型评估脚本，支持多种参数高效微调方法的评估：

**功能特点：**

- 支持加载基础模型权重（--weight）
- 支持LoRA权重评估（--lora_weight + --rank）
- 支持Adapter权重评估（--adapter_weight + --middle_features）
- 支持Prompt Tuning评估（--prompt_tuning num_virtual_tokens）
- 支持Projection Head分类器评估（--projectionhead 或 --projectionhead_cls_tuning + --cls_tuning_weight）
- 支持多种评估模式：生成式分类（让模型生成类别名称）和直接分类（使用projection head）

**运行命令示例：**

```bash
# 评估LoRA模型
python eval_llm_metrics.py --weight full_sft --lora_weight lora_classifier --rank 16

# 评估Adapter模型
python eval_llm_metrics.py --weight full_sft --adapter_weight adapter_classifier --middle_features 8

# 评估Prompt Tuning模型
python eval_llm_metrics.py --weight full_sft --prompt_tuning 20

# 评估Projection Head分类器
python eval_llm_metrics.py --weight full_sft --projectionhead_cls_tuning --cls_tuning_weight cls_tuning_projection_head_classifier
```

### 2. LLM as a Judge评估（llm_as_a_judge.py）

使用大语言模型作为评判者来评估分类任务的准确性。实现了一个可选的评估模块，通过调用外部API（如DeepSeek）来判断模型生成的分类结果是否正确。

**注意：** 由于API调用成本较高，此功能在实际项目中未使用，仅作为参考实现。

**运行命令：**

```bash
python eval_llm_metrics.py --llm_as_a_judge --deepseek_api_key YOUR_API_KEY
```

### 3. 注意力可视化（attentionvisualize.ipynb）

实现了注意力权重的读取和可视化功能，可以在Jupyter Notebook中查看模型在处理文本时的注意力分布。

**功能说明：**

- 支持加载训练好的模型（包括LoRA等变体）
- 可以提取和可视化Transformer各层的注意力权重
- 帮助理解模型的关注重点和决策过程
- **注意：** 当前功能尚不完善，仍在开发中

**使用方法：**

在Jupyter Notebook中打开`attentionvisualize.ipynb`，配置模型路径和参数后运行即可。
