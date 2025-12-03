# minimind-for-personal-practice
这个项目是个人基于MiniMind进行的一些探索
##### 运行之前需要先做预处理：把这个仓库的文件替换原文件夹中的同名文件，新的文件就直接放进对应位置，然后把数据集下载下来放进dataset文件夹，进入data_process.ipynb处理数据
核心目标是做分类任务，尝试的方法包括：
1、LoRA训练
运行命令：python train/train_lora.py --rank 16 --epochs 15
2、Prompt Tuning
运行命令：python train/train_prompt_tuning.py
3、Projection Head分类器
运行命令：python train/train_projection_head.py
4、Adapter训练
运行命令：python train/train_adapter.py


有一些别的修改，在这里说明
1、增加了llm_as_a_judge.py，用于评估模型的分类能力（虽然最后因为太贵了没用上）
运行命令：python eval_llm_metrics.py --llm_as_a_judge --deepseek_api_key（在这里填入你的api key）
2、实现了注意力的可视化，可以进attentionvisualize.ipynb中试试
3、增加了train/train_lora.py中的传入参数rank，使其能够训练不同rank的LoRA；调整了model/model_lora.py中的apply_lora函数，使其能够正确地应用LoRA的rank
4、修改了model/model_minimind.py中的forward函数，使其能够适配prompt tuning里面的替换开头embedding的操作（增加了新的参数inputs_embeds）
