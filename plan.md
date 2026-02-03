# 项目背景
针对医疗领域的多模态准确问答，构建一个集成文本、图像和结构化数据的智能问答系统，提升医疗信息检索和患者咨询的效率与准确性。

# model
qwen3-vl-4B, medgemma-4B
# data
基于MedVL-thinker和Med-Max数据集
放射学 - IU-Xray
眼科学 - Harvard-FairVLMed
病理学 - PMC-OA
# benchmark
- 文本：MedQA, MMLU-med
- 图像分类：mimic-cxr,chestX-ray14
- VQA：MedXpertQA, VQA-RAD
# method
## 1、naive method
基于VLM直接进行测试
pmc-vqa100题 = 0.45 

## 3、多模态RAG构建
需要检索病例文本和图像信息，结合生成模型进行问答。
文本-> 文本+图像，结合生成模型进行问答。
### 多路检索机制
1、构建基于BM25的文本检索器
2、基于qwen3-vl-embedding的图像+文本检索器
3、融合两路检索结果，结合生成模型进行问答
### 数据
query可以是image、text
document是image+report
### 微调qwen3-vl-embedding
数据：
- 1、图像问答：通过qwen3-vl-8B或者medgemma-4B对病例图像和文本改写为image+问题格式，检索得到对应image+report作为正样本
- 2、文本检索：改写文本问题，检索得到对应image+report作为正样本
(对于VQA数据集生成report， 对于带有report的数据集生成query)
负样本采样：
- 1、检索top-10样本。令$$ s < s^++\delta $$，选取为负样本，避免FN（假阴性）


统一图像文本空间，通过对比学习微调embedding提高召回率。
主要指标：
Recall@k:
    准备一个测试集 (Query, Gold_Document_ID)，计算你的模型 Top-5 或 Top-10 检索结果中包含 Gold_Document_ID 的比例。
MRR(mean reciprocal rank):
    正确文档的（1/Rank）
    计算测试集中每个查询的平均倒数排名 (Mean Reciprocal Rank)，反映模型在检索任务中的整体表现。

需要

## 4、agentic RAG训练后测试
使用verl框架，使用grpo进行多模态QA rl训练
奖励函数先使用简单的文本匹配
设计两个工具-- 文本检索工具、图像检索工具