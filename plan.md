# 项目背景
针对医疗领域的多模态准确问答，构建一个集成文本、图像和结构化数据的智能问答系统，提升医疗信息检索和患者咨询的效率与准确性。

# model
qwen3-vl-4B, medgemma-4B
# data
基于MedVL-thinker和Med-Max数据集
# benchmark
- 文本：MedQA, MMLU-med
- 图像分类：mimic-cxr,chesstX-ray14
- VQA：MedXpertQA, VQA-RAD
# method
## 1、naive method
基于VLM直接进行测试
### + naive RAG
## 2、微调方法测试
## 3、多模态RAG构建测试
需要检索病例文本和图像信息，结合生成模型进行问答。
分别检索文本和图像信息，结合生成模型进行问答。
统一图像文本空间，通过对比学习微调embedding提高召回率。
## 4、agentic RAG训练后测试