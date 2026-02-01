# #!/bin/bash

# # 安装gdown(如果未安装)
# pip install gdown

# # 从Google Drive下载文件 （harward fairVLMed）
# FILE_ID="1n97RmUl8T8sqbVe6NaS4COShDbbAa8Sw"
# gdown "https://drive.google.com/uc?id=${FILE_ID}" -O downloaded_file.zip

# # 创建datasets目录(如果不存在)
# mkdir -p datasets

# # 解压到datasets文件夹
# unzip downloaded_file.zip -d datasets

# # 删除下载的zip文件
# rm downloaded_file.zip

# echo "下载并解压完成！"

# # 下载第二个文件
# FILE_ID_2="1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg"
# gdown "https://drive.google.com/uc?id=${FILE_ID_2}" -O downloaded_file2.zip

# # 解压第二个文件到datasets文件夹
# unzip downloaded_file2.zip -d datasets

# # 删除第二个zip文件
# rm downloaded_file2.zip

# echo "第二个文件下载并解压完成！"

# 创建目录
# mkdir -p ./datasets/medmax

# # 下载整个数据集
# huggingface-cli download \
#   --repo-type dataset \
#   mint-medmax/medmax_data \
#   --local-dir ./datasets/medmax
mkdir -p ./datasets/pmc-oa

# 下载整个数据集
hf download \
  --repo-type dataset \
  axiong/pmc_oa \
  --local-dir ./datasets/pmc-oa