
# 下载embedding model
git clone https://www.modelscope.cn/Xorbits/bge-m3.git /demo/bge-m3

# download dataset
git clone https://www.modelscope.cn/datasets/issaccv/aiops2024-challenge-dataset.git demo/dataset
# 问答数据移到demo
mv demo/dataset/question.jsonl demo/question.jsonl

# process dataset
mv demo/dataset/director.zedx demo/dataset/director.zip
mv demo/dataset/emsplus.zedx demo/dataset/emsplus.zip
mv demo/dataset/rcp.zedx demo/dataset/rcp.zip
mv demo/dataset/umac.zedx demo/dataset/umac.zip
mkdir demo/dataset/multi

unzip demo/dataset/director.zip -d demo/dataset/multi/director
unzip demo/dataset/emsplus.zip -d demo/dataset/multi/emsplus
unzip demo/dataset/rcp.zip -d demo/dataset/multi/rcp
unzip demo/dataset/umac.zip -d demo/dataset/multi/umac

pip3 install -r demo/requirements.txt
echo "数据处理：zedx转zip..."
python demo/zedx2txt.py
echo "数据处理：解析节点..."
python demo/Node.py
echo "数据处理：文本切分..."
python demo/Chunk.py
echo "数据处理：向量化..."
python demo/Embedding
echo "RAG..."
python demo/RAG.py
echo "结束"