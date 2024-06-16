import os
import json
import glob
from sentence_transformers import SentenceTransformer
from zlai.llms import Zhipu, GLM4GenerateConfig
from Retrieve import Search
from Prompt import KnowledgeAgent
from tqdm import tqdm
import time

os.environ["ZHIPU_API_KEY"] = "xxx"

print("加载模型....")
model = SentenceTransformer("demo/model/bge-m3")
llm = Zhipu(generate_config=GLM4GenerateConfig())

print("模型加载成功，加载数据....")
embedding_path = glob.glob("demo/embeddings-normalized/*")
documents = []
for embed in embedding_path:
    documents += json.load(open(embed, 'r', encoding='utf-8'))
chunked = [d[1] for d in documents]
embeds = [d[-1] for d in documents]
chunked_tokens = json.load(open("demo/chunked/total_chunked_tokens.json", 'r', encoding='utf-8'))

retrieve = Search(chunked, embeds, model, chunked_tokens, top=5)

print("数据加载成功，准备生成...")
knowledge = KnowledgeAgent(
    llm=llm,
    verbose=False,
    log_path="answers/0616_log.txt",
    search=retrieve,
    recall_nums=5)

# task_completion = knowledge("Npcf_SMPolicyControl服务包含哪些操作？")

questions = []
with open("demo/question.jsonl", 'r', encoding='utf-8') as f:
    for line in f:
        questions.append(json.loads(line))

answers = []
with open('demo/answers/answers.jsonl', 'w+', encoding='utf-8') as f:
    for quest in tqdm(questions):
        try:
            answer = knowledge(quest['query'])
        except TimeoutError:
            time.sleep(15)
            answer = knowledge(quest['query'])

        quest["answer"] = answer.content
        answers.append(quest)
        json.dump(quest, f, ensure_ascii=False)
        f.write('\n')

# with open('./answers.jsonl', 'w', encoding='utf-8') as f:
#     for item in answers:
#         json.dump(item, f, ensure_ascii=False)
#         f.write('\n')
