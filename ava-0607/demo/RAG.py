import json
import glob
from sentence_transformers import SentenceTransformer
from zlai.llms import Zhipu, ZhipuGLM4
from Retrieve import Search
from Prompt import KnowledgeAgent

model = SentenceTransformer("demo/bge-m3")
llm = Zhipu(generate_config=ZhipuGLM4())

embedding_path = glob.glob("demo/embeddings-normalized/*")
documents = []

for embed in embedding_path:
    documents += json.load(open(embed, 'r', encoding='utf-8'))

chunked = [d[1] for d in documents]
embeds = [d[-1] for d in documents]
retrieve = Search(chunked, embeds, model)

knowledge = KnowledgeAgent(
    llm=llm, verbose=True, search=retrieve, recall_nums=5)

task_completion = knowledge("director的文档指南是什么？")

questions = []
with open("demo/question.jsonl", 'r', encoding='utf-8') as f:
    for line in f:
        questions.append(json.loads(line))

answers = []
for quest in questions:
    answer = knowledge(quest['query'])
    quest["answer"] = answer.content
    answers.append(quest)

with open('demo/answers.jsonl', 'w', encoding='utf-8') as f:
    for item in answers:
        json.dump(item, f, ensure_ascii=False, indent=4)
        f.write('\n')
