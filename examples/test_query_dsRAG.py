import sys
sys.path.append("../")
# sys.path.append("/home/bqc/project/dsRAG")
from dsrag.knowledge_base import KnowledgeBase
from dsrag.document_parsing import extract_text_from_pdf
from dsrag.rse import get_best_segments
import cohere
import os
from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt
import requests
import json

# set API keys
import os
os.environ["OPENAI_API_KEY"] = "sk-33lvzHShmlPJY9FFlRvkIa0Sr6JCiMRNKoHm6UEt2Bgeq0x2"
os.environ["CO_API_KEY"] = ""

# load in some data
file_path = "/home/bqc/project/dsRAG/tests/data/base_goal.txt"
doc_id = os.path.basename(file_path) # grab the file name without the extension so we can use it as the doc_id
print("doc_id:",doc_id)
# kb_id = "nike_10k"
kb_id = "base_goal"

if file_path.endswith(".pdf"):
    document_text = extract_text_from_pdf(file_path)
else:
    with open(file_path, "r") as f:
        document_text = f.read()
# 只是读取文档的过程
# print (document_text[:1000])

# load in chunks
# kb = KnowledgeBase(kb_id=kb_id, exists_ok=True, storage_directory='examples/example_kb_data')
kb = KnowledgeBase(kb_id=kb_id, exists_ok=True, storage_directory='/home/bqc/project/dsRAG')
print("什么是kb:",kb)
num_chunks = len(kb.chunk_db.data[doc_id])

chunks = []
for i in range(num_chunks):
    chunk = {
        "section_title": kb.chunk_db.get_section_title(doc_id, i),
        "chunk_text": kb.chunk_db.get_chunk_text(doc_id, i),
    }

    chunks.append(chunk)

print("打印分区的总数",len(chunks))
print("打印第2分区的内容",chunks[2])
print("打印全部的分区标题section_title:")
# print all section titles
unique_section_titles = []
for i in range(num_chunks):
    section_title = chunks[i]["section_title"]
    if section_title not in unique_section_titles:
        print (section_title)
        unique_section_titles.append(section_title)


if kb_id == "levels_of_agi":
    document_context = "Document: Levels of AGI"
elif kb_id == "nike_10k":
    document_context = "Document: Nike 10-K FY2023"
elif kb_id=="base_goal":
    document_context="员工考核准则"
else:
    document_context = "Document: Unknown"

documents = []
documents_no_context = [] # baseline for comparison
for i in range(num_chunks):
    section_context = f"Section: {chunks[i]['section_title']}"
    chunk_text = chunks[i]["chunk_text"]
    document = f"{document_context}\n{section_context}\n\n{chunk_text}"
    documents.append(document)
    documents_no_context.append(chunk_text)
print("--------------------查询索引index=2的文本内容-------------------------")
chunk_index_to_inspect = 2
print (documents[chunk_index_to_inspect])
print("------------------- 查询索引index=2的文本内容-------------------------")

# 如果有cohere/rerank模型可以进行对[索引=2]的文档进行rerank查询
def transform(x):
    a, b = 0.4, 0.4  # These can be adjusted to change the distribution shape
    return beta.cdf(x, a, b)

# 调用xinference的rerank模型
def rerank_documents_xinference(url, model, query, documents):
    """
    发送 POST 请求到指定的 URL 进行重排操作。

    参数:
    url (str): 重排服务的 URL。
    model (str): 模型的 UID。
    query (str): 查询语句。
    documents (list): 需要重排的文档列表。

    返回:
    dict: 重排结果。
    """
    # 定义请求的 headers
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    
    # 定义请求的 payload
    data = {
        "model": model,
        "query": query,
        "documents": documents
    }
    
    # 将 payload 转换为 JSON 格式的字符串
    payload = json.dumps(data)
    
    # 发送 POST 请求
    response = requests.post(url, headers=headers, data=payload)
    
    # 检查响应状态码
    if response.status_code == 200:
        # 解析响应内容
        return response.json()
    else:
        # 打印错误信息并返回 None
        print(f"Error: {response.status_code}, {response.text}")
        return None

def rerank_documents(query: str, documents: list) -> list:
    """
    Use Cohere Rerank API to rerank the search results
    """
    # model = "bge-reranker-large"
    # client = cohere.Client(api_key=os.environ["CO_API_KEY"])
    decay_rate = 30
    reranked_results = rerank_documents_xinference(url="http://10.3.24.40:9997/v1/rerank",
                                                   model="bge-reranker-large",
                                                   query=query, 
                                                   documents=documents)
    results = reranked_results['results']
    print("打印输出检索后重排的得分结果和段落索引:",results)
    reranked_indices = [result['index'] for result in results]
    # print("rerankde_indices:",reranked_indices)
    reranked_similarity_scores = [result['relevance_score'] for result in results] # in order of reranked_indices
    
    # convert back to order of original documents and calculate the chunk values
    similarity_scores = [0] * len(documents)
    chunk_values = [0] * len(documents)
    for i, index in enumerate(reranked_indices):
        absolute_relevance_value = transform(reranked_similarity_scores[i])
        similarity_scores[index] = absolute_relevance_value
        v = np.exp(-i/decay_rate)*absolute_relevance_value # decay the relevance value based on the rank
        chunk_values[index] = v

    return similarity_scores, chunk_values

query = "工作态度是什么"
print("--------------本次打印输出:添加header是否影响检索效果---------------")
# run this chunk through the Cohere Rerank API with and without the context header
similarity_scores, chunk_values = rerank_documents(query, [documents[chunk_index_to_inspect], documents_no_context[chunk_index_to_inspect]])

print (f"Similarity with contextual chunk header: {similarity_scores[0]}")
print (f"Similarity without contextual chunk header: {similarity_scores[1]}")

# be sure you're using the Nike 10-K KB for these next few cells, as we'll be focusing on a single example query for that document

print("--------------本次打印输出:直接根据query和document进行重排---------------")
query = "工作态度是什么"
similarity_scores, chunk_values = rerank_documents(query, documents)
print("similarity_scores, chunk_values的分数和数值:",similarity_scores, chunk_values)

plt.figure(figsize=(12, 5))
plt.title(f"Similarity of each chunk in the document to the search query")
plt.ylim(0, 1)
plt.xlabel("Chunk index")
plt.ylabel("Query-chunk similarity")
plt.scatter(range(len(chunk_values)), chunk_values)
plt.show()

irrelevant_chunk_penalty = 0.2
all_relevance_values = [[v - irrelevant_chunk_penalty for v in chunk_values]]
document_splits = []
max_length = 30
overall_max_length = 50
minimum_value = 0.5

# get_best_segments solves a constrained version of the maximum sum subarray problem
best_segments, scores = get_best_segments(all_relevance_values, document_splits, max_length, 
                                          overall_max_length, minimum_value)
print("打印输出获取最好的段落")
print (best_segments)
print (scores)
print ()

# print the best segments
for segment_start, segment_end in best_segments:
    # concatenate the text of the chunks in the segment
    segment_text = f"[{document_context}]\n"
    print("什么是segment_text:",segment_text)
    for i in range(segment_start, segment_end):
        print(f"什么是chunks[{i}]和chunk_text:",chunks[i],'\n---\n',chunk_text)
        chunk_text = chunks[i]["chunk_text"]
        segment_text += chunk_text + "\n"

    print (segment_text)
    print ("\n---\n")

# plot the relevance values of the best segment
best_segment_chunk_indexes = list(range(best_segments[0][0], best_segments[0][1]))
best_segment_chunk_values = chunk_values[best_segments[0][0]:best_segments[0][1]]
print("打印段落索引",best_segment_chunk_indexes)
print("打印段落得分",best_segment_chunk_values)
print ("\n---\n")

plt.figure(figsize=(12, 5))
plt.title(f"Relevance values of the best segment")
plt.ylim(0, 1)
plt.xlabel("Chunk index")
plt.ylabel("Query-chunk similarity")
plt.scatter(best_segment_chunk_indexes, best_segment_chunk_values)
plt.show()

# print the individual chunks in the best segment - annotated with their chunk indexes and relevance values
for chunk_index in best_segment_chunk_indexes:
    chunk_text = chunks[chunk_index]["chunk_text"]
    chunk_value = chunk_values[chunk_index]
    print (f"Chunk index: {chunk_index} - Relevance value: {chunk_value}")
    print (chunk_text)
    print ("\n---\n")