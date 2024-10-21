from dsrag.create_kb import create_kb_from_file
# set API keys
import os
os.environ["OPENAI_API_KEY"] = "sk-33lvzHShmlPJY9FFlRvkIa0Sr6JCiMRNKoHm6UEt2Bgeq0x2"
os.environ["CO_API_KEY"] = ""
os.environ["DSRAG_OPENAI_BASE_URL"] = "https://api.chatanywhere.tech/v1"

file_path = "/home/bqc/project/dsRAG/tests/data/领克汽车使用手册.pdf"
kb_id = "car"

# 如果没有创建kb知识块，默认设置exists_ok=false
kb = create_kb_from_file(kb_id, file_path)
print("输出kb:",kb)
print("看看是什么内容:",kb.chunk_db.data)


# 如果文件已经存在，那么执行加载文件的代码
# from dsrag.knowledge_base import KnowledgeBase

# kb = KnowledgeBase("base_goal")
# search_queries = ["工作态度是？"]
# results = kb.query(search_queries)
# for segment in results:
#     print(segment)