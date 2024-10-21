from dsrag.document_parsing import extract_text_from_pdf
from dsrag.llm import OpenAIChatAPI
from dsrag.reranker import NoReranker
from dsrag.knowledge_base import KnowledgeBase
llm = OpenAIChatAPI(model='gpt-4o-mini')
reranker = NoReranker()

kb = KnowledgeBase(kb_id="levels_of_agi", reranker=reranker, auto_context_model=llm)
file_path = "dsRAG/tests/data/levels_of_agi.pdf"
text = extract_text_from_pdf(file_path)
kb.add_document(doc_id=file_path, text=text)

