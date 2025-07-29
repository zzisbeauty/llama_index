import sys, os

print(sys.executable)  
print(sys.path)  
  
# 检查包是否可用  
import importlib.util  
spec = importlib.util.find_spec("llama_index.llms.ollama")  
print(f"Package found: {spec is not None}")

from llama_index.core import SimpleDirectoryReader

# 方法1：读取单个文件  
documents = SimpleDirectoryReader(input_files=["/home/llama_index/hw_client/testdatas/红楼梦.txt"]).load_data()

# 方法2：读取整个目录（包含txt文件）  
# documents = SimpleDirectoryReader(input_dir="/home/llama_index/my_client").load_data()
# print(documents)

# # 方法3：更多配置选项  
# reader = SimpleDirectoryReader(  
#     input_files=["your_file.txt"],  
#     encoding="utf-8",  # 文件编码  
#     filename_as_id=True,  # 使用文件名作为文档ID  
#     recursive=True,  # 递归读取子目录  
#     exclude_hidden=True,  # 排除隐藏文件  
# )  
# documents = reader.load_data()


""" 对于 txt 文件，LlamaIndex 会直接读取文件内容并创建 Document 对象。
    如果您的 txt 文件很大，建议在创建索引时使用文本分割器来将内容分成更小的块，这样可以提高检索效果。您也可以通过 file_metadata 参数自定义元数据提取逻辑。
"""

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding  


# 配置模型
Settings.embed_model = OllamaEmbedding(model_name="bge-m3:latest", base_url="http://10.30.30.97:11434")
# Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")  
Settings.llm = Ollama(model="qwen3:4b", request_timeout=360, context_window=8000, base_url="http://10.30.30.97:11434", temperature=0.7)


# 对读取的文档创建索引  
index = VectorStoreIndex.from_documents(documents)  
query_engine = index.as_query_engine()  


# 创建索引  
index = VectorStoreIndex.from_documents(documents)  


# 创建查询引擎  
query_engine = index.as_query_engine()  
  
# 现在您可以对文档内容进行查询  
response = query_engine.query("文章主要讲了什么故事")  
print(response)
