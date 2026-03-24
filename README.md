# RAG 系统 - 模块化实现

一个结构清晰、模块解耦、可扩展的 RAG（检索增强生成）系统。

## 项目特点

- ✅ **离线索引构建与在线检索完全解耦**
- ✅ **检索与生成 pipeline 显式实现**，不依赖黑盒框架
- ✅ **HTTP API 服务**
- ✅ **CLI 客户端**
- ✅ **离线评估系统**
- ✅ **YAML 配置系统**

## 项目结构

```
llamaindex-rag/
│
├── api/
│   └── app.py              # FastAPI 服务入口
│
├── engine/
│   ├── __init__.py
│   └── rag_engine.py       # 在线检索与生成 pipeline
│
├── index/
│   ├── __init__.py
│   ├── build_index.py      # 离线索引构建脚本
│   ├── loader.py           # 文档加载
│   ├── chunker.py          # 文本切分
│   └── vector_store.py     # 向量存储与检索
│
├── evaluation/
│   ├── evaluator.py        # 离线评估脚本
│   └── dataset.json        # 测试集
│
├── client/
│   └── cli_chat.py         # CLI 客户端
│
├── config/
│   ├── __init__.py
│   └── config.yaml         # 配置文件
│
├── data/                   # 原始文档
│
├── storage/                # 向量存储（自动生成）
│
├── main.py                 # 调试入口
└── requirements.txt        # 依赖清单
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

在项目根目录创建 `.env` 文件：

```bash
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_API_BASE=https://api.deepseek.com/v1
```

### 3. 准备数据

将文档（PDF、TXT 等格式）放入 `data/` 目录

### 4. 构建索引

```bash
python index/build_index.py
```

执行后会：
1. 从 `data/` 读取文档
2. 使用 llama-index 加载和切分文档
3. 调用 embedding 模型生成向量
4. 保存到 `storage/` 目录

### 5. 启动 HTTP API 服务

```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

API 端点：
- `GET /health` - 健康检查
- `POST /query` - RAG 查询
  - 请求体：`{"query": "你的问题"}`
  - 响应：`{"answer": "回答内容"}`

### 6. 启动 CLI 客户端

在另一个终端窗口运行：

```bash
python client/cli_chat.py
```

### 7. 运行评估

```bash
python evaluation/evaluator.py
```

会输出：
- 每条样本的详细结果
- 整体准确率和平均得分
- 结果保存到 `evaluation/evaluation_result.json`

## 配置说明

编辑 `config/config.yaml` 调整参数：

```yaml
# Embedding 模型
embedding_model: "BAAI/bge-small-zh-v1.5"

# 文本切分
chunk_size: 500
chunk_overlap: 50

# 检索配置
top_k: 3

# LLM 配置
llm_model: "deepseek-chat"

# 路径配置
vector_store_path: "./storage"
data_path: "./data"

# API 配置
api_host: "0.0.0.0"
api_port: 8000
```

## 模块说明

### Index 模块（离线索引构建）

- **loader.py**: 文档加载器，支持 PDF、TXT 等格式
- **chunker.py**: 文本切分器，智能分割成长度合适的 chunk
- **vector_store.py**: 本地向量存储，使用 JSON 持久化
- **build_index.py**: 索引构建脚本，串联完整流程

### Engine 模块（在线检索引擎）

**RAGEngine 类**提供以下方法：

```python
engine = RAGEngine(config)

# 检索相关文档
contexts = engine.retrieve(query)

# 构建 prompt
prompt = engine.build_prompt(query, contexts)

# 生成回答
answer = engine.generate(prompt)

# 完整 pipeline
answer = engine.query(query)
```

### API 模块（HTTP 服务）

基于 FastAPI 实现：
- 自动启动时加载 RAG 引擎
- 异步处理查询请求
- 完整的错误处理

### Client 模块（CLI 客户端）

- 循环读取用户输入
- 调用 HTTP API
- 友好的交互界面

### Evaluation 模块（离线评估）

- 读取测试集
- 批量执行查询
- 计算准确率指标
- 生成详细报告

## 技术栈

- **Python**: 3.10+
- **Web 框架**: FastAPI
- **向量数据库**: 本地 JSON 存储（可扩展到 FAISS、Milvus 等）
- **Embedding**: sentence-transformers (BAAI/bge-small-zh-v1.5)
- **LLM**: DeepSeek (通过 llama-index 封装)
- **文档处理**: llama-index-core（仅用于文档加载和切分）

## 设计原则

1. **模块化**: 每个模块职责单一，易于理解和维护
2. **解耦**: 离线索引与在线检索完全分离
3. **透明**: 不使用黑盒框架，pipeline 显式实现
4. **可配置**: 所有参数通过 YAML 配置
5. **可扩展**: 易于替换组件（如向量数据库、LLM 等）

## 常见问题

### Q: 如何更换 Embedding 模型？
A: 修改 `config.yaml` 中的 `embedding_model` 字段

### Q: 如何更换 LLM？
A: 修改 `config.yaml` 中的 `llm_model` 字段，并在 `.env` 中配置对应的 API Key

### Q: 向量存储可以扩展吗？
A: 可以。只需实现 `vector_store.py` 中的接口，替换底层存储实现即可

### Q: 如何添加新的文档格式支持？
A: 在 `loader.py` 中添加对应的解析器即可

## License

MIT
