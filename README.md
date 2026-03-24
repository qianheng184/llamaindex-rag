# RAG System v2.0 - 智能问答系统

一个基于 RAG（检索增强生成）技术的智能问答系统，采用模块化、可扩展的架构设计。

## 🎯 核心特性

### ✨ v2.0 新特性

- **服务层解耦**：IndexingService, RetrievalService, GenerationService, CacheService 独立运作
- **推理层抽象**：支持 DeepSeek、Ollama 等多种 LLM Provider
- **依赖注入**：所有服务通过构造函数注入，便于测试和替换
- **流式输出**：完整的 SSE (Server-Sent Events) 支持
- **向量数据库升级**：ChromaDB 替换 JSON 存储
- **Rerank 模块**：支持重排序提高相关性
- **缓存服务**：内存缓存加速重复查询
- **CLI 工具**：离线索引管理和交互式查询客户端

## 📁 项目结构

```
llamaindex-rag/
│
├── api/                        # HTTP API 层
│   ├── __init__.py
│   ├── routes.py              # REST 路由
│   └── sse.py                 # SSE 流式处理
│
├── orchestrator/              # 编排层
│   ├── __init__.py
│   └── rag_orchestrator.py    # RAG 流程编排
│
├── services/                  # 服务层
│   ├── __init__.py
│   ├── indexing_service.py    # 离线索引服务
│   ├── retrieval_service.py   # 在线检索服务
│   ├── generation_service.py  # 生成服务
│   └── cache_service.py       # 缓存服务
│
├── core/                      # 核心组件层
│   ├── __init__.py
│   ├── embedder.py           # Embedding 编码器
│   ├── vector_store.py       # 向量存储抽象
│   ├── vector_store_chroma.py # ChromaDB 实现
│   ├── reranker.py          # Rerank 模块
│   └── inference/           # 推理层
│       ├── __init__.py
│       ├── llm_provider.py   # LLM 抽象
│       ├── deepseek_client.py
│       └── ollama_client.py
│
├── models/                    # 数据模型层
│   ├── __init__.py
│   └── schemas.py            # Pydantic 模型
│
├── events/                    # 事件系统
│   └── __init__.py
│
├── utils/                     # 工具函数
│   ├── __init__.py
│   └── config.py             # 配置加载
│
├── data/                      # 原始文档
│   ├── pdf/
│   ├── txt/
│   └── docs/
│
├── database/                  # 持久化存储
│   └── chroma_db/
│
├── config/
│   ├── __init__.py
│   └── config.yaml           # 配置文件
│
├── main.py                    # Web API 入口
├── cli_index.py              # 离线索引 CLI
├── cli_query.py              # 在线查询 CLI
├── test_init.py              # 组件初始化测试
├── test_v2.py                # 完整功能测试
├── requirements.txt          # 依赖清单
├── QUICKSTART.md            # 快速开始指南
└── README.md                # 本文档
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

**要求**：Python 3.10+

### 2. 配置环境变量

在项目根目录创建 `.env` 文件：

```bash
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_API_BASE=https://api.deepseek.com/v1
```

### 3. 构建索引

```bash
python cli_index.py build
```

这将：
- 从 `data/` 目录读取文档
- 使用 llama-index 加载和切分文档
- 调用 embedding 模型生成向量
- 保存到 ChromaDB 向量数据库

### 4. 启动 Web API 服务

```bash
python main.py
```

服务将启动在 `http://localhost:8000`
 
### 5. 使用 CLI 客户端

在另一个终端窗口：

```bash
python cli_query.py
```

进入交互式问答界面。

## 📖 API 接口

### 健康检查

```bash
curl http://localhost:8000/health
```

### 普通查询

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "你的问题", "top_k": 5}'
```

### 流式查询（SSE）

```bash
curl -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "你的问题"}'
```

### 获取统计信息

```bash
curl http://localhost:8000/stats
```

### 索引管理

#### 构建索引

```bash
curl -X POST http://localhost:8000/index/build \
  -H "Content-Type: application/json" \
  -d '{"force_rebuild": false}'
```

#### 删除索引

```bash
curl -X DELETE http://localhost:8000/index
```

#### 查看状态

```bash
curl http://localhost:8000/index/status
```

## 🔧 配置说明

编辑 `config/config.yaml`：

```yaml
system:
  name: "RAG System"
  version: "2.0"
  log_level: "INFO"

indexing:
  data_dir: "./data"
  chunk_size: 500
  chunk_overlap: 50
  batch_size: 32

retrieval:
  top_k: 5
  rerank_top_k: 10
  use_hybrid_search: true

embedding:
  model: "BAAI/bge-small-zh-v1.5"
  dimension: 512

inference:
  provider: "deepseek"  # 可选：deepseek, ollama
  providers:
    deepseek:
      api_key: "${DEEPSEEK_API_KEY}"
      model: "deepseek-chat"
      base_url: "https://api.deepseek.com/v1"
    ollama:
      base_url: "http://localhost:11434"
      model: "qwen2.5:1.5b"

database:
  type: "chromadb"
  path: "./database/chroma_db"

cache:
  enabled: true
  type: "memory"
  ttl: 3600

api:
  host: "0.0.0.0"
  port: 8000
  enable_stream: true
```

## 🛠️ CLI 工具使用

### cli_index.py - 索引管理

```bash
# 构建索引
python cli_index.py build

# 强制重建
python cli_index.py build --force

# 更新索引
python cli_index.py update

# 删除索引
python cli_index.py delete

# 查看状态
python cli_index.py status
```

### cli_query.py - 交互式查询

```bash
# 进入交互模式
python cli_query.py

# 直接提问
python cli_query.py -q "你的问题"

# 指定 API 地址
python cli_query.py -u http://localhost:8000

# 禁用流式
python cli_query.py -q "你的问题" --no-stream
```

## 🏗️ 架构设计

### 核心分层

1. **API 层** (`api/`): HTTP 接口，负责接收请求和返回响应
2. **编排层** (`orchestrator/`): 协调各个服务完成复杂流程
3. **服务层** (`services/`): 核心业务逻辑，独立可测试
4. **核心组件层** (`core/`): 基础组件，可插拔实现
5. **数据模型层** (`models/`): 数据结构定义

### 依赖注入示例

```python
# 初始化核心组件
embedder = Embedder(model_name, dimension)
vector_store = ChromaVectorStore(persist_dir)
llm_provider = create_llm_provider(provider_type, config)

# 初始化服务层（依赖注入）
retrieval_service = RetrievalService(
    embedder=embedder,
    vector_store=vector_store,
    reranker=reranker
)

generation_service = GenerationService(
    llm_provider=llm_provider
)

# 初始化编排层
orchestrator = RAGOrchestrator(
    retrieval_service=retrieval_service,
    generation_service=generation_service,
    cache_service=cache_service
)
```

## 🧪 测试

### 组件初始化测试

```bash
python test_init.py
```

### 完整功能测试

```bash
python test_v2.py
```

## 📊 性能优化建议

1. **批量处理**：索引构建时使用较大的 `batch_size`
2. **缓存启用**：对重复查询启用缓存
3. **混合检索**：启用 BM25 + 向量混合检索（待实现）
4. **异步处理**：索引构建使用后台任务

## 🔮 未来计划

- [ ] 实现 BM25 混合检索
- [ ] 集成 FlagEmbedding Rerank 模型
- [ ] Redis 缓存支持
- [ ] 多模态文档支持（图片、表格）
- [ ] 对话历史管理
- [ ] 评估系统
- [ ] Docker 容器化
- [ ] 监控和日志系统

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License

## 🙏 致谢

- [LlamaIndex](https://github.com/run-llama/llama_index)
- [ChromaDB](https://github.com/chroma-core/chroma)
- [FastAPI](https://github.com/tiangolo/fastapi)
- [SentenceTransformers](https://github.com/UKPLab/sentence-transformers)

---

**提示**：更多详细信息请查看 [QUICKSTART.md](QUICKSTART.md)
