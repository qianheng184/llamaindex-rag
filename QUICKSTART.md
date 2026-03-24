# RAG System v2.0 - 快速开始指南

## 项目结构

```
llamaindex-rag/
├── api/                        # HTTP 接口层（待实现）
├── orchestrator/              # 编排层 ✨
│   └── rag_orchestrator.py    # RAG 流程编排
├── services/                  # 服务层 ✨
│   ├── indexing_service.py    # 离线索引服务
│   ├── retrieval_service.py   # 在线检索服务
│   ├── generation_service.py  # 生成服务
│   └── cache_service.py       # 缓存服务
├── core/                      # 核心组件层 ✨
│   ├── document_parser.py     # 文档解析（待实现）
│   ├── chunker.py            # 文本分块（待实现）
│   ├── embedder.py           # Embedding
│   ├── vector_store.py       # 向量存储抽象
│   ├── vector_store_chroma.py # ChromaDB 实现
│   ├── reranker.py          # Rerank
│   └── inference/           # 推理层 ✨
│       ├── llm_provider.py   # LLM 抽象
│       ├── deepseek_client.py
│       └── ollama_client.py
├── models/                    # 数据模型层 ✨
│   └── schemas.py            # Pydantic 模型
├── events/                    # 事件系统（待实现）
├── utils/                     # 工具函数 ✨
│   └── config.py             # 配置加载
├── data/                      # 原始文档 ✨
├── database/                  # 持久化存储 ✨
│   └── chroma_db/
├── config/
│   └── config.yaml           # 配置文件（已更新 v2.0）
├── main_v2.py                 # Web API 入口 ✨
├── test_v2.py                 # 测试脚本 ✨
└── requirements.txt           # 依赖清单（已更新）
```

## 安装依赖

```bash
pip install -r requirements.txt
```

**注意**：需要 Python 3.10+

## 配置环境变量

在项目根目录创建 `.env` 文件：

```bash
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_API_BASE=https://api.deepseek.com/v1
```

## 快速测试

### 1. 运行组件测试

```bash
python test_v2.py
```

这将测试：
- ✓ 配置加载
- ✓ Embedding 模型
- ✓ LLM Provider

### 2. 启动 Web API 服务

```bash
python main_v2.py
```

服务将启动在 `http://localhost:8000`

### 3. 测试 API 端点

#### 健康检查
```bash
curl http://localhost:8000/health
```

#### 普通查询
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "你的问题", "top_k": 5}'
```

#### 流式查询
```bash
curl -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "你的问题"}'
```

#### 获取统计信息
```bash
curl http://localhost:8000/stats
```

## 配置文件说明

编辑 `config/config.yaml` 调整参数：

```yaml
system:
  name: "RAG System"
  version: "2.0"

indexing:
  data_dir: "./data"
  chunk_size: 500
  chunk_overlap: 50

retrieval:
  top_k: 5
  rerank_top_k: 10

embedding:
  model: "BAAI/bge-small-zh-v1.5"
  dimension: 512

inference:
  provider: "deepseek"  # 可选：deepseek, ollama
  providers:
    deepseek:
      api_key: "${DEEPSEEK_API_KEY}"
      model: "deepseek-chat"
    ollama:
      base_url: "http://localhost:11434"
      model: "qwen2.5:1.5b"

database:
  type: "chromadb"
  path: "./database/chroma_db"

cache:
  enabled: true
  ttl: 3600
```

## 核心特性

### 1. 服务层解耦 ✨

- **IndexingService**: 离线索引构建
- **RetrievalService**: 在线检索
- **GenerationService**: LLM 生成
- **CacheService**: 查询缓存

### 2. 推理层抽象 ✨

- **LLMProvider** 抽象基类
- **DeepSeekClient**: DeepSeek API 调用
- **OllamaClient**: 本地 Ollama 模型
- 支持运行时切换 Provider

### 3. 依赖注入 ✨

所有服务通过构造函数注入依赖，便于测试和替换。

### 4. 流式输出 ✨

- SSE (Server-Sent Events) 支持
- `/query/stream` 端点
- 实时显示生成内容

### 5. 向量数据库升级 ✨

- ChromaDB 替换 JSON 存储
- 支持混合检索（向量 + BM25，待实现）
- Rerank 模块（简单策略已实现）

## 架构优势

1. **模块化设计**: 每个组件职责单一，易于维护
2. **依赖注入**: 便于单元测试和组件替换
3. **异步支持**: 流式输出使用 async/await
4. **配置驱动**: 所有参数通过 YAML 配置
5. **向后兼容**: 保留 v1.0 代码作为参考

## 下一步计划

### Phase 4: 完善

- [ ] 创建 CLI 工具（cli_index.py, cli_query.py）
- [ ] 实现完整的 HTTP 路由（api/routes.py）
- [ ] 实现 SSE 流式处理（api/sse.py）
- [ ] 编写 README.md 文档
- [ ] 添加更多测试用例
- [ ] 性能优化

## 常见问题

### Q: 如何切换到 Ollama Provider？
A: 修改 `config.yaml` 中的 `inference.provider` 为 `ollama`

### Q: ChromaDB 数据存储在哪里？
A: 默认在 `./database/chroma_db` 目录

### Q: 如何清空索引？
A: 删除 `database/chroma_db` 目录或调用 API 的删除接口（待实现）

### Q: Rerank 模型可用吗？
A: 当前仅实现简单策略，基于 FlagEmbedding 的 Rerank 待集成

## 开发进度

✅ **Phase 1: 基础架构** - 完成
- [x] 目录结构
- [x] 配置加载
- [x] Pydantic 模型
- [x] LLM Provider 抽象和实现

✅ **Phase 2: 核心服务** - 完成
- [x] Embedding
- [x] VectorStore (ChromaDB)
- [x] Reranker
- [x] CacheService
- [x] RetrievalService
- [x] GenerationService
- [x] IndexingService

✅ **Phase 3: 编排层和 API** - 完成
- [x] RAGOrchestrator
- [x] main_v2.py (Web API)
- [x] requirements.txt
- [x] test_v2.py

⏳ **Phase 4: 完善** - 进行中
- [ ] CLI 工具
- [ ] 完整文档
- [ ] 性能优化

## 许可证

MIT
