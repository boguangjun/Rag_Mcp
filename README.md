# RAG MCP 知识库系统

[![gitcgr](https://gitcgr.com/badge/boguangjun/Rag_Mcp.svg)](https://gitcgr.com/boguangjun/Rag_Mcp)

基于 ChromaDB 的 RAG（检索增强生成）知识库系统，提供 MCP 接口供大模型调用。

## 功能特性

- **知识库管理**：创建、删除、列出知识库
- **智能存储**：自动使用 LLM 生成摘要和关键词，提升搜索准确率
- **两步搜索**：先推荐相关知识库，再精确搜索内容
- **元知识库**：自动维护知识库索引，支持快速定位
- **实时笔记**：任务级别的笔记管理，支持冲突检测
- **GUI 界面**：Tkinter 图形界面，支持 Excel 批量导入

## 技术栈

| 组件 | 技术 |
|-----|------|
| 向量数据库 | ChromDB |
| 嵌入模型 | Qwen3-Embedding-0.6B |
| LLM 摘要 | DeepSeek API |
| MCP 协议 | mcp 1.26.0 |
| GUI | Tkinter |

## 项目结构

```
RAG_Mcp/
├── rag_backend.py       # 后端服务（HTTP 端口 19527）
├── mcp_shim.py          # MCP 垫片（轻量级，秒启动）
├── rag_manager.py       # RAG 核心逻辑
├── note_manager.py      # 笔记管理
├── ll_summarizer.py     # LLM 摘要生成
├── embedding.py         # 嵌入模型封装
├── rag_gui.py           # GUI 界面
├── run_backend.bat      # 启动后端服务
├── run_gui.bat          # 启动 GUI
├── chroma_db/           # 知识库数据
└── qwen3_embedding_model/  # 嵌入模型
```

## 快速开始

### 1. 安装依赖

```bash
python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 启动后端服务

双击 `run_backend.bat` 或：

```bash
call venv\Scripts\activate
python rag_backend.py
```

等待模型加载完成，显示 `[RAG后端] 服务已启动，等待请求...`

### 3. 配置 MCP（Trae IDE）

在 Trae 的 MCP 配置中添加：

```json
{
  "mcpServers": {
    "rag-knowledge-base": {
      "command": "(你的虚拟环境所在位置)\\venv\\Scripts\\python.exe",
      "args": ["(你的程序所在位置)\\RAG_Mcp\\mcp_shim.py"],
      "cwd": "(你的程序所在位置)"
    }
  }
}
```

### 4. 启动 GUI（可选）

双击 `run_gui.bat` 或：

```bash
call venv\Scripts\activate
python rag_gui.py
```

## MCP 接口

### 知识库管理

| 接口 | 描述 |
|-----|------|
| `list_knowledge_bases` | 列出所有知识库 |
| `create_knowledge_base` | 创建知识库 |
| `delete_knowledge_base` | 删除知识库 |
| `get_knowledge_base_info` | 获取知识库详情 |

### 知识操作

| 接口 | 描述 |
|-----|------|
| `add_knowledge` | 添加知识（自动生成摘要） |
| `add_knowledge_batch` | 批量添加知识 |
| `update_knowledge` | 更新知识 |
| `delete_knowledge` | 删除知识 |
| `get_knowledge` | 获取指定知识 |

### 搜索

| 接口 | 描述 |
|-----|------|
| `recommend_knowledge_base` | 推荐相关知识库 |
| `search_knowledge` | 在指定知识库搜索 |
| `global_search` | 全局搜索所有知识库 |

### 实时笔记

| 接口 | 描述 |
|-----|------|
| `ensure_note_kb` | 确保笔记库存在 |
| `read_notes` | 读取相关笔记 |
| `write_note` | 写入笔记 |
| `write_note_with_conflict_check` | 写入笔记并处理冲突 |
| `list_notes` | 列出所有笔记 |

## 使用流程

### 两步搜索

```
用户提问 → recommend_knowledge_base → 获取相关知识库 → search_knowledge → 返回结果
```

1. **第一步**：调用 `recommend_knowledge_base` 搜索元知识库，找到最相关的知识库
2. **第二步**：调用 `search_knowledge` 在推荐的知识库中精确搜索

### 实时笔记

```
开始任务 → ensure_note_kb → 对话前 read_notes → 对话 → 对话后 write_note_with_conflict_check
```

## 架构说明

```
┌─────────────┐     stdio      ┌─────────────┐
│   Trae IDE  │ ────────────── │  MCP 垫片   │
│             │                │ (mcp_shim)  │
└─────────────┘                └──────┬──────┘
                                      │ HTTP
                                      ▼
                               ┌─────────────┐
                               │  后端服务   │
                               │(rag_backend)│
                               │  预加载模型  │
                               └─────────────┘
```

- **后端服务**：预加载嵌入模型，常驻内存，HTTP 服务端口 19527
- **MCP 垫片**：轻量级，秒启动，通过 HTTP 调用后端

## 配置

### LLM 摘要服务

编辑 `ll_summarizer.py` 配置 DeepSeek API：

```python
API_URL = "https://api.deepseek.com"
API_KEY = "your-api-key"
MODEL = "deepseek-chat"
```

### 嵌入模型

模型会自动下载到 `qwen3_embedding_model/` 目录，支持 CUDA 加速。

## 许可证

MIT License
