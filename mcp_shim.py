import mcp.server.stdio
import mcp.types as types
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import json
import asyncio
import urllib.request
import urllib.error

BACKEND_URL = "http://127.0.0.1:19527"


def call_backend(action: str, params: dict = None) -> dict:
    data = json.dumps({"action": action, "params": params or {}}).encode('utf-8')
    
    req = urllib.request.Request(
        BACKEND_URL,
        data=data,
        headers={'Content-Type': 'application/json'}
    )
    
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.URLError as e:
        return {"success": False, "message": f"后端连接失败: {str(e)}"}


class RAGMCPServer:
    def __init__(self):
        self.server = Server("rag-mcp-server")
        self._setup_handlers()
    
    def _setup_handlers(self):
        @self.server.list_tools()
        async def list_tools() -> list[types.Tool]:
            return [
                types.Tool(
                    name="create_knowledge_base",
                    description="创建一个新的知识库",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "知识库名称"},
                            "description": {"type": "string", "description": "知识库描述（可选）"}
                        },
                        "required": ["name"]
                    }
                ),
                types.Tool(
                    name="delete_knowledge_base",
                    description="删除一个知识库",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "要删除的知识库名称"}
                        },
                        "required": ["name"]
                    }
                ),
                types.Tool(
                    name="list_knowledge_bases",
                    description="列出所有知识库（包括名称、文档数量、描述等信息）",
                    inputSchema={"type": "object", "properties": {}}
                ),
                types.Tool(
                    name="get_knowledge_base_info",
                    description="获取指定知识库的详细信息",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "kb_name": {"type": "string", "description": "知识库名称"}
                        },
                        "required": ["kb_name"]
                    }
                ),
                types.Tool(
                    name="recommend_knowledge_base",
                    description="根据查询内容推荐最相关的知识库。大模型应先调用此接口确定应该搜索哪个知识库。",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "用户的查询内容"},
                            "n_results": {"type": "integer", "description": "返回结果数量，默认10", "default": 10}
                        },
                        "required": ["query"]
                    }
                ),
                types.Tool(
                    name="add_knowledge",
                    description="向知识库中添加一条知识（自动生成摘要和关键词）",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "kb_name": {"type": "string", "description": "知识库名称"},
                            "index": {"type": "string", "description": "知识的唯一索引/ID"},
                            "content": {"type": "string", "description": "知识内容"},
                            "metadata": {"type": "object", "description": "可选的元数据"},
                            "auto_summarize": {"type": "boolean", "description": "是否自动生成摘要，默认true", "default": True}
                        },
                        "required": ["kb_name", "index", "content"]
                    }
                ),
                types.Tool(
                    name="add_knowledge_batch",
                    description="批量向知识库中添加知识",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "kb_name": {"type": "string", "description": "知识库名称"},
                            "items": {
                                "type": "array",
                                "description": "知识项列表",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "index": {"type": "string"},
                                        "content": {"type": "string"},
                                        "metadata": {"type": "object"}
                                    },
                                    "required": ["index", "content"]
                                }
                            },
                            "auto_summarize": {"type": "boolean", "default": True}
                        },
                        "required": ["kb_name", "items"]
                    }
                ),
                types.Tool(
                    name="add_knowledge_raw",
                    description="向知识库添加知识（手动提供摘要和关键词）",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "kb_name": {"type": "string", "description": "知识库名称"},
                            "index": {"type": "string", "description": "知识的唯一索引/ID"},
                            "content": {"type": "string", "description": "知识内容"},
                            "summary": {"type": "string", "description": "知识摘要"},
                            "keywords": {"type": "array", "items": {"type": "string"}, "description": "关键词列表"},
                            "metadata": {"type": "object", "description": "可选的元数据"}
                        },
                        "required": ["kb_name", "index", "content", "summary", "keywords"]
                    }
                ),
                types.Tool(
                    name="update_knowledge",
                    description="更新知识库中的知识",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "kb_name": {"type": "string", "description": "知识库名称"},
                            "index": {"type": "string", "description": "要更新的知识索引"},
                            "content": {"type": "string", "description": "新的知识内容"},
                            "metadata": {"type": "object", "description": "可选的元数据"},
                            "auto_summarize": {"type": "boolean", "default": True}
                        },
                        "required": ["kb_name", "index", "content"]
                    }
                ),
                types.Tool(
                    name="search_knowledge",
                    description="在指定知识库中搜索相关知识",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "kb_name": {"type": "string", "description": "知识库名称"},
                            "query": {"type": "string", "description": "搜索查询文本"},
                            "n_results": {"type": "integer", "description": "返回结果数量，默认5", "default": 5}
                        },
                        "required": ["kb_name", "query"]
                    }
                ),
                types.Tool(
                    name="global_search",
                    description="在所有知识库中全局搜索",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "搜索查询文本"},
                            "n_results": {"type": "integer", "description": "返回结果数量，默认5", "default": 5}
                        },
                        "required": ["query"]
                    }
                ),
                types.Tool(
                    name="delete_knowledge",
                    description="从知识库中删除知识",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "kb_name": {"type": "string", "description": "知识库名称"},
                            "indices": {"type": "array", "items": {"type": "string"}, "description": "要删除的知识索引列表"}
                        },
                        "required": ["kb_name", "indices"]
                    }
                ),
                types.Tool(
                    name="get_knowledge",
                    description="根据索引获取知识库中的特定知识",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "kb_name": {"type": "string", "description": "知识库名称"},
                            "index": {"type": "string", "description": "知识索引"}
                        },
                        "required": ["kb_name", "index"]
                    }
                ),
                types.Tool(
                    name="ensure_note_kb",
                    description="确保任务笔记知识库存在。在开始新任务时调用，如果笔记库不存在则自动创建。",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "task_name": {"type": "string", "description": "任务名称（用于标识笔记库）"},
                            "description": {"type": "string", "description": "任务描述（可选）"}
                        },
                        "required": ["task_name"]
                    }
                ),
                types.Tool(
                    name="read_notes",
                    description="读取任务笔记。在对话开始前调用，搜索与当前对话相关的历史笔记，返回内容附加到提示词中。",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "task_name": {"type": "string", "description": "任务名称"},
                            "query": {"type": "string", "description": "搜索查询（当前对话内容或关键词）"},
                            "n_results": {"type": "integer", "description": "返回结果数量，默认5", "default": 5}
                        },
                        "required": ["task_name", "query"]
                    }
                ),
                types.Tool(
                    name="write_note",
                    description="写入笔记。对话结束后调用，将归纳的要点写入笔记库。",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "task_name": {"type": "string", "description": "任务名称"},
                            "note_id": {"type": "string", "description": "笔记ID（用于标识和更新）"},
                            "content": {"type": "string", "description": "笔记内容"},
                            "auto_summarize": {"type": "boolean", "description": "是否自动生成摘要，默认true", "default": True}
                        },
                        "required": ["task_name", "note_id", "content"]
                    }
                ),
                types.Tool(
                    name="write_note_with_conflict_check",
                    description="写入笔记并检查冲突。对话结束后调用，自动检测与新内容冲突的旧笔记并删除，然后写入新笔记。",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "task_name": {"type": "string", "description": "任务名称"},
                            "note_id": {"type": "string", "description": "笔记ID"},
                            "content": {"type": "string", "description": "笔记内容"},
                            "conflict_threshold": {"type": "number", "description": "冲突检测阈值（0-1，越小越严格），默认0.3", "default": 0.3},
                            "auto_summarize": {"type": "boolean", "default": True}
                        },
                        "required": ["task_name", "note_id", "content"]
                    }
                ),
                types.Tool(
                    name="find_note_conflicts",
                    description="查找与新内容冲突的笔记。用于检测笔记中是否存在与新信息矛盾的内容。",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "task_name": {"type": "string", "description": "任务名称"},
                            "new_content": {"type": "string", "description": "新内容"},
                            "threshold": {"type": "number", "description": "冲突检测阈值，默认0.3", "default": 0.3}
                        },
                        "required": ["task_name", "new_content"]
                    }
                ),
                types.Tool(
                    name="delete_note",
                    description="删除指定笔记",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "task_name": {"type": "string", "description": "任务名称"},
                            "note_id": {"type": "string", "description": "要删除的笔记ID"}
                        },
                        "required": ["task_name", "note_id"]
                    }
                ),
                types.Tool(
                    name="list_notes",
                    description="列出任务的所有笔记",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "task_name": {"type": "string", "description": "任务名称"}
                        },
                        "required": ["task_name"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
            result = call_backend(name, arguments)
            return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]
    
    async def run(self):
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="rag-mcp-server",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(tools_changed=False),
                        experimental_capabilities=None
                    )
                )
            )


def main():
    result = call_backend("ping")
    if not result.get("success"):
        print(f"[MCP垫片] 错误: 无法连接后端服务 {BACKEND_URL}")
        print(f"[MCP垫片] 请先运行 run_backend.bat 启动后端服务")
        return
    
    server = RAGMCPServer()
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
