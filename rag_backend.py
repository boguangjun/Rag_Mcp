import http.server
import json
import threading
from rag_manager import RAGManager, DEFAULT_KB_NAME
from note_manager import NoteManager

HOST = "127.0.0.1"
PORT = 19527

class RAGBackend:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.rag_manager = RAGManager()
                    cls._instance.note_manager = NoteManager(cls._instance.rag_manager)
        return cls._instance
    
    def handle(self, action: str, params: dict) -> dict:
        rag = self.rag_manager
        note = self.note_manager
        
        if action == "list_knowledge_bases":
            return rag.list_knowledge_bases()
        
        elif action == "create_knowledge_base":
            return rag.create_knowledge_base(
                name=params["name"],
                description=params.get("description")
            )
        
        elif action == "delete_knowledge_base":
            return rag.delete_knowledge_base(name=params["name"])
        
        elif action == "get_knowledge_base_info":
            return rag.get_knowledge_base_info(kb_name=params["kb_name"])
        
        elif action == "recommend_knowledge_base":
            return rag.recommend_knowledge_base(
                query=params["query"],
                n_results=params.get("n_results", 10)
            )
        
        elif action == "add_knowledge":
            return rag.add_knowledge(
                kb_name=params["kb_name"],
                index=params["index"],
                content=params["content"],
                metadata=params.get("metadata"),
                auto_summarize=params.get("auto_summarize", True)
            )
        
        elif action == "add_knowledge_batch":
            return rag.add_knowledge_batch(
                kb_name=params["kb_name"],
                items=params["items"],
                auto_summarize=params.get("auto_summarize", True)
            )
        
        elif action == "add_knowledge_batch_raw":
            return rag.add_knowledge_batch_raw(
                kb_name=params["kb_name"],
                items=params["items"]
            )
        
        elif action == "add_knowledge_raw":
            return rag.add_knowledge_raw(
                kb_name=params["kb_name"],
                index=params["index"],
                content=params["content"],
                summary=params["summary"],
                keywords=params["keywords"],
                metadata=params.get("metadata")
            )
        
        elif action == "update_knowledge":
            return rag.update_knowledge(
                kb_name=params["kb_name"],
                index=params["index"],
                content=params["content"],
                metadata=params.get("metadata"),
                auto_summarize=params.get("auto_summarize", True)
            )
        
        elif action == "search_knowledge":
            return rag.search(
                kb_name=params["kb_name"],
                query=params["query"],
                n_results=params.get("n_results", 5)
            )
        
        elif action == "global_search":
            return rag.global_search(
                query=params["query"],
                n_results=params.get("n_results", 5)
            )
        
        elif action == "delete_knowledge":
            return rag.delete_knowledge(
                kb_name=params["kb_name"],
                indices=params["indices"]
            )
        
        elif action == "get_knowledge":
            return rag.get_knowledge(
                kb_name=params["kb_name"],
                index=params["index"]
            )
        
        elif action == "ping":
            return {"success": True, "message": "pong"}
        
        elif action == "ensure_note_kb":
            return note.ensure_note_kb(
                task_name=params["task_name"],
                description=params.get("description", "")
            )
        
        elif action == "read_notes":
            return note.read_notes(
                task_name=params["task_name"],
                query=params["query"],
                n_results=params.get("n_results", 5)
            )
        
        elif action == "write_note":
            return note.write_note(
                task_name=params["task_name"],
                note_id=params["note_id"],
                content=params["content"],
                auto_summarize=params.get("auto_summarize", True)
            )
        
        elif action == "write_note_with_conflict_check":
            return note.write_note_with_conflict_check(
                task_name=params["task_name"],
                note_id=params["note_id"],
                content=params["content"],
                conflict_threshold=params.get("conflict_threshold", 0.3),
                auto_summarize=params.get("auto_summarize", True)
            )
        
        elif action == "find_note_conflicts":
            return note.find_conflicts(
                task_name=params["task_name"],
                new_content=params["new_content"],
                threshold=params.get("threshold", 0.3)
            )
        
        elif action == "delete_note":
            return note.delete_note(
                task_name=params["task_name"],
                note_id=params["note_id"]
            )
        
        elif action == "list_notes":
            return note.list_notes(task_name=params["task_name"])
        
        else:
            return {"success": False, "message": f"未知操作: {action}"}


class RequestHandler(http.server.BaseHTTPRequestHandler):
    backend = RAGBackend()
    
    def log_message(self, format, *args):
        pass
    
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8')
        
        try:
            data = json.loads(body)
            action = data.get("action")
            params = data.get("params", {})
            
            result = self.backend.handle(action, params)
            
            response = json.dumps(result, ensure_ascii=False)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(response.encode('utf-8'))
            
        except Exception as e:
            error_response = json.dumps({"success": False, "message": str(e)}, ensure_ascii=False)
            self.send_response(500)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(error_response.encode('utf-8'))
    
    def do_GET(self):
        if self.path == "/ping":
            result = self.backend.handle("ping", {})
            response = json.dumps(result, ensure_ascii=False)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(response.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()


def main():
    print(f"[RAG后端] 正在启动...")
    print(f"[RAG后端] 地址: http://{HOST}:{PORT}")
    
    server = http.server.HTTPServer((HOST, PORT), RequestHandler)
    
    print(f"[RAG后端] 服务已启动，等待请求...")
    server.serve_forever()


if __name__ == "__main__":
    main()
