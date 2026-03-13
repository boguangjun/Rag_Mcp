import json
from rag_manager import RAGManager, DEFAULT_KB_NAME

NOTE_KB_PREFIX = "note_"

class NoteManager:
    def __init__(self, rag_manager: RAGManager = None):
        if rag_manager:
            self.rag = rag_manager
        else:
            from rag_manager import RAGManager
            self.rag = RAGManager()
    
    def _get_note_kb_name(self, task_name: str) -> str:
        safe_name = task_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        return f"{NOTE_KB_PREFIX}{safe_name}"
    
    def ensure_note_kb(self, task_name: str, description: str = "") -> dict:
        kb_name = self._get_note_kb_name(task_name)
        
        all_kbs = self.rag.list_knowledge_bases()
        if all_kbs.get("success"):
            for kb in all_kbs["knowledge_bases"]:
                if kb["name"] == kb_name:
                    return {
                        "success": True,
                        "kb_name": kb_name,
                        "message": f"笔记知识库已存在: {kb_name}",
                        "is_new": False
                    }
        
        result = self.rag.create_knowledge_base(
            name=kb_name,
            description=f"任务'{task_name}'的实时笔记 - {description}"
        )
        
        if result["success"]:
            return {
                "success": True,
                "kb_name": kb_name,
                "message": f"笔记知识库创建成功: {kb_name}",
                "is_new": True
            }
        else:
            return result
    
    def read_notes(self, task_name: str, query: str, n_results: int = 5) -> dict:
        kb_name = self._get_note_kb_name(task_name)
        
        kb_info = self.rag.get_knowledge_base_info(kb_name)
        if not kb_info.get("success"):
            return {
                "success": True,
                "notes": [],
                "message": "笔记知识库不存在，这是新任务"
            }
        
        if kb_info.get("total_count", 0) == 0:
            return {
                "success": True,
                "notes": [],
                "message": "笔记为空"
            }
        
        result = self.rag.search(kb_name, query, n_results=n_results)
        
        if result.get("success"):
            notes = []
            for item in result.get("results", []):
                notes.append({
                    "index": item.get("index"),
                    "content": item.get("content"),
                    "summary": item.get("summary"),
                    "distance": item.get("distance")
                })
            
            return {
                "success": True,
                "notes": notes,
                "message": f"找到 {len(notes)} 条相关笔记"
            }
        else:
            return result
    
    def write_note(self, task_name: str, note_id: str, content: str, 
                   auto_summarize: bool = True) -> dict:
        kb_name = self._get_note_kb_name(task_name)
        
        ensure_result = self.ensure_note_kb(task_name)
        if not ensure_result.get("success"):
            return ensure_result
        
        existing = self.rag.get_knowledge(kb_name, note_id)
        if existing.get("success"):
            return self.rag.update_knowledge(kb_name, note_id, content, auto_summarize=auto_summarize)
        else:
            return self.rag.add_knowledge(kb_name, note_id, content, auto_summarize=auto_summarize)
    
    def find_conflicts(self, task_name: str, new_content: str, threshold: float = 0.3) -> dict:
        kb_name = self._get_note_kb_name(task_name)
        
        kb_info = self.rag.get_knowledge_base_info(kb_name)
        if not kb_info.get("success") or kb_info.get("total_count", 0) == 0:
            return {
                "success": True,
                "conflicts": [],
                "message": "没有现有笔记，无冲突"
            }
        
        result = self.rag.search(kb_name, new_content, n_results=10)
        
        conflicts = []
        if result.get("success"):
            for item in result.get("results", []):
                distance = item.get("distance", 1)
                if distance < threshold:
                    conflicts.append({
                        "index": item.get("index"),
                        "content": item.get("content"),
                        "distance": distance,
                        "summary": item.get("summary")
                    })
        
        return {
            "success": True,
            "conflicts": conflicts,
            "message": f"找到 {len(conflicts)} 条可能冲突的笔记"
        }
    
    def delete_note(self, task_name: str, note_id: str) -> dict:
        kb_name = self._get_note_kb_name(task_name)
        return self.rag.delete_knowledge(kb_name, [note_id])
    
    def list_notes(self, task_name: str) -> dict:
        kb_name = self._get_note_kb_name(task_name)
        
        kb_info = self.rag.get_knowledge_base_info(kb_name)
        if not kb_info.get("success"):
            return {
                "success": True,
                "notes": [],
                "message": "笔记知识库不存在"
            }
        
        collection = self.rag._get_collection(kb_name)
        if collection is None:
            return {
                "success": True,
                "notes": [],
                "message": "笔记知识库不存在"
            }
        
        try:
            all_items = collection.get(include=["metadatas"])
            
            notes = []
            if all_items["ids"]:
                for i, note_id in enumerate(all_items["ids"]):
                    meta = all_items["metadatas"][i] if all_items["metadatas"] else {}
                    notes.append({
                        "index": note_id,
                        "summary": meta.get("summary", ""),
                        "keywords": meta.get("keywords", "")
                    })
            
            return {
                "success": True,
                "notes": notes,
                "total_count": len(notes),
                "message": f"共 {len(notes)} 条笔记"
            }
        except Exception as e:
            return {
                "success": False,
                "notes": [],
                "message": f"获取笔记列表失败: {str(e)}"
            }
    
    def write_note_with_conflict_check(self, task_name: str, note_id: str, 
                                        content: str, conflict_threshold: float = 0.3,
                                        auto_summarize: bool = True) -> dict:
        conflicts = self.find_conflicts(task_name, content, conflict_threshold)
        
        conflict_indices = []
        if conflicts.get("success") and conflicts.get("conflicts"):
            for c in conflicts["conflicts"]:
                if c["index"] != note_id:
                    conflict_indices.append(c["index"])
        
        if conflict_indices:
            kb_name = self._get_note_kb_name(task_name)
            self.rag.delete_knowledge(kb_name, conflict_indices)
        
        return self.write_note(task_name, note_id, content, auto_summarize)
