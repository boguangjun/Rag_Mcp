import chromadb
from chromadb.config import Settings
from typing import Optional
import os
import json
import requests
from embedding import get_embedding_function, check_cuda

DEFAULT_KB_NAME = "meta_knowledge_base"


class LLSummarizer:
    def __init__(self, api_base: str = "https://api.deepseek.com", 
                 model: str = "deepseek-reasoner",
                 api_key: str = "sk-4cbb40a3f4b443479f6cc5c0653938f1"):
        self.api_base = api_base
        self.model = model
        self.api_key = api_key
    
    def generate_summary_and_keywords(self, content: str, context: str = "godot引擎文档") -> dict:
        prompt = f"""下面是{context}的内容，请分析内容，生成简洁的摘要和关键词。

内容：
{content}

请严格按照以下JSON格式返回结果，不要包含任何其他文字：
{{
    "summary": "用一句话概括这段知识的核心内容（50字以内）",
    "keywords": ["关键词1", "关键词2", "关键词3", "关键词4", "关键词5"]
}}

要求：
1. 摘要要抓住核心要点，便于快速理解内容
2. 关键词要覆盖主要概念、实体、技术术语等
3. 关键词数量3-8个
4. 只返回JSON，不要有其他内容"""

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3
            }
            
            response = requests.post(
                f"{self.api_base}/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                content_text = result["choices"][0]["message"]["content"]
                
                json_start = content_text.find("{")
                json_end = content_text.rfind("}") + 1
                if json_start != -1 and json_end > json_start:
                    json_str = content_text[json_start:json_end]
                    parsed = json.loads(json_str)
                    return {
                        "success": True,
                        "summary": parsed.get("summary", ""),
                        "keywords": parsed.get("keywords", [])
                    }
                
                return {"success": False, "message": "无法解析大模型返回结果"}
            else:
                return {"success": False, "message": f"API调用失败: {response.status_code}"}
        
        except json.JSONDecodeError as e:
            return {"success": False, "message": f"JSON解析失败: {str(e)}"}
        except Exception as e:
            return {"success": False, "message": f"生成摘要失败: {str(e)}"}


class RAGManager:
    _embedding_function = None
    
    def __init__(self, persist_directory: str = "./chroma_db",
                 api_base: str = "https://api.deepseek.com",
                 model: str = "deepseek-reasoner",
                 api_key: str = "sk-4cbb40a3f4b443479f6cc5c0653938f1",
                 model_dir: str = None):
        self.persist_directory = persist_directory
        self.summarizer = LLSummarizer(api_base, model, api_key)
        
        if RAGManager._embedding_function is None:
            print("\n" + "=" * 50)
            print("初始化Embedding模型")
            print("=" * 50)
            device = check_cuda()
            RAGManager._embedding_function = get_embedding_function(model_dir=model_dir, device=device)
            print("=" * 50 + "\n")
        
        self.embedding_function = RAGManager._embedding_function
        
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collections: dict = {}
        
        self._ensure_meta_knowledge_base()
    
    def _ensure_meta_knowledge_base(self):
        existing_collections = [c.name for c in self.client.list_collections()]
        if DEFAULT_KB_NAME not in existing_collections:
            self.client.create_collection(
                name=DEFAULT_KB_NAME,
                metadata={"description": "元知识库：存储所有知识库的索引信息，用于快速定位相关知识库", "is_meta": True}
            )
            print(f"[元知识库] 已自动创建默认知识库: {DEFAULT_KB_NAME}")
            self._sync_meta_knowledge_base()
    
    def _sync_meta_knowledge_base(self):
        try:
            meta_collection = self._get_collection(DEFAULT_KB_NAME)
            if meta_collection is None:
                return
            
            all_kbs = self.list_knowledge_bases()
            if not all_kbs.get("success"):
                return
            
            kb_list = all_kbs["knowledge_bases"]
            filtered_kbs = [kb for kb in kb_list if kb["name"] != DEFAULT_KB_NAME]
            
            existing = meta_collection.get()
            existing_ids = set(existing["ids"]) if existing["ids"] else set()
            current_ids = set(kb["name"] for kb in filtered_kbs)
            
            to_delete = existing_ids - current_ids
            if to_delete:
                meta_collection.delete(ids=list(to_delete))
            
            for kb in filtered_kbs:
                kb_name = kb["name"]
                description = kb.get("metadata", {}).get("description", "")
                doc_count = kb["count"]
                
                searchable_text = f"知识库名称: {kb_name}\n描述: {description}\n文档数量: {doc_count}"
                
                meta_metadata = {
                    "original_content": searchable_text,
                    "summary": f"知识库'{kb_name}': {description}" if description else f"知识库'{kb_name}'",
                    "keywords": json.dumps([kb_name], ensure_ascii=False),
                    "type": "knowledge_base_index",
                    "kb_name": kb_name,
                    "kb_description": description,
                    "kb_count": str(doc_count)
                }
                
                embeddings = self.embedding_function.embed_documents([searchable_text])
                
                try:
                    meta_collection.delete(ids=[kb_name])
                except:
                    pass
                
                meta_collection.add(
                    documents=[searchable_text],
                    ids=[kb_name],
                    embeddings=embeddings,
                    metadatas=[meta_metadata]
                )
            
            print(f"[元知识库] 已同步知识库索引，共 {len(filtered_kbs)} 个知识库")
            
        except Exception as e:
            print(f"[元知识库] 同步失败: {str(e)}")
    
    def _add_kb_to_meta(self, name: str, description: str = ""):
        try:
            meta_collection = self._get_collection(DEFAULT_KB_NAME)
            if meta_collection is None:
                return
            
            searchable_text = f"知识库名称: {name}\n描述: {description}\n文档数量: 0"
            
            meta_metadata = {
                "original_content": searchable_text,
                "summary": f"知识库'{name}': {description}" if description else f"知识库'{name}'",
                "keywords": json.dumps([name], ensure_ascii=False),
                "type": "knowledge_base_index",
                "kb_name": name,
                "kb_description": description,
                "kb_count": "0"
            }
            
            embeddings = self.embedding_function.embed_documents([searchable_text])
            meta_collection.add(
                documents=[searchable_text],
                ids=[name],
                embeddings=embeddings,
                metadatas=[meta_metadata]
            )
            
            print(f"[元知识库] 已添加知识库索引: {name}")
        except Exception as e:
            print(f"[元知识库] 添加索引失败: {str(e)}")
    
    def _remove_kb_from_meta(self, name: str):
        try:
            meta_collection = self._get_collection(DEFAULT_KB_NAME)
            if meta_collection is None:
                return
            
            meta_collection.delete(ids=[name])
            print(f"[元知识库] 已删除知识库索引: {name}")
        except Exception as e:
            print(f"[元知识库] 删除索引失败: {str(e)}")
    
    def create_knowledge_base(self, name: str, description: Optional[str] = None) -> dict:
        if name == DEFAULT_KB_NAME:
            return {"success": False, "message": f"不能使用保留名称 '{DEFAULT_KB_NAME}'"}
        
        if name in [c.name for c in self.client.list_collections()]:
            return {"success": False, "message": f"知识库 '{name}' 已存在"}
        
        try:
            collection = self.client.create_collection(
                name=name,
                metadata={"description": description or ""} if description else None
            )
            self.collections[name] = collection
            
            self._add_kb_to_meta(name, description or "")
            
            return {"success": True, "message": f"知识库 '{name}' 创建成功"}
        except Exception as e:
            return {"success": False, "message": f"创建知识库失败: {str(e)}"}
    
    def delete_knowledge_base(self, name: str) -> dict:
        if name == DEFAULT_KB_NAME:
            return {"success": False, "message": f"不能删除元知识库 '{DEFAULT_KB_NAME}'"}
        
        if name not in [c.name for c in self.client.list_collections()]:
            return {"success": False, "message": f"知识库 '{name}' 不存在"}
        
        try:
            self.client.delete_collection(name=name)
            if name in self.collections:
                del self.collections[name]
            
            self._remove_kb_from_meta(name)
            
            return {"success": True, "message": f"知识库 '{name}' 删除成功"}
        except Exception as e:
            return {"success": False, "message": f"删除知识库失败: {str(e)}"}
    
    def list_knowledge_bases(self) -> dict:
        try:
            collections = self.client.list_collections()
            result = []
            for col in collections:
                info = {
                    "name": col.name,
                    "count": col.count(),
                    "metadata": col.metadata or {},
                    "is_meta": col.name == DEFAULT_KB_NAME
                }
                result.append(info)
            return {"success": True, "knowledge_bases": result}
        except Exception as e:
            return {"success": False, "message": f"获取知识库列表失败: {str(e)}"}
    
    def _get_collection(self, name: str):
        if name in self.collections:
            return self.collections[name]
        
        try:
            collection = self.client.get_collection(name=name)
            self.collections[name] = collection
            return collection
        except Exception:
            return None
    
    def add_knowledge(self, kb_name: str, index: str, content: str, 
                      metadata: Optional[dict] = None,
                      auto_summarize: bool = True) -> dict:
        if kb_name == DEFAULT_KB_NAME:
            return {"success": False, "message": f"不能直接向元知识库添加内容"}
        
        collection = self._get_collection(kb_name)
        if collection is None:
            return {"success": False, "message": f"知识库 '{kb_name}' 不存在"}
        
        summary = ""
        keywords = []
        
        if auto_summarize:
            summary_result = self.summarizer.generate_summary_and_keywords(content)
            if summary_result["success"]:
                summary = summary_result["summary"]
                keywords = summary_result["keywords"]
            else:
                print(f"警告: 摘要生成失败 - {summary_result.get('message', '未知错误')}")
        
        searchable_text = self._build_searchable_text(content, summary, keywords)
        
        full_metadata = {
            "original_content": content,
            "summary": summary,
            "keywords": json.dumps(keywords, ensure_ascii=False) if keywords else "[]",
            **(metadata or {})
        }
        
        try:
            embeddings = self.embedding_function.embed_documents([searchable_text])
            collection.add(
                documents=[searchable_text],
                ids=[index],
                embeddings=embeddings,
                metadatas=[full_metadata]
            )
            
            return {
                "success": True,
                "message": f"知识添加成功",
                "index": index,
                "content": content,
                "summary": summary,
                "keywords": keywords
            }
        except Exception as e:
            return {"success": False, "message": f"添加知识失败: {str(e)}"}
    
    def add_knowledge_batch(self, kb_name: str, items: list, 
                            auto_summarize: bool = True) -> dict:
        if kb_name == DEFAULT_KB_NAME:
            return {"success": False, "message": f"不能直接向元知识库添加内容"}
        
        collection = self._get_collection(kb_name)
        if collection is None:
            return {"success": False, "message": f"知识库 '{kb_name}' 不存在"}
        
        try:
            indices = []
            searchable_texts = []
            metadatas = []
            
            for item in items:
                index = item["index"]
                content = item["content"]
                user_metadata = item.get("metadata", {})
                
                summary = ""
                keywords = []
                
                if auto_summarize:
                    summary_result = self.summarizer.generate_summary_and_keywords(content)
                    if summary_result["success"]:
                        summary = summary_result["summary"]
                        keywords = summary_result["keywords"]
                
                searchable_text = self._build_searchable_text(content, summary, keywords)
                
                full_metadata = {
                    "original_content": content,
                    "summary": summary,
                    "keywords": json.dumps(keywords, ensure_ascii=False) if keywords else "[]",
                    **user_metadata
                }
                
                indices.append(index)
                searchable_texts.append(searchable_text)
                metadatas.append(full_metadata)
            
            embeddings = self.embedding_function.embed_documents(searchable_texts)
            collection.add(
                documents=searchable_texts,
                ids=indices,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            return {
                "success": True,
                "message": f"批量添加 {len(items)} 条知识成功",
                "count": len(items)
            }
        except Exception as e:
            return {"success": False, "message": f"批量添加知识失败: {str(e)}"}
    
    def add_knowledge_raw(self, kb_name: str, index: str, content: str,
                          summary: str, keywords: list,
                          metadata: Optional[dict] = None) -> dict:
        if kb_name == DEFAULT_KB_NAME:
            return {"success": False, "message": f"不能直接向元知识库添加内容"}
        
        collection = self._get_collection(kb_name)
        if collection is None:
            return {"success": False, "message": f"知识库 '{kb_name}' 不存在"}
        
        searchable_text = self._build_searchable_text(content, summary, keywords)
        
        full_metadata = {
            "original_content": content,
            "summary": summary,
            "keywords": json.dumps(keywords, ensure_ascii=False) if keywords else "[]",
            **(metadata or {})
        }
        
        try:
            embeddings = self.embedding_function.embed_documents([searchable_text])
            collection.add(
                documents=[searchable_text],
                ids=[index],
                embeddings=embeddings,
                metadatas=[full_metadata]
            )
            
            return {
                "success": True,
                "message": f"知识添加成功（手动摘要）",
                "index": index,
                "content": content,
                "summary": summary,
                "keywords": keywords
            }
        except Exception as e:
            return {"success": False, "message": f"添加知识失败: {str(e)}"}
    
    def add_knowledge_batch_raw(self, kb_name: str, items: list) -> dict:
        if kb_name == DEFAULT_KB_NAME:
            return {"success": False, "message": f"不能直接向元知识库添加内容"}
        
        collection = self._get_collection(kb_name)
        if collection is None:
            return {"success": False, "message": f"知识库 '{kb_name}' 不存在"}
        
        try:
            indices = []
            searchable_texts = []
            metadatas = []
            seen_indices = set()
            
            for item in items:
                index = item["index"]
                content = item["content"]
                summary = item.get("summary", "")
                keywords = item.get("keywords", [])
                user_metadata = item.get("metadata", {})
                
                original_index = index
                counter = 1
                while index in seen_indices:
                    index = f"{original_index}_{counter}"
                    counter += 1
                seen_indices.add(index)
                
                searchable_text = self._build_searchable_text(content, summary, keywords)
                
                full_metadata = {
                    "original_content": content,
                    "summary": summary,
                    "keywords": json.dumps(keywords, ensure_ascii=False) if keywords else "[]",
                    **user_metadata
                }
                
                indices.append(index)
                searchable_texts.append(searchable_text)
                metadatas.append(full_metadata)
            
            embeddings = self.embedding_function.embed_documents(searchable_texts)
            collection.add(
                documents=searchable_texts,
                ids=indices,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            return {
                "success": True,
                "message": f"批量添加 {len(items)} 条知识成功",
                "count": len(items)
            }
        except Exception as e:
            return {"success": False, "message": f"批量添加知识失败: {str(e)}"}
    
    def _build_searchable_text(self, content: str, summary: str, keywords: list) -> str:
        parts = []
        
        if summary:
            parts.append(f"【摘要】{summary}")
        
        if keywords:
            parts.append(f"【关键词】{' '.join(keywords)}")
        
        parts.append(f"【内容】{content}")
        
        return "\n".join(parts)
    
    def search(self, kb_name: str, query: str, n_results: int = 5) -> dict:
        collection = self._get_collection(kb_name)
        if collection is None:
            return {"success": False, "message": f"知识库 '{kb_name}' 不存在"}
        
        try:
            query_embedding = self.embedding_function.embed_query(query)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            search_results = []
            if results["ids"] and results["ids"][0]:
                for i, idx in enumerate(results["ids"][0]):
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    
                    original_content = metadata.get("original_content", "")
                    summary = metadata.get("summary", "")
                    keywords_str = metadata.get("keywords", "[]")
                    try:
                        keywords = json.loads(keywords_str) if isinstance(keywords_str, str) else keywords_str
                    except:
                        keywords = []
                    
                    result_item = {
                        "index": idx,
                        "content": original_content,
                        "summary": summary,
                        "keywords": keywords,
                        "distance": results["distances"][0][i] if results["distances"] else None,
                        "metadata": {k: v for k, v in metadata.items() 
                                    if k not in ["original_content", "summary", "keywords"]}
                    }
                    search_results.append(result_item)
            
            return {
                "success": True,
                "query": query,
                "results": search_results,
                "count": len(search_results)
            }
        except Exception as e:
            return {"success": False, "message": f"搜索失败: {str(e)}"}
    
    def delete_knowledge(self, kb_name: str, indices: list) -> dict:
        if kb_name == DEFAULT_KB_NAME:
            return {"success": False, "message": f"不能直接删除元知识库内容"}
        
        collection = self._get_collection(kb_name)
        if collection is None:
            return {"success": False, "message": f"知识库 '{kb_name}' 不存在"}
        
        try:
            collection.delete(ids=indices)
            
            return {
                "success": True,
                "message": f"删除 {len(indices)} 条知识成功",
                "deleted_indices": indices
            }
        except Exception as e:
            return {"success": False, "message": f"删除知识失败: {str(e)}"}
    
    def get_knowledge(self, kb_name: str, index: str) -> dict:
        collection = self._get_collection(kb_name)
        if collection is None:
            return {"success": False, "message": f"知识库 '{kb_name}' 不存在"}
        
        try:
            result = collection.get(ids=[index], include=["documents", "metadatas"])
            if result["ids"]:
                metadata = result["metadatas"][0] if result["metadatas"] else {}
                keywords_str = metadata.get("keywords", "[]")
                try:
                    keywords = json.loads(keywords_str) if isinstance(keywords_str, str) else keywords_str
                except:
                    keywords = []
                return {
                    "success": True,
                    "index": result["ids"][0],
                    "content": metadata.get("original_content", ""),
                    "summary": metadata.get("summary", ""),
                    "keywords": keywords,
                    "metadata": {k: v for k, v in metadata.items() 
                                if k not in ["original_content", "summary", "keywords"]}
                }
            else:
                return {"success": False, "message": f"索引 '{index}' 不存在"}
        except Exception as e:
            return {"success": False, "message": f"获取知识失败: {str(e)}"}
    
    def update_knowledge(self, kb_name: str, index: str, content: str,
                         metadata: Optional[dict] = None,
                         auto_summarize: bool = True) -> dict:
        if kb_name == DEFAULT_KB_NAME:
            return {"success": False, "message": f"不能直接更新元知识库内容"}
        
        collection = self._get_collection(kb_name)
        if collection is None:
            return {"success": False, "message": f"知识库 '{kb_name}' 不存在"}
        
        try:
            existing = collection.get(ids=[index])
            if not existing["ids"]:
                return {"success": False, "message": f"索引 '{index}' 不存在"}
            
            collection.delete(ids=[index])
            
            result = self.add_knowledge(kb_name, index, content, metadata, auto_summarize)
            
            return result
        except Exception as e:
            return {"success": False, "message": f"更新知识失败: {str(e)}"}
    
    def recommend_knowledge_base(self, query: str, n_results: int = 10) -> dict:
        meta_collection = self._get_collection(DEFAULT_KB_NAME)
        if meta_collection is None:
            return {"success": False, "message": "元知识库不存在"}
        
        try:
            query_embedding = self.embedding_function.embed_query(query)
            results = meta_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            if results["ids"] and results["ids"][0]:
                recommended = []
                for i, kb_id in enumerate(results["ids"][0]):
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    recommended.append({
                        "name": kb_id,
                        "description": metadata.get("kb_description", ""),
                        "document_count": int(metadata.get("kb_count", 0)),
                        "distance": results["distances"][0][i] if results["distances"] else None
                    })
                
                return {
                    "success": True,
                    "query": query,
                    "recommended_knowledge_bases": recommended,
                    "message": f"找到 {len(recommended)} 个相关知识库，按相关度排序"
                }
            
            return {"success": False, "message": "未找到相关知识库"}
        except Exception as e:
            return {"success": False, "message": f"推荐知识库失败: {str(e)}"}
    
    def get_knowledge_base_info(self, kb_name: str) -> dict:
        if kb_name == DEFAULT_KB_NAME:
            return {"success": False, "message": "不能查询元知识库详细信息"}
        
        collection = self._get_collection(kb_name)
        if collection is None:
            return {"success": False, "message": f"知识库 '{kb_name}' 不存在"}
        
        try:
            all_items = collection.get(include=["metadatas"])
            
            total_count = len(all_items["ids"]) if all_items["ids"] else 0
            
            all_keywords = set()
            all_summaries = []
            
            if all_items["metadatas"]:
                for meta in all_items["metadatas"]:
                    keywords_str = meta.get("keywords", "[]")
                    try:
                        keywords = json.loads(keywords_str) if isinstance(keywords_str, str) else keywords_str
                        all_keywords.update(keywords)
                    except:
                        pass
                    
                    summary = meta.get("summary", "")
                    if summary:
                        all_summaries.append(summary)
            
            return {
                "success": True,
                "name": kb_name,
                "count": total_count,
                "metadata": collection.metadata or {},
                "all_keywords": list(all_keywords),
                "summaries": all_summaries[:10]
            }
        except Exception as e:
            return {"success": False, "message": f"获取知识库信息失败: {str(e)}"}
    
    def global_search(self, query: str, n_results: int = 5) -> dict:
        all_kbs = self.list_knowledge_bases()
        if not all_kbs.get("success"):
            return all_kbs
        
        all_results = []
        
        for kb in all_kbs["knowledge_bases"]:
            kb_name = kb["name"]
            if kb_name == DEFAULT_KB_NAME:
                continue
            
            search_result = self.search(kb_name, query, n_results=2)
            if search_result.get("success") and search_result.get("results"):
                for result in search_result["results"]:
                    result["knowledge_base"] = kb_name
                    all_results.append(result)
        
        all_results.sort(key=lambda x: x.get("distance", 1) or 1)
        
        return {
            "success": True,
            "query": query,
            "results": all_results[:n_results],
            "total_searched": len([kb for kb in all_kbs["knowledge_bases"] if kb["name"] != DEFAULT_KB_NAME])
        }
