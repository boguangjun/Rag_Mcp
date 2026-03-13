import os
import sys
from typing import List, Optional
import numpy as np


def check_cuda():
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            device_count = torch.cuda.device_count()
            print(f"[Embedding] CUDA可用! 检测到 {device_count} 个GPU: {device_name}")
            return "cuda"
        else:
            print("[Embedding] CUDA不可用，使用CPU运行")
            return "cpu"
    except ImportError:
        print("[Embedding] PyTorch未安装，使用CPU运行")
        return "cpu"


class QwenEmbeddingFunction:
    MODELSCOPE_MODEL = "Qwen/Qwen3-Embedding-0.6B"
    MODEL_DIR_NAME = "qwen3-embedding-0.6b"
    
    def __init__(self, model_dir: str = None, device: str = None):
        self.model_dir = model_dir or os.path.join(os.path.dirname(__file__), "models", self.MODEL_DIR_NAME)
        self.device = device or check_cuda()
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        print(f"[Embedding] 正在加载Qwen3-Embedding模型...")
        print(f"[Embedding] 模型目录: {self.model_dir}")
        print(f"[Embedding] 运行设备: {self.device}")
        
        try:
            if self._is_valid_model_dir(self.model_dir):
                print(f"[Embedding] 从本地目录加载模型...")
                self._load_from_local()
            else:
                print(f"[Embedding] 首次运行，正在从ModelScope下载模型...")
                self._download_from_modelscope()
                self._load_from_local()
            
            print(f"[Embedding] 模型加载成功!")
            
        except Exception as e:
            print(f"[Embedding] 模型加载失败: {e}")
            raise
    
    def _is_valid_model_dir(self, path):
        if not os.path.exists(path):
            return False
        
        has_config = os.path.exists(os.path.join(path, "config.json"))
        has_model = os.path.exists(os.path.join(path, "model.safetensors")) or \
                    os.path.exists(os.path.join(path, "pytorch_model.bin"))
        return has_config and has_model
    
    def _download_from_modelscope(self):
        try:
            from modelscope import snapshot_download
            
            print(f"[Embedding] 正在从ModelScope下载模型...")
            os.makedirs(self.model_dir, exist_ok=True)
            
            model_path = snapshot_download(
                self.MODELSCOPE_MODEL,
                cache_dir=self.model_dir,
                revision="master"
            )
            
            print(f"[Embedding] 模型下载完成: {model_path}")
            
            if model_path != self.model_dir:
                import shutil
                if os.path.exists(self.model_dir):
                    for item in os.listdir(model_path):
                        src = os.path.join(model_path, item)
                        dst = os.path.join(self.model_dir, item)
                        if os.path.isdir(src):
                            shutil.copytree(src, dst, dirs_exist_ok=True)
                        else:
                            shutil.copy2(src, dst)
            
        except ImportError:
            print(f"[Embedding] modelscope未安装，尝试安装...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "modelscope", "-q"])
            self._download_from_modelscope()
    
    def _load_from_local(self):
        from transformers import AutoModel, AutoTokenizer
        import torch
        
        print(f"[Embedding] 加载tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir, 
            trust_remote_code=True,
            use_fast=False
        )
        
        print(f"[Embedding] 加载模型...")
        self.model = AutoModel.from_pretrained(
            self.model_dir, 
            trust_remote_code=True
        )
        
        if self.device == "cuda":
            self.model = self.model.cuda()
        
        self.model.eval()
        print(f"[Embedding] 模型加载完成!")
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.embed_documents(input)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        import torch
        
        embeddings = []
        
        for text in texts:
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            )
            
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                last_hidden_state = outputs.last_hidden_state
                attention_mask = inputs["attention_mask"]
                
                mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                embedding = (sum_embeddings / sum_mask)
                
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
                embeddings.append(embedding[0].cpu().numpy().tolist())
        
        return embeddings
    
    def embed_query(self, query: str) -> List[float]:
        return self.embed_documents([query])[0]


def get_embedding_function(model_dir: str = None, device: str = None) -> QwenEmbeddingFunction:
    return QwenEmbeddingFunction(model_dir=model_dir, device=device)


if __name__ == "__main__":
    print("=" * 50)
    print("测试Qwen3-Embedding模型")
    print("=" * 50)
    
    emb = QwenEmbeddingFunction()
    
    test_texts = [
        "Python是一种高级编程语言",
        "机器学习是人工智能的一个分支",
        "深度学习使用神经网络进行学习",
        "今天天气真好",
        "Python编程语言由Guido创建"
    ]
    
    print("\n测试文本:")
    for i, text in enumerate(test_texts):
        print(f"  {i+1}. {text}")
    
    print("\n生成embedding...")
    embeddings = emb.embed_documents(test_texts)
    
    print(f"\nEmbedding维度: {len(embeddings[0])}")
    print(f"Embedding数量: {len(embeddings)}")
    
    print("\n计算相似度:")
    from numpy import dot
    from numpy.linalg import norm
    
    query = "Python是什么？"
    query_emb = emb.embed_query(query)
    print(f"查询: {query}")
    
    similarities = []
    for i, text in enumerate(test_texts):
        sim = dot(query_emb, embeddings[i]) / (norm(query_emb) * norm(embeddings[i]))
        similarities.append((i, sim, text))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    for i, sim, text in similarities:
        print(f"  [{sim:.4f}] {text}")
    
    print("\n测试完成!")
