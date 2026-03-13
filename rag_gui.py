import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import json
import os
import pandas as pd
from openpyxl import load_workbook
from concurrent.futures import ThreadPoolExecutor, as_completed
from rag_manager import RAGManager, LLSummarizer


class RAGGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("RAG知识库管理系统")
        self.root.geometry("1200x800")
        
        self.rag_manager = RAGManager()
        self.summarizer = LLSummarizer()
        
        self.excel_data = None
        self.excel_file_path = None
        
        self._create_ui()
        self._refresh_kb_list()
    
    def _create_ui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self._create_excel_tab()
        self._create_manage_tab()
        self._create_search_tab()
    
    def _create_excel_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Excel批量导入")
        
        toolbar = ttk.Frame(frame)
        toolbar.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(toolbar, text="导入Excel", command=self._import_excel).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="保存Excel", command=self._save_excel).pack(side=tk.LEFT, padx=2)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        ttk.Label(toolbar, text="并发数:").pack(side=tk.LEFT, padx=2)
        self.concurrent_var = tk.StringVar(value="10")
        concurrent_spin = ttk.Spinbox(toolbar, from_=1, to=50, width=5, textvariable=self.concurrent_var)
        concurrent_spin.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(toolbar, text="批量生成摘要", command=self._batch_summarize).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="停止", command=self._stop_summarize).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        ttk.Label(toolbar, text="目标知识库:").pack(side=tk.LEFT, padx=2)
        self.excel_kb_var = tk.StringVar()
        self.excel_kb_combo = ttk.Combobox(toolbar, textvariable=self.excel_kb_var, width=15)
        self.excel_kb_combo.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(toolbar, text="一键导入知识库", command=self._import_to_kb).pack(side=tk.LEFT, padx=2)
        
        self.progress_frame = ttk.Frame(frame)
        self.progress_frame.pack(fill=tk.X, padx=5, pady=2)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, side=tk.LEFT, expand=True)
        
        self.progress_label = ttk.Label(self.progress_frame, text="就绪")
        self.progress_label.pack(side=tk.LEFT, padx=5)
        
        columns = ("index", "content", "summary", "keywords", "status")
        self.excel_tree = ttk.Treeview(frame, columns=columns, show="headings", height=20)
        
        self.excel_tree.heading("index", text="索引")
        self.excel_tree.heading("content", text="内容")
        self.excel_tree.heading("summary", text="摘要")
        self.excel_tree.heading("keywords", text="关键词")
        self.excel_tree.heading("status", text="状态")
        
        self.excel_tree.column("index", width=100)
        self.excel_tree.column("content", width=300)
        self.excel_tree.column("summary", width=250)
        self.excel_tree.column("keywords", width=200)
        self.excel_tree.column("status", width=80)
        
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.excel_tree.yview)
        self.excel_tree.configure(yscrollcommand=scrollbar.set)
        
        self.excel_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(btn_frame, text="重新生成选中行摘要", command=self._regenerate_selected).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="编辑选中行", command=self._edit_selected).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="删除选中行", command=self._delete_selected).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="添加新行", command=self._add_row).pack(side=tk.LEFT, padx=2)
        
        self._stop_flag = False
    
    def _create_manage_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="知识库管理")
        
        left_frame = ttk.LabelFrame(frame, text="知识库操作")
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        ttk.Label(left_frame, text="知识库名称:").pack(padx=5, pady=2)
        self.kb_name_var = tk.StringVar()
        ttk.Entry(left_frame, textvariable=self.kb_name_var, width=20).pack(padx=5, pady=2)
        
        ttk.Label(left_frame, text="描述:").pack(padx=5, pady=2)
        self.kb_desc_var = tk.StringVar()
        ttk.Entry(left_frame, textvariable=self.kb_desc_var, width=20).pack(padx=5, pady=2)
        
        ttk.Button(left_frame, text="创建知识库", command=self._create_kb).pack(padx=5, pady=5, fill=tk.X)
        ttk.Button(left_frame, text="删除知识库", command=self._delete_kb).pack(padx=5, pady=5, fill=tk.X)
        
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        ttk.Label(left_frame, text="选择知识库:").pack(padx=5, pady=2)
        self.manage_kb_var = tk.StringVar()
        self.manage_kb_combo = ttk.Combobox(left_frame, textvariable=self.manage_kb_var, width=18)
        self.manage_kb_combo.pack(padx=5, pady=2)
        self.manage_kb_combo.bind("<<ComboboxSelected>>", lambda e: self._load_kb_content())
        
        ttk.Button(left_frame, text="刷新列表", command=self._load_kb_content).pack(padx=5, pady=5, fill=tk.X)
        
        right_frame = ttk.Frame(frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        toolbar = ttk.Frame(right_frame)
        toolbar.pack(fill=tk.X, pady=5)
        
        ttk.Button(toolbar, text="添加知识", command=self._add_knowledge).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="编辑选中", command=self._edit_knowledge).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="删除选中", command=self._delete_knowledge).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="刷新", command=self._load_kb_content).pack(side=tk.LEFT, padx=2)
        
        columns = ("index", "content", "summary", "keywords")
        self.manage_tree = ttk.Treeview(right_frame, columns=columns, show="headings", height=20)
        
        self.manage_tree.heading("index", text="索引")
        self.manage_tree.heading("content", text="内容")
        self.manage_tree.heading("summary", text="摘要")
        self.manage_tree.heading("keywords", text="关键词")
        
        self.manage_tree.column("index", width=120)
        self.manage_tree.column("content", width=350)
        self.manage_tree.column("summary", width=250)
        self.manage_tree.column("keywords", width=200)
        
        scrollbar = ttk.Scrollbar(right_frame, orient=tk.VERTICAL, command=self.manage_tree.yview)
        self.manage_tree.configure(yscrollcommand=scrollbar.set)
        
        self.manage_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _create_search_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="搜索测试")
        
        top_frame = ttk.Frame(frame)
        top_frame.pack(fill=tk.X, padx=5, pady=10)
        
        ttk.Label(top_frame, text="选择知识库:").pack(side=tk.LEFT, padx=5)
        self.search_kb_var = tk.StringVar()
        self.search_kb_combo = ttk.Combobox(top_frame, textvariable=self.search_kb_var, width=15)
        self.search_kb_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(top_frame, text="匹配模式:").pack(side=tk.LEFT, padx=5)
        self.match_mode_var = tk.StringVar(value="综合匹配")
        match_modes = ["关键词匹配", "摘要匹配", "综合匹配"]
        ttk.Combobox(top_frame, textvariable=self.match_mode_var, values=match_modes, width=12, state="readonly").pack(side=tk.LEFT, padx=5)
        
        ttk.Label(top_frame, text="返回数量:").pack(side=tk.LEFT, padx=5)
        self.result_count_var = tk.StringVar(value="10")
        ttk.Spinbox(top_frame, from_=1, to=100, width=5, textvariable=self.result_count_var).pack(side=tk.LEFT, padx=5)
        
        search_frame = ttk.Frame(frame)
        search_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(search_frame, text="搜索内容:").pack(side=tk.LEFT, padx=5)
        self.search_entry = ttk.Entry(search_frame, width=60)
        self.search_entry.pack(side=tk.LEFT, padx=5)
        self.search_entry.bind("<Return>", lambda e: self._search())
        
        ttk.Button(search_frame, text="搜索", command=self._search).pack(side=tk.LEFT, padx=10)
        ttk.Button(search_frame, text="清空结果", command=self._clear_search).pack(side=tk.LEFT, padx=5)
        
        result_frame = ttk.Frame(frame)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        columns = ("rank", "index", "content", "summary", "keywords", "distance")
        self.search_tree = ttk.Treeview(result_frame, columns=columns, show="headings", height=20)
        
        self.search_tree.heading("rank", text="排名")
        self.search_tree.heading("index", text="索引")
        self.search_tree.heading("content", text="内容")
        self.search_tree.heading("summary", text="摘要")
        self.search_tree.heading("keywords", text="关键词")
        self.search_tree.heading("distance", text="距离")
        
        self.search_tree.column("rank", width=50)
        self.search_tree.column("index", width=100)
        self.search_tree.column("content", width=300)
        self.search_tree.column("summary", width=200)
        self.search_tree.column("keywords", width=150)
        self.search_tree.column("distance", width=80)
        
        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.search_tree.yview)
        self.search_tree.configure(yscrollcommand=scrollbar.set)
        
        self.search_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _refresh_kb_list(self):
        result = self.rag_manager.list_knowledge_bases()
        kb_names = [kb["name"] for kb in result.get("knowledge_bases", [])]
        
        self.excel_kb_combo["values"] = kb_names
        self.manage_kb_combo["values"] = kb_names
        self.search_kb_combo["values"] = kb_names
        
        if kb_names:
            if not self.excel_kb_var.get():
                self.excel_kb_combo.current(0)
            if not self.manage_kb_var.get():
                self.manage_kb_combo.current(0)
            if not self.search_kb_var.get():
                self.search_kb_combo.current(0)
    
    def _import_excel(self):
        file_path = filedialog.askopenfilename(
            title="选择Excel文件",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )
        if not file_path:
            return
        
        try:
            self.excel_file_path = file_path
            df = pd.read_excel(file_path)
            
            required_cols = ["index", "content"]
            for col in required_cols:
                if col not in df.columns:
                    messagebox.showerror("错误", f"Excel缺少必需列: {col}\n需要: index, content, summary(可选), keywords(可选)")
                    return
            
            if "summary" not in df.columns:
                df["summary"] = ""
            if "keywords" not in df.columns:
                df["keywords"] = ""
            if "status" not in df.columns:
                df["status"] = "待处理"
            
            self.excel_data = df
            
            self._refresh_excel_tree()
            messagebox.showinfo("成功", f"成功导入 {len(df)} 条数据")
            
        except Exception as e:
            messagebox.showerror("错误", f"导入失败: {str(e)}")
    
    def _refresh_excel_tree(self):
        for item in self.excel_tree.get_children():
            self.excel_tree.delete(item)
        
        if self.excel_data is not None:
            for _, row in self.excel_data.iterrows():
                keywords = row.get("keywords", "")
                if isinstance(keywords, list):
                    keywords = ", ".join(keywords)
                
                self.excel_tree.insert("", tk.END, values=(
                    row.get("index", ""),
                    row.get("content", "")[:100] + "..." if len(str(row.get("content", ""))) > 100 else row.get("content", ""),
                    row.get("summary", "")[:50] + "..." if len(str(row.get("summary", ""))) > 50 else row.get("summary", ""),
                    keywords[:50] + "..." if len(str(keywords)) > 50 else keywords,
                    row.get("status", "待处理")
                ))
    
    def _save_excel(self):
        if self.excel_data is None:
            messagebox.showwarning("警告", "没有数据可保存")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="保存Excel文件",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")]
        )
        if not file_path:
            return
        
        try:
            self.excel_data.to_excel(file_path, index=False)
            self.excel_file_path = file_path
            messagebox.showinfo("成功", f"文件已保存: {file_path}")
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {str(e)}")
    
    def _batch_summarize(self):
        if self.excel_data is None:
            messagebox.showwarning("警告", "请先导入Excel文件")
            return
        
        pending = self.excel_data[self.excel_data["status"] != "已完成"]
        if len(pending) == 0:
            messagebox.showinfo("提示", "所有数据已处理完成")
            return
        
        self._stop_flag = False
        concurrent = int(self.concurrent_var.get())
        total = len(pending)
        completed = 0
        
        self.progress_var.set(0)
        self.progress_label.config(text=f"处理中 0/{total}")
        
        def process_row(idx):
            if self._stop_flag:
                return idx, None, None, "已停止"
            
            content = self.excel_data.loc[idx, "content"]
            result = self.summarizer.generate_summary_and_keywords(str(content))
            
            if result["success"]:
                return idx, result["summary"], result["keywords"], "已完成"
            else:
                return idx, None, None, f"失败: {result.get('message', '未知错误')}"
        
        def run_batch():
            nonlocal completed
            
            with ThreadPoolExecutor(max_workers=concurrent) as executor:
                futures = {}
                for idx in pending.index:
                    future = executor.submit(process_row, idx)
                    futures[future] = idx
                
                for future in as_completed(futures):
                    if self._stop_flag:
                        break
                    
                    idx, summary, keywords, status = future.result()
                    
                    self.excel_data.loc[idx, "summary"] = summary or ""
                    self.excel_data.loc[idx, "keywords"] = ", ".join(keywords) if keywords else ""
                    self.excel_data.loc[idx, "status"] = status
                    
                    completed += 1
                    progress = (completed / total) * 100
                    
                    self.root.after(0, lambda p=progress, c=completed, t=total: [
                        self.progress_var.set(p),
                        self.progress_label.config(text=f"处理中 {c}/{t}"),
                        self._refresh_excel_tree()
                    ])
            
            self.root.after(0, lambda: [
                self.progress_label.config(text=f"完成 {completed}/{total}"),
                messagebox.showinfo("完成", f"批量处理完成\n成功: {completed}")
            ])
        
        threading.Thread(target=run_batch, daemon=True).start()
    
    def _stop_summarize(self):
        self._stop_flag = True
        self.progress_label.config(text="正在停止...")
    
    def _regenerate_selected(self):
        selected = self.excel_tree.selection()
        if not selected:
            messagebox.showwarning("警告", "请先选择要重新生成的行")
            return
        
        item = selected[0]
        values = self.excel_tree.item(item, "values")
        
        idx = None
        for i, row in self.excel_data.iterrows():
            if str(row["index"]) == values[0]:
                idx = i
                break
        
        if idx is None:
            return
        
        content = self.excel_data.loc[idx, "content"]
        result = self.summarizer.generate_summary_and_keywords(str(content))
        
        if result["success"]:
            self.excel_data.loc[idx, "summary"] = result["summary"]
            self.excel_data.loc[idx, "keywords"] = ", ".join(result["keywords"])
            self.excel_data.loc[idx, "status"] = "已完成"
            self._refresh_excel_tree()
            messagebox.showinfo("成功", "摘要已重新生成")
        else:
            messagebox.showerror("错误", f"生成失败: {result.get('message', '未知错误')}")
    
    def _edit_selected(self):
        selected = self.excel_tree.selection()
        if not selected:
            messagebox.showwarning("警告", "请先选择要编辑的行")
            return
        
        item = selected[0]
        values = self.excel_tree.item(item, "values")
        
        idx = None
        for i, row in self.excel_data.iterrows():
            if str(row["index"]) == values[0]:
                idx = i
                break
        
        if idx is None:
            return
        
        EditDialog(self.root, self.excel_data, idx, self._refresh_excel_tree)
    
    def _delete_selected(self):
        selected = self.excel_tree.selection()
        if not selected:
            messagebox.showwarning("警告", "请先选择要删除的行")
            return
        
        if not messagebox.askyesno("确认", "确定要删除选中的行吗？"):
            return
        
        for item in selected:
            values = self.excel_tree.item(item, "values")
            for i, row in self.excel_data.iterrows():
                if str(row["index"]) == values[0]:
                    self.excel_data.drop(i, inplace=True)
                    break
        
        self.excel_data.reset_index(drop=True, inplace=True)
        self._refresh_excel_tree()
    
    def _add_row(self):
        if self.excel_data is None:
            self.excel_data = pd.DataFrame(columns=["index", "content", "summary", "keywords", "status"])
        
        new_idx = len(self.excel_data)
        self.excel_data.loc[new_idx] = ["", "", "", "", "待处理"]
        self._refresh_excel_tree()
    
    def _import_to_kb(self):
        kb_name = self.excel_kb_var.get()
        if not kb_name:
            messagebox.showwarning("警告", "请选择目标知识库")
            return
        
        if self.excel_data is None:
            messagebox.showwarning("警告", "没有数据可导入")
            return
        
        ready = self.excel_data[self.excel_data["status"] == "已完成"]
        if len(ready) == 0:
            messagebox.showwarning("警告", "没有已完成处理的数据")
            return
        
        items = []
        for _, row in ready.iterrows():
            keywords_str = row.get("keywords", "")
            if isinstance(keywords_str, str):
                keywords = [k.strip() for k in keywords_str.split(",") if k.strip()]
            else:
                keywords = keywords_str if isinstance(keywords_str, list) else []
            
            items.append({
                "index": str(row["index"]),
                "content": str(row["content"]),
                "summary": str(row.get("summary", "")),
                "keywords": keywords
            })
        
        self.progress_var.set(0)
        self.progress_label.config(text=f"正在导入 {len(items)} 条数据...")
        
        def do_import():
            result = self.rag_manager.add_knowledge_batch_raw(kb_name, items)
            
            self.root.after(0, lambda: [
                self.progress_var.set(100),
                self.progress_label.config(text="导入完成"),
                messagebox.showinfo("导入完成", f"成功: {result.get('count', 0)}\n失败: 0" if result["success"] else f"失败: {result.get('message', '未知错误')}")
            ])
        
        threading.Thread(target=do_import, daemon=True).start()
    
    def _create_kb(self):
        name = self.kb_name_var.get().strip()
        if not name:
            messagebox.showwarning("警告", "请输入知识库名称")
            return
        
        result = self.rag_manager.create_knowledge_base(name, self.kb_desc_var.get())
        if result["success"]:
            messagebox.showinfo("成功", result["message"])
            self._refresh_kb_list()
        else:
            messagebox.showerror("错误", result["message"])
    
    def _delete_kb(self):
        name = self.kb_name_var.get().strip()
        if not name:
            messagebox.showwarning("警告", "请输入知识库名称")
            return
        
        if not messagebox.askyesno("确认", f"确定要删除知识库 '{name}' 吗？\n此操作不可恢复！"):
            return
        
        result = self.rag_manager.delete_knowledge_base(name)
        if result["success"]:
            messagebox.showinfo("成功", result["message"])
            self._refresh_kb_list()
        else:
            messagebox.showerror("错误", result["message"])
    
    def _load_kb_content(self):
        kb_name = self.manage_kb_var.get()
        if not kb_name:
            return
        
        for item in self.manage_tree.get_children():
            self.manage_tree.delete(item)
        
        result = self.rag_manager.search(kb_name, "", n_results=1000)
        
        if result["success"]:
            for item in result["results"]:
                keywords = item.get("keywords", [])
                keywords_str = ", ".join(keywords) if isinstance(keywords, list) else str(keywords)
                
                self.manage_tree.insert("", tk.END, values=(
                    item["index"],
                    item["content"][:100] + "..." if len(str(item["content"])) > 100 else item["content"],
                    item.get("summary", "")[:50] + "..." if len(str(item.get("summary", ""))) > 50 else item.get("summary", ""),
                    keywords_str[:50] + "..." if len(keywords_str) > 50 else keywords_str
                ))
    
    def _add_knowledge(self):
        kb_name = self.manage_kb_var.get()
        if not kb_name:
            messagebox.showwarning("警告", "请选择知识库")
            return
        
        KnowledgeEditDialog(self.root, self.rag_manager, kb_name, None, self._load_kb_content)
    
    def _edit_knowledge(self):
        kb_name = self.manage_kb_var.get()
        if not kb_name:
            messagebox.showwarning("警告", "请选择知识库")
            return
        
        selected = self.manage_tree.selection()
        if not selected:
            messagebox.showwarning("警告", "请选择要编辑的知识")
            return
        
        values = self.manage_tree.item(selected[0], "values")
        index = values[0]
        
        result = self.rag_manager.get_knowledge(kb_name, index)
        if result["success"]:
            KnowledgeEditDialog(self.root, self.rag_manager, kb_name, result, self._load_kb_content)
    
    def _delete_knowledge(self):
        kb_name = self.manage_kb_var.get()
        if not kb_name:
            messagebox.showwarning("警告", "请选择知识库")
            return
        
        selected = self.manage_tree.selection()
        if not selected:
            messagebox.showwarning("警告", "请选择要删除的知识")
            return
        
        if not messagebox.askyesno("确认", "确定要删除选中的知识吗？"):
            return
        
        indices = [self.manage_tree.item(item, "values")[0] for item in selected]
        result = self.rag_manager.delete_knowledge(kb_name, indices)
        
        if result["success"]:
            messagebox.showinfo("成功", result["message"])
            self._load_kb_content()
        else:
            messagebox.showerror("错误", result["message"])
    
    def _search(self):
        kb_name = self.search_kb_var.get()
        if not kb_name:
            messagebox.showwarning("警告", "请选择知识库")
            return
        
        query = self.search_entry.get().strip()
        if not query:
            messagebox.showwarning("警告", "请输入搜索内容")
            return
        
        n_results = int(self.result_count_var.get())
        match_mode = self.match_mode_var.get()
        
        search_query = query
        if match_mode == "关键词匹配":
            search_query = f"【关键词】{query}"
        elif match_mode == "摘要匹配":
            search_query = f"【摘要】{query}"
        
        result = self.rag_manager.search(kb_name, search_query, n_results)
        
        for item in self.search_tree.get_children():
            self.search_tree.delete(item)
        
        if result["success"]:
            for rank, item in enumerate(result["results"], 1):
                keywords = item.get("keywords", [])
                keywords_str = ", ".join(keywords) if isinstance(keywords, list) else str(keywords)
                
                distance = item.get("distance")
                distance_str = f"{distance:.4f}" if distance is not None else "N/A"
                
                self.search_tree.insert("", tk.END, values=(
                    rank,
                    item["index"],
                    item["content"][:100] + "..." if len(str(item["content"])) > 100 else item["content"],
                    item.get("summary", "")[:50] + "..." if len(str(item.get("summary", ""))) > 50 else item.get("summary", ""),
                    keywords_str[:50] + "..." if len(keywords_str) > 50 else keywords_str,
                    distance_str
                ))
        else:
            messagebox.showerror("错误", result["message"])
    
    def _clear_search(self):
        for item in self.search_tree.get_children():
            self.search_tree.delete(item)
        self.search_entry.delete(0, tk.END)


class EditDialog(tk.Toplevel):
    def __init__(self, parent, data, idx, callback):
        super().__init__(parent)
        self.title("编辑数据")
        self.geometry("600x400")
        self.data = data
        self.idx = idx
        self.callback = callback
        
        self.transient(parent)
        self.grab_set()
        
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(main_frame, text="索引:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.index_var = tk.StringVar(value=data.loc[idx, "index"])
        ttk.Entry(main_frame, textvariable=self.index_var, width=60).grid(row=0, column=1, sticky=tk.EW, pady=2)
        
        ttk.Label(main_frame, text="内容:").grid(row=1, column=0, sticky=tk.NW, pady=2)
        self.content_text = tk.Text(main_frame, width=60, height=8)
        self.content_text.grid(row=1, column=1, sticky=tk.EW, pady=2)
        self.content_text.insert("1.0", str(data.loc[idx, "content"]))
        
        ttk.Label(main_frame, text="摘要:").grid(row=2, column=0, sticky=tk.NW, pady=2)
        self.summary_text = tk.Text(main_frame, width=60, height=3)
        self.summary_text.grid(row=2, column=1, sticky=tk.EW, pady=2)
        self.summary_text.insert("1.0", str(data.loc[idx, "summary"]))
        
        ttk.Label(main_frame, text="关键词:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.keywords_var = tk.StringVar(value=data.loc[idx, "keywords"])
        ttk.Entry(main_frame, textvariable=self.keywords_var, width=60).grid(row=3, column=1, sticky=tk.EW, pady=2)
        
        ttk.Label(main_frame, text="(关键词用逗号分隔)").grid(row=4, column=1, sticky=tk.W)
        
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=5, column=0, columnspan=2, pady=10)
        
        ttk.Button(btn_frame, text="保存", command=self._save).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="取消", command=self.destroy).pack(side=tk.LEFT, padx=5)
        
        main_frame.columnconfigure(1, weight=1)
    
    def _save(self):
        self.data.loc[self.idx, "index"] = self.index_var.get()
        self.data.loc[self.idx, "content"] = self.content_text.get("1.0", tk.END).strip()
        self.data.loc[self.idx, "summary"] = self.summary_text.get("1.0", tk.END).strip()
        self.data.loc[self.idx, "keywords"] = self.keywords_var.get()
        self.data.loc[self.idx, "status"] = "已完成"
        
        self.callback()
        self.destroy()


class KnowledgeEditDialog(tk.Toplevel):
    def __init__(self, parent, rag_manager, kb_name, data, callback):
        super().__init__(parent)
        self.title("添加知识" if data is None else "编辑知识")
        self.geometry("600x450")
        self.rag_manager = rag_manager
        self.kb_name = kb_name
        self.data = data
        self.callback = callback
        
        self.transient(parent)
        self.grab_set()
        
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(main_frame, text="索引:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.index_var = tk.StringVar(value=data["index"] if data else "")
        index_entry = ttk.Entry(main_frame, textvariable=self.index_var, width=60)
        index_entry.grid(row=0, column=1, sticky=tk.EW, pady=2)
        if data:
            index_entry.config(state="readonly")
        
        ttk.Label(main_frame, text="内容:").grid(row=1, column=0, sticky=tk.NW, pady=2)
        self.content_text = tk.Text(main_frame, width=60, height=10)
        self.content_text.grid(row=1, column=1, sticky=tk.EW, pady=2)
        if data:
            self.content_text.insert("1.0", str(data.get("content", "")))
        
        ttk.Label(main_frame, text="摘要:").grid(row=2, column=0, sticky=tk.NW, pady=2)
        self.summary_text = tk.Text(main_frame, width=60, height=3)
        self.summary_text.grid(row=2, column=1, sticky=tk.EW, pady=2)
        if data:
            self.summary_text.insert("1.0", str(data.get("summary", "")))
        
        ttk.Label(main_frame, text="关键词:").grid(row=3, column=0, sticky=tk.W, pady=2)
        keywords = data.get("keywords", []) if data else []
        keywords_str = ", ".join(keywords) if isinstance(keywords, list) else str(keywords)
        self.keywords_var = tk.StringVar(value=keywords_str)
        ttk.Entry(main_frame, textvariable=self.keywords_var, width=60).grid(row=3, column=1, sticky=tk.EW, pady=2)
        ttk.Label(main_frame, text="(关键词用逗号分隔)").grid(row=4, column=1, sticky=tk.W)
        
        self.auto_summarize_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(main_frame, text="自动生成摘要和关键词", variable=self.auto_summarize_var).grid(row=5, column=1, sticky=tk.W, pady=5)
        
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=6, column=0, columnspan=2, pady=10)
        
        ttk.Button(btn_frame, text="保存", command=self._save).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="取消", command=self.destroy).pack(side=tk.LEFT, padx=5)
        
        main_frame.columnconfigure(1, weight=1)
    
    def _save(self):
        index = self.index_var.get().strip()
        content = self.content_text.get("1.0", tk.END).strip()
        summary = self.summary_text.get("1.0", tk.END).strip()
        keywords_str = self.keywords_var.get().strip()
        keywords = [k.strip() for k in keywords_str.split(",") if k.strip()]
        
        if not index or not content:
            messagebox.showwarning("警告", "索引和内容不能为空")
            return
        
        if self.data is None:
            result = self.rag_manager.add_knowledge_raw(
                self.kb_name, index, content, summary, keywords
            )
        else:
            self.rag_manager.delete_knowledge(self.kb_name, [index])
            result = self.rag_manager.add_knowledge_raw(
                self.kb_name, index, content, summary, keywords
            )
        
        if result["success"]:
            messagebox.showinfo("成功", result["message"])
            self.callback()
            self.destroy()
        else:
            messagebox.showerror("错误", result["message"])


def main():
    root = tk.Tk()
    app = RAGGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
