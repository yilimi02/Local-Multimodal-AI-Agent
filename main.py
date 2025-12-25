import argparse
import os
import readline
import shutil
import warnings
from typing import List, Optional

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 忽略一些库的警告信息，保持输出整洁
warnings.filterwarnings("ignore")

import chromadb
from chromadb.utils import embedding_functions
import pdfplumber
from PIL import Image
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm

# ==========================================
# 配置与常量
# ==========================================
DB_PATH = "./chroma_db"
LIBRARY_PATH = "./my_papers"  # 论文自动归档的根目录
TEXT_MODEL_NAME = "all-MiniLM-L6-v2"  # 轻量级文本模型
CLIP_MODEL_NAME = "clip-ViT-B-32"  # CLIP 图像模型


class LocalAIAgent:
    def __init__(self):
        print("正在初始化 AI Agent (加载模型与数据库)...")

        # 1. 初始化 ChromaDB (持久化存储)
        self.client = chromadb.PersistentClient(path=DB_PATH)

        # 2. 初始化嵌入模型 (使用 SentenceTransformers)
        # 文本模型
        self.text_model = SentenceTransformer(TEXT_MODEL_NAME)
        # 图像模型 (CLIP)
        self.clip_model = SentenceTransformer(CLIP_MODEL_NAME)

        # 3. 获取或创建集合 (Collections)
        # 注意：ChromaDB 原生支持 embedding function，但为了灵活性我们手动计算 embedding
        self.paper_collection = self.client.get_or_create_collection(name="papers")
        self.image_collection = self.client.get_or_create_collection(name="images")

        # 确保归档目录存在
        if not os.path.exists(LIBRARY_PATH):
            os.makedirs(LIBRARY_PATH)

    # ==========================================
    # 核心功能：论文管理
    # ==========================================
    def _extract_text_from_pdf(self, pdf_path: str, max_chars: int = 2000) -> str:
        """读取PDF前几页的文本用于索引和分类"""
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages[:5]:  # 通常摘要和引言在前2页
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
            return text[:max_chars]  # 截取前N个字符，避免Token溢出
        except Exception as e:
            print(f"读取 PDF 失败 {pdf_path}: {e}")
            return ""

    def _classify_text(self, text: str, topics: List[str]) -> str:
        """零样本分类：计算文本与Topic的相似度，返回最匹配的Topic"""
        if not topics or not text:
            return "Uncategorized"

        # 编码文本和Topics
        text_emb = self.text_model.encode(text, convert_to_tensor=True)
        topic_embs = self.text_model.encode(topics, convert_to_tensor=True)

        # 计算余弦相似度
        cos_scores = util.cos_sim(text_emb, topic_embs)[0]
        best_idx = torch.argmax(cos_scores).item()

        return topics[best_idx]

    def add_paper(self, file_path: str, topics: Optional[str] = None):
        """添加论文：提取文本 -> 存入向量库 -> (可选)自动分类移动"""
        if not os.path.exists(file_path):
            print(f"错误: 文件不存在 {file_path}")
            return

        print(f"正在处理: {os.path.basename(file_path)}...")

        # 1. 提取文本
        content = self._extract_text_from_pdf(file_path)
        if not content:
            return

        # 2. 生成嵌入并存入数据库
        embedding = self.text_model.encode(content).tolist()
        file_name = os.path.basename(file_path)

        self.paper_collection.upsert(
            ids=[file_name],
            documents=[content],
            metadatas=[{"filename": file_name, "path": file_path}],
            embeddings=[embedding]
        )
        print(f"✅ 已索引: {file_name}")

        # 3. 自动分类与移动 (如果指定了 topics)
        if topics:
            topic_list = [t.strip() for t in topics.split(',')]
            category = self._classify_text(content, topic_list)

            # 创建分类文件夹
            target_dir = os.path.join(LIBRARY_PATH, category)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            # 移动文件
            target_path = os.path.join(target_dir, file_name)
            shutil.move(file_path, target_path)

            # 更新数据库中的路径信息
            self.paper_collection.update(
                ids=[file_name],
                metadatas=[{"filename": file_name, "path": target_path, "category": category}]
            )
            print(f"📂 已归档到: {category}/")

    def search_paper(self, query: str, n_results: int = 3):
        """语义搜索论文"""
        print(f"🔍 正在搜索文献: '{query}'...")
        query_emb = self.text_model.encode(query).tolist()

        results = self.paper_collection.query(
            query_embeddings=[query_emb],
            n_results=n_results
        )

        print("\n--- 搜索结果 ---")
        if not results['ids'][0]:
            print("未找到相关文档。")
            return

        for i, doc_id in enumerate(results['ids'][0]):
            meta = results['metadatas'][0][i]
            dist = results['distances'][0][i]
            print(f"[{i + 1}] 文件名: {doc_id}")
            print(f"    路径: {meta.get('path', 'Unknown')}")
            print(f"    分类: {meta.get('category', 'N/A')}")
            print(f"    相关性距离: {dist:.4f}")
            print("-" * 30)

    def organize_folder(self, source_folder: str, topics: str):
        """批量整理文件夹"""
        if not os.path.exists(source_folder):
            print("文件夹不存在")
            return

        files = [f for f in os.listdir(source_folder) if f.lower().endswith('.pdf')]
        print(f"发现 {len(files)} 个 PDF 文件，准备整理...")

        for f in tqdm(files):
            full_path = os.path.join(source_folder, f)
            self.add_paper(full_path, topics)

    # ==========================================
    # 核心功能：图像管理 (以文搜图)
    # ==========================================
    def add_image(self, image_path: str):
        """添加图片索引"""
        try:
            img = Image.open(image_path)
            # CLIP 编码图片
            embedding = self.clip_model.encode(img).tolist()
            file_name = os.path.basename(image_path)

            self.image_collection.upsert(
                ids=[file_name],
                embeddings=[embedding],
                metadatas=[{"path": image_path}]
            )
            print(f"✅ 图片已索引: {file_name}")
        except Exception as e:
            print(f"处理图片失败 {image_path}: {e}")

    def search_image(self, query: str, n_results: int = 3):
        """以文搜图"""
        print(f"🖼️ 正在搜索图片: '{query}'...")
        # CLIP 编码文本 (因为 CLIP 是多模态对齐的，文本和图片在同一空间)
        query_emb = self.clip_model.encode(query).tolist()

        results = self.image_collection.query(
            query_embeddings=[query_emb],
            n_results=n_results
        )

        print("\n--- 图片搜索结果 ---")
        if not results['ids'][0]:
            print("未找到相关图片。")
            return

        for i, doc_id in enumerate(results['ids'][0]):
            meta = results['metadatas'][0][i]
            print(f"[{i + 1}] 图片: {doc_id}")
            print(f"    路径: {meta.get('path')}")
            print("-" * 30)

    def batch_add_images(self, folder_path: str):
        """批量添加图片"""
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in valid_exts]

        print(f"正在索引 {len(files)} 张图片...")
        for f in tqdm(files):
            self.add_image(os.path.join(folder_path, f))


# ==========================================
# 命令行接口 (CLIP)
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Local Multimodal AI Agent")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: add_paper
    parser_add = subparsers.add_parser("add_paper", help="Add and classify a paper")
    parser_add.add_argument("path", type=str, help="Path to PDF file")
    parser_add.add_argument("--topics", type=str, help="Comma separated topics (e.g., 'CV,NLP,RL')")

    # Command: search_paper
    parser_search = subparsers.add_parser("search_paper", help="Semantic search for papers")
    parser_search.add_argument("query", type=str, help="Search query")

    # Command: organize
    parser_org = subparsers.add_parser("organize", help="Batch organize a folder")
    parser_org.add_argument("folder", type=str, help="Folder containing PDFs")
    parser_org.add_argument("--topics", type=str, required=True, help="Topics for classification")

    # Command: add_image_folder
    parser_img_add = subparsers.add_parser("index_images", help="Index a folder of images")
    parser_img_add.add_argument("folder", type=str, help="Folder path")

    # Command: search_image
    parser_img_search = subparsers.add_parser("search_image", help="Text to Image search")
    parser_img_search.add_argument("query", type=str, help="Description of image")

    # 添加一个新的子命令 'interactive'
    parser_interactive = subparsers.add_parser("interactive", help="Run in interactive mode")

    args = parser.parse_args()

    # 如果是交互模式
    if args.command == "interactive":
        agent = LocalAIAgent()  # 只初始化一次
        print("\n=== 进入交互模式 (输入 'exit' 退出) ===")
        print("支持命令: search_paper <query> | search_image <query> | add_paper <path>")

        while True:
            user_input = input("\n(LocalAI) >>> ").strip()
            if user_input in ["exit", "quit"]:
                break

            # 简单的命令解析
            parts = user_input.split(" ", 1)
            cmd = parts[0]
            param = parts[1] if len(parts) > 1 else ""

            if cmd == "search_paper":
                agent.search_paper(param)
            elif cmd == "search_image":
                agent.search_image(param)
            elif cmd == "add_paper":
                agent.add_paper(param)
            else:
                print("未知命令，请重试。")

    # 原有的命令行逻辑
    elif args.command:
        agent = LocalAIAgent()

        if args.command == "add_paper":
            agent.add_paper(args.path, args.topics)
        elif args.command == "search_paper":
            agent.search_paper(args.query)
        elif args.command == "organize":
            agent.organize_folder(args.folder, args.topics)
        elif args.command == "index_images":
            agent.batch_add_images(args.folder)
        elif args.command == "search_image":
            agent.search_image(args.query)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()