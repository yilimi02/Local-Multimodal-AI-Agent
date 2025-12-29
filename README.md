# 本地多模态 AI 智能助手 (Local Multimodal AI Agent)

## 1. 项目简介 (Project Introduction)
项目GitHub地址：[yilimi02/Local-Multimodal-AI-Agent](https://github.com/yilimi02/Local-Multimodal-AI-Agent?tab=readme-ov-file#本地-ai-智能文献与图像管理助手-local-multimodal-ai-agent)

使用测试文件：https://pan.baidu.com/s/1qVs_6oS3cIksUo1yGPiMNA?pwd=m4qj

本项目是一个基于 Python 的本地化多模态 AI 助手，旨在帮助用户高效管理本地的**学术文献 (PDF)** 和 **图像素材**。

不同于传统的文件名搜索，本项目利用 **SentenceTransformers** 和 **CLIP** 模型，将文本和图像转化为向量存储在本地的 **ChromaDB** 数据库中，实现了：

- **语义搜索**：用自然语言（如“Transformer 的核心架构”）搜索相关论文。
- **以文搜图**：用文字描述（如“海边的日落”）搜索本地图片。
- **自动分类**：根据论文内容自动将其归档到对应主题的文件夹。

> **特点**：所有数据和模型均在本地运行，保护隐私，无需联网上传数据（模型下载除外）。代码内置了国内镜像源加速，解决了 HuggingFace 连接困难的问题。

## 2. 核心功能要求 (Core Features)

### 2.1 智能文献管理
*   **语义搜索**: 支持使用自然语言提问（如“Transformer 的核心架构是什么？”）。系统基于语义理解返回最相关的论文文件。
*   **自动分类与整理**:
    *   **单文件处理**: 添加新论文时，根据指定的主题（如 "CV, NLP, RL"）自动分析内容，将其归类并移动到对应的子文件夹中。
    *   **批量整理**: 支持对现有的混乱文件夹进行“一键整理”，自动扫描所有 PDF，识别主题并归档到相应目录。
*   **文件索引**: 支持仅返回相关文件列表，方便快速定位所需文献。

### 2.2 智能图像管理
*   **以文搜图**: 利用多模态图文匹配技术，支持通过自然语言描述（如“海边的日落”）来查找本地图片库中最匹配的图像。

## 3. 技术选型与模型 (Technical Stack)

本项目采用了轻量级且高效的模块化设计：

- **文本嵌入模型**: `all-MiniLM-L6-v2`
  - 速度快，内存占用小，适合普通笔记本运行。
- **图像嵌入模型**: `clip-ViT-B-32` (OpenAI CLIP)
  - 经典的图文对齐模型，能够理解图像与文本之间的语义关联。
- **向量数据库**: `ChromaDB`
  - 嵌入式数据库，无需安装服务器，数据保存在本地 `./chroma_db` 目录。
- **PDF 处理**: `pdfplumber`
  - 提取 PDF 前 5 页的文本（摘要和引言通常在此），兼顾速度与准确性。

## 4. 环境要求 (Environment)

### 4.1 基础要求

*   **操作系统**: Windows / macOS / Linux
*   **Python 版本**: 建议 Python 3.8 及以上
*   **内存**: 建议 8GB 及以上

### 4.2 安装依赖

项目依赖文件已整理，请在终端运行以下命令安装所需库（推荐使用国内源加速）：

```python
pip install requirements.txt
```

**依赖库说明：**

- `chromadb`: 向量数据库，用于存储和检索向量。
- `sentence-transformers`: 加载文本和图像嵌入模型。
- `pdfplumber`: 用于提取 PDF 中的文本。
- `Pillow`: 图像处理库。
- `torch`: 深度学习框架后端。

## 5. 使用说明

项目提供了两种使用方式： **单次命令模式** 和 **交互模式（推荐）** 。

确保你的终端位于项目根目录下（即 `main.py` 所在目录）。

### 5.1 方式一：单次命令行工具
#### 1. 添加并分类单篇论文

将 PDF 文件索引到数据库，并根据主题自动移动到 `my_papers/` 文件夹。

示例：

```
python main.py add_paper "CLIP.pdf" --topics "Trajectory Prediction, Domain Generalization, Diffusion Model, Pose Estimation, Large Language Model"
```

![image-20251229074609589](C:\Users\haperay\AppData\Roaming\Typora\typora-user-images\image-20251229074609589.png)

![image-20251229074740030](C:\Users\haperay\AppData\Roaming\Typora\typora-user-images\image-20251229074740030.png)

#### 2. 批量整理文件夹

扫描指定文件夹中的所有 PDF，自动分类并移动到 `my_papers/`。

示例：

```python
python main.py organize "./downloads" --topics "Trajectory Prediction, Domain Generalization, Diffusion Model, Pose Estimation, Large Language Model"
```

![image-20251229085425202](C:\Users\haperay\AppData\Roaming\Typora\typora-user-images\image-20251229085425202.png)

![image-20251229085449057](C:\Users\haperay\AppData\Roaming\Typora\typora-user-images\image-20251229085449057.png)

#### 3. 语义搜索论文

示例：

```python
python main.py search_paper "How does self-attention work?"
```

![image-20251229085624627](C:\Users\haperay\AppData\Roaming\Typora\typora-user-images\image-20251229085624627.png)

#### 4. 建立图片索引

扫描指定文件夹下的图片（.jpg, .png 等），建立向量索引。

示例：

```python
python main.py index_images "./my_photos"
```

![image-20251229100533334](C:\Users\haperay\AppData\Roaming\Typora\typora-user-images\image-20251229100533334.png)

#### 5. 以文搜图

示例：

```python
python main.py search_image "a white cat"
```

![image-20251229101048950](C:\Users\haperay\AppData\Roaming\Typora\typora-user-images\image-20251229101048950.png)

### 5.2 交互模式 (推荐)
#### 1. **启动交互模式**

```python
python main.py interactive
```

![image-20251229101357454](C:\Users\haperay\AppData\Roaming\Typora\typora-user-images\image-20251229101357454.png)

#### 2. 语义搜索论文

示例：

```python
search_paper "how to predict trajectory?"
```

![image-20251229101640529](C:\Users\haperay\AppData\Roaming\Typora\typora-user-images\image-20251229101640529.png)

#### 3. 以文搜图

示例：

```python
search_image "一只鸟"
```

![image-20251229102653254](C:\Users\haperay\AppData\Roaming\Typora\typora-user-images\image-20251229102653254.png)
