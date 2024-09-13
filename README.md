# 基于华为 MindSpore 的多模态时序大模型-新能源时空智能引擎

## 项目简介

本项目创新性地提出了新能源行业 AI 一站式解决方案，基于华为 Mindspore 深度学习框架和云服务构建，涵盖风/光功率预测、电力市场预测、智能知识沉淀等功能，并提供题库生成与智能批改的扩展应用。方案广泛适用于能源、环境、物联网等多个领域，具备高适配性和安全性，可通过微调快速迁移应用。

## 功能展示

### 视频演示

<video width="400" height="300" controls>
  <source src="./figures/video.mp4" type="video/mp4">
  你的浏览器不支持视频标签。
</video>

### 管理员界面

展示前端功能页面。

<p align="center">
<img src="./figures/admin.png" align=center />
</p>

### 检测检测情况

检测不同的时间段模型预测情况。

<p align="center">
<img src="./figures/DiffTime.png" align=center />
</p>

## 系统框架

### 主要组件功能

\*搜索规划：利用 LLM 的 query 分类，多轮改写，复杂查询分解，实现精准搜索。

\*时序预测：使用 Transformer 架构进行时序信号预测，实现工业领域长短期精准预测

\*企业知识库：使用搜索增强生成 RAG 解决私域知识融合、大模型落地中的幻觉、时效性

\*专业题库：利用 LLM 智能推理能力与概括归纳能力，实现出题批改，有效提高员工专业知识水平

<p align="center">
<img src="./figures/frame.png" alt="300" align=center />
</p>

## 数据集

使用 Unified Time Series Datasets (UTSD)以促进大型时间序列模型和预训练的研究。数据集可在[huggingface](https://huggingface.co/datasets/thuml/UTSD)获取，以方便时间序列领域大型模型的研究和预训练。

<p align="center">
<img src="./figures/utsd.png" align=center />
</p>

## 安装与启动

### 使用 Docker 启动项目

1. 创建项目文件夹并进入目录：

   ```bash
   mkdir SeqProject
   cd SeqProject
   ```

2. 启动 docker

   ```bash
   docker-compose up
   ```

## 评估结果

选择仅解码器结构，对比其它已有的模型和结构，较普通模型具有极强泛化性仅用少量的数据便能击败 SOTA 模型。

<p align="center">
<img src="./figures/result.png" align=center />
</p>
