# ADVI - Smart Academic Advisor

ADVI is an intelligent academic advisor for the Faculty of Engineering at Mansoura University. Built with FastAPI, LangChain, and advanced LLMs (OpenAI, Gemini), ADVI provides students with accurate answers regarding academic regulations, courses, schedules, and more, leveraging Retrieval-Augmented Generation (RAG).

## Features

- **Interactive Chat**: Text and voice-based chat for natural interaction.
- **Gemini Live Sessions**: Real-time bidirectional voice chat using Google Gemini Live.
- **RAG Architecture**: Accurately retrieves information from the Faculty of Engineering's academic regulations and documents.
- **Smart Document Processing**: Handles PDF, DOCX, and TXT files, with intelligent table extraction using Microsoft MarkItDown.
- **Multi-Agent System**: Specialized agents for vector database retrieval, job market inquiries, and course recommendations.
- **Hybrid Search**: Combines semantic search (FAISS/Qdrant) and keyword search (BM25) with Reciprocal Rank Fusion (RRF).

## Requirements

- Python 3.8 or later

### Install Python using MiniConda

1) Download and install MiniConda from [here](https://docs.anaconda.com/free/miniconda/#quick-command-line-install)
2) Create a new environment using the following command:
```bash
conda create -n advi python=3.8
```
3) Activate the environment:
```bash
$ conda activate advi
```

### (Optional) Setup you command line interface for better readability

```bash
export PS1="\[\033[01;32m\]\u@\h:\w\n\[\033[00m\]\$ "
```

## Installation

### Install the required packages

```bash
$ pip install -r requirements.txt
```

### Setup the environment variables

```bash
$ cp .env.example .env
```

Set your environment variables in the `.env` file. Like `OPENAI_API_KEY` value.

## Run Docker Compose Services

```bash
$ cd docker
$ cp .env.example .env
```

- update `.env` with your credentials



```bash
$ cd docker
$ sudo docker compose up -d
```

## Run the FastAPI server

```bash
$ uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
