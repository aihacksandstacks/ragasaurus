# RAGasaurus

A powerful Retrieval-Augmented Generation (RAG) system built with LlamaIndex and MCP (Machine Communication Protocol) support.

## Overview

RAGasaurus is a modular, extensible RAG system designed to enhance AI assistant capabilities with customized knowledge bases. It leverages LlamaIndex for efficient document processing and retrieval, while offering a clean MCP interface that allows AI coding assistants to easily use its capabilities.

### Key Features

- **Document Ingestion**: Support for local files, web content, and remote sources
- **Vector Storage**: PostgreSQL-based vector storage for efficient similarity search
- **MCP Integration**: Machine Communication Protocol support for AI assistant interoperability
- **Modular Architecture**: Clean separation of concerns across ingestion, storage, and retrieval
- **Extensible Design**: Easy to add new document types, embeddings, or retrieval strategies

## Architecture

RAGasaurus is organized into several core components:

- **Data Ingestion**: Modules for ingesting content from various sources
- **Persistence**: Database and storage management
- **RAG Core**: Core functionality for indexing and querying documents
- **Models**: Configuration and setup for embedding models
- **MCP Server**: Interface for AI coding assistants

## Getting Started

### Prerequisites

- Python 3.12+
- PostgreSQL with pgvector extension
- Required Python packages (see `requirements.txt`)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/RAGasaurus.git
   cd RAGasaurus
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure your environment variables or update `config.py` with your settings.

### Usage

#### Starting the MCP Server

```bash
python mcp_server.py
```

#### Using the CLI to Interact with the Server

List available tools:
```bash
python mcp_client_cli.py list-tools coding_assistant_rag_server
```

Call a specific tool:
```bash 
python mcp_client_cli.py call-tool coding_assistant_rag_server ingest_local_fs_directory --tool-args-json '{"directory_path": "/path/to/documents", "force_refresh": false}'
```

Query the knowledge base:
```bash
python mcp_client_cli.py call-tool coding_assistant_rag_server query_knowledge_base --tool-args-json '{"query_text": "What is RAGasaurus?", "top_k": 5}'
```

## MCP Tool Reference

RAGasaurus exposes the following tools through its MCP interface:

- **ingest_local_fs_directory**: Ingest documents from a local directory
- **ingest_web_content**: Crawl and ingest content from a website
- **ingest_url_list_from_remote_file**: Process a list of URLs from a remote file
- **ingest_raw_text_content_from_single_url**: Fetch and ingest a single text file
- **query_knowledge_base**: Send a natural language query to the RAG system

## Development

### Project Structure

```
RAGasaurus/
├── data_ingestion/        # Document ingestion modules
├── models/                # Embedding model configuration
├── persistence/           # Database and storage management
├── rag_core/              # Core indexing and querying functionality
├── utils/                 # Utility functions
├── config.py              # Configuration settings
├── mcp_server.py          # MCP server implementation
├── mcp_client_cli.py      # CLI for interacting with the MCP server
└── requirements.txt       # Project dependencies
```

## License

[Your chosen license]

## Acknowledgments

- LlamaIndex for the core RAG functionality
- FastMCP for the Machine Communication Protocol implementation
