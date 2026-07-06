# RAG Project

A Retrieval-Augmented Generation (RAG) pipeline that ingests PDF documents, embeds them with Google Gemini, stores vectors in Pinecone, and answers questions using retrieved context and Gemini generation.

The project includes both a Jupyter notebook for building and experimenting with the pipeline, and a Streamlit web app for interactive Q&A over the indexed documents.

## How it works

```mermaid
flowchart LR
    PDF[PDF documents] --> Load[PyPDFLoader]
    Load --> Split[Text splitter]
    Split --> Embed[Gemini embeddings]
    Embed --> Pinecone[(Pinecone index)]
    Query[User question] --> QEmbed[Query embedding]
    QEmbed --> Search[Similarity search]
    Pinecone --> Search
    Search --> Filter[Score filter]
    Filter --> Context[Top-k chunks]
    Context --> LLM[Gemini 2.5 Flash]
    LLM --> Answer[Generated answer + sources]
```

1. **Load** — Read pages from PDFs with LangChain's `PyPDFLoader`.
2. **Chunk** — Split text into overlapping segments (`chunk_size=500`, `overlap=50`).
3. **Embed** — Generate vectors with `models/gemini-embedding-2` (3072 dimensions).
4. **Store** — Upsert vectors and metadata into a Pinecone serverless index.
5. **Retrieve** — Embed the user query, fetch the top matching chunks, and filter by a minimum relevance score.
6. **Generate** — Pass retrieved context to `models/gemini-2.5-flash` to produce an answer grounded in the documents, along with the source chunks.

## Tech stack

| Component | Technology |
|-----------|------------|
| Document loading | LangChain + PyPDF |
| Embeddings | Google Gemini (`gemini-embedding-2`) |
| Vector store | Pinecone (serverless) |
| Generation | Google Gemini (`gemini-2.5-flash`) |
| Web UI | Streamlit |
| Environment | Python 3.13, Jupyter |

## Prerequisites

- Python 3.10+
- API keys for:
  - [Google AI Studio](https://aistudio.google.com/) (Gemini)
  - [Pinecone](https://www.pinecone.io/)
- One or more PDF files to index

## Setup

### 1. Clone and enter the project

```bash
cd "RAG project"
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

```env
GEMINI_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=rag-llm-index
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
PDF_DIR=./data
MIN_RELEVANCE_SCORE=0.75
```

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Google Gemini API key |
| `PINECONE_API_KEY` | Yes | Pinecone API key |
| `PINECONE_INDEX_NAME` | No | Pinecone index name (default: `rag-llm-index`) |
| `PINECONE_CLOUD` | No | Pinecone cloud provider (default: `aws`) |
| `PINECONE_REGION` | No | Pinecone region (default: `us-east-1`) |
| `PDF_DIR` | No | Directory containing source PDFs (default: `./data`) |
| `MIN_RELEVANCE_SCORE` | No | Minimum similarity score to include a chunk (default: `0.75`) |

> **Note:** `.env` is gitignored. Never commit API keys.

### 5. Add your PDFs

Place your PDF documents inside the directory set by `PDF_DIR` (default: `./data/`). PDF files are gitignored by default.

## Usage

### Step 1 — Build the index (notebook)

Run the notebook once to load your PDFs, embed all chunks, and populate the Pinecone index:

```bash
jupyter notebook rag_pipeline.ipynb
```

Run all cells top to bottom. You only need to re-run this when your documents change.

#### Notebook cells (in order)

| Step | What it does |
|------|----------------|
| Imports & SSL setup | Load libraries and configure certificates |
| Environment | Load `.env` and validate keys / PDF directory |
| Load PDFs | Read all pages from every PDF in `PDF_DIR` |
| Chunk text | Split into searchable segments |
| Pinecone index | Create index if missing (3072-dim, dotproduct metric) |
| Embed & upsert | Embed each chunk and upload to Pinecone in batches |
| `retrieve()` | Search Pinecone and print top matches (retrieval only) |
| `generate_answer()` | Retrieve context + generate an answer with Gemini |

#### Example: retrieval only

```python
retrieve("What is the inclusion-exclusion principle?")
```

#### Example: full RAG answer

```python
generate_answer("How is Pascal's triangle related to the binomial theorem?")
```

### Step 2 — Run the web app

Once the index is built, launch the Streamlit chat interface:

```bash
streamlit run app.py
```

The app will open in your browser. Type a question and receive a Gemini-generated answer with expandable source chunks showing the relevant passages and their similarity scores.

## Project structure

```
RAG project/
├── app.py               # Streamlit web app (chat UI)
├── rag_core.py          # Shared RAG logic (init_clients, generate_answer)
├── rag_pipeline.ipynb   # Notebook for building and populating the index
├── requirements.txt     # Python dependencies
├── .env                 # Local secrets (not committed)
├── .env.example         # Environment variable template
├── .gitignore
└── README.md
```

## Configuration notes

- **Embedding model:** `models/gemini-embedding-2` produces **3072-dimensional** vectors. The Pinecone index must be created with `dimension=3072`. If you reuse an older index with a different dimension (e.g. 1536), upserts will fail with a dimension mismatch error — delete the old index or use a new index name.
- **Relevance filter:** Only chunks with a similarity score ≥ `MIN_RELEVANCE_SCORE` are passed to the LLM. If no chunks pass the threshold, the app returns a "no relevant information found" message instead of hallucinating.
- **Rate limiting:** The notebook sleeps between embedding calls (`0.7s`) to reduce API throttling on large documents.
- **Batch uploads:** Vectors are upserted to Pinecone in batches of 50.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Missing GEMINI_API_KEY` / `Missing PINECONE_API_KEY` | Add keys to `.env` |
| `PDF directory not found` | Create the `PDF_DIR` directory and add at least one PDF |
| Index not found when running `app.py` | Run `rag_pipeline.ipynb` first to build the index |
| `Vector dimension 3072 does not match the dimension of the index` | Delete the Pinecone index and re-run index creation, or change `PINECONE_INDEX_NAME` |
| SSL / certificate errors on Windows | The notebook includes a certifi SSL workaround cell — run it before API calls. `rag_core.py` applies the same fix automatically. |
| App returns "לא נמצא מידע רלוונטי" for every question | Lower `MIN_RELEVANCE_SCORE` in `.env` (e.g. `0.6`) or verify the index was populated |

## License

MIT 
