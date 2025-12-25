# Vector Databases Production Guide

## Introduction

Vector databases are specialized databases designed for storing and querying high-dimensional vectors (embeddings). They're the foundation of modern RAG systems, semantic search, recommendation engines, and similarity-based applications.

**What You'll Learn:**
- Vector database comparison and selection
- Production deployment patterns
- Embedding model selection
- Performance optimization
- Cost analysis and scaling

---

## Table of Contents

1. [Vector Database Comparison](#vector-database-comparison)
2. [Embedding Models](#embedding-models)
3. [Production Architecture](#production-architecture)
4. [Implementation Examples](#implementation-examples)
5. [Performance Optimization](#performance-optimization)
6. [Cost Analysis](#cost-analysis)
7. [Migration Strategies](#migration-strategies)

---

## Vector Database Comparison

### Overview Matrix

| Database | Type | Best For | Pricing | Scale | Cloud/Self-Hosted |
|----------|------|----------|---------|-------|-------------------|
| **Pinecone** | Managed | Production RAG | $70/mo+ | 100M+ vectors | Cloud only |
| **Weaviate** | Open Source | Hybrid search | Free/Enterprise | 10B+ vectors | Both |
| **Qdrant** | Open Source | High performance | Free/Cloud | 1B+ vectors | Both |
| **Milvus** | Open Source | Large scale | Free/Enterprise | 10B+ vectors | Both |
| **Chroma** | Open Source | Development | Free | 1M+ vectors | Self-hosted |
| **FAISS** | Library | Research/Prototyping | Free | Memory limited | Self-hosted |
| **pgvector** | PostgreSQL Extension | Simple use cases | Free | 1M+ vectors | Self-hosted |

### Detailed Comparison

#### Pinecone

```python
# Pinecone - Managed, Production-Ready

✅ Pros:
• Fully managed (no ops overhead)
• Excellent performance and reliability
• Auto-scaling and backups included
• Simple API
• Good documentation

❌ Cons:
• Most expensive option
• Vendor lock-in
• Limited customization
• Cloud-only (no self-hosting)

💰 Cost:
• Starter: $70/month (1M vectors, 1 pod)
• Standard: $0.095/hour per pod
• ~$600/month for 10M vectors

🎯 Use When:
• Need turnkey solution
• Limited ops resources
• Can afford premium pricing
• Require 99.9% uptime SLA
```

```python
# Example: Pinecone Setup
import pinecone
from sentence_transformers import SentenceTransformer

# Initialize
pinecone.init(
    api_key="your-api-key",
    environment="us-west1-gcp"
)

# Create index
index_name = "production-docs"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=768,  # all-mpnet-base-v2
        metric="cosine",
        pods=2,  # For high availability
        replicas=2,  # For fault tolerance
        pod_type="p1.x1"  # Performance tier
    )

index = pinecone.Index(index_name)

# Embed and upsert
model = SentenceTransformer('all-mpnet-base-v2')
texts = ["Document 1", "Document 2", "Document 3"]
embeddings = model.encode(texts)

# Batch upsert
vectors = [
    (f"doc_{i}", emb.tolist(), {"text": text})
    for i, (emb, text) in enumerate(zip(embeddings, texts))
]
index.upsert(vectors=vectors, namespace="production")

# Query
query_embedding = model.encode(["search query"])[0]
results = index.query(
    vector=query_embedding.tolist(),
    top_k=5,
    namespace="production",
    include_metadata=True
)

for match in results['matches']:
    print(f"Score: {match['score']:.4f} - {match['metadata']['text']}")
```

#### Weaviate

```python
# Weaviate - Open Source, Hybrid Search

✅ Pros:
• True hybrid search (dense + sparse)
• Built-in vectorization modules
• GraphQL API
• Active community
• Can self-host or use cloud

❌ Cons:
• More complex setup
• Requires more ops knowledge
• Module system can be confusing
• Resource intensive

💰 Cost:
• Self-hosted: Infrastructure only (~$200-500/month)
• Weaviate Cloud: Starting at $25/month
• Enterprise: Custom pricing

🎯 Use When:
• Need hybrid search
• Have ops resources
• Want flexible deployment
• Need advanced filtering
```

```python
# Example: Weaviate Setup
import weaviate
from weaviate.util import generate_uuid5

# Connect to Weaviate
client = weaviate.Client(
    url="http://localhost:8080",
    additional_headers={
        "X-OpenAI-Api-Key": "your-openai-key"  # For vectorization
    }
)

# Define schema
schema = {
    "classes": [{
        "class": "Document",
        "description": "A document with text content",
        "vectorizer": "text2vec-openai",
        "moduleConfig": {
            "text2vec-openai": {
                "model": "text-embedding-ada-002",
                "type": "text"
            }
        },
        "properties": [
            {
                "name": "content",
                "dataType": ["text"],
                "description": "Document content",
                "moduleConfig": {
                    "text2vec-openai": {
                        "skip": False,
                        "vectorizePropertyName": False
                    }
                }
            },
            {
                "name": "title",
                "dataType": ["string"],
                "description": "Document title"
            },
            {
                "name": "created_at",
                "dataType": ["date"],
                "description": "Creation timestamp"
            }
        ]
    }]
}

# Create schema
client.schema.create(schema)

# Add documents
documents = [
    {"content": "AI is transforming industries", "title": "AI Report"},
    {"content": "Machine learning requires data", "title": "ML Basics"},
]

with client.batch as batch:
    batch.batch_size = 100
    for doc in documents:
        batch.add_data_object(
            data_object=doc,
            class_name="Document",
            uuid=generate_uuid5(doc["title"])
        )

# Hybrid search (dense + sparse)
result = client.query.get(
    "Document",
    ["content", "title"]
).with_hybrid(
    query="artificial intelligence",
    alpha=0.5  # 0=keyword, 1=vector, 0.5=balanced
).with_limit(5).do()

for doc in result['data']['Get']['Document']:
    print(f"{doc['title']}: {doc['content']}")
```

#### Qdrant

```python
# Qdrant - High Performance, Developer-Friendly

✅ Pros:
• Excellent performance
• Rich filtering capabilities
• Great documentation
• Active development
• Rust-based (fast)
• Good Python client

❌ Cons:
• Smaller community than Milvus/Weaviate
• Fewer integrations
• Cloud offering is newer

💰 Cost:
• Self-hosted: Free (infrastructure cost)
• Qdrant Cloud: $25/month starter
• Enterprise: Custom

🎯 Use When:
• Need high performance
• Want rich filtering
• Value developer experience
• Can self-host
```

```python
# Example: Qdrant Setup
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import uuid

# Initialize client
client = QdrantClient(
    url="http://localhost:6333",
    # Or cloud: url="https://xyz.qdrant.io", api_key="your-key"
)

# Create collection
collection_name = "documents"
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=768,  # Embedding dimension
        distance=Distance.COSINE
    )
)

# Prepare data
model = SentenceTransformer('all-mpnet-base-v2')
documents = [
    {"text": "Quantum computing is emerging", "category": "tech"},
    {"text": "Climate change impacts us all", "category": "environment"},
    {"text": "AI ethics matter", "category": "tech"},
]

# Create embeddings and upsert
points = []
for doc in documents:
    embedding = model.encode(doc["text"])
    points.append(
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding.tolist(),
            payload=doc
        )
    )

client.upsert(
    collection_name=collection_name,
    points=points
)

# Search with filtering
query_text = "artificial intelligence"
query_embedding = model.encode(query_text)

results = client.search(
    collection_name=collection_name,
    query_vector=query_embedding.tolist(),
    limit=5,
    query_filter={
        "must": [
            {"key": "category", "match": {"value": "tech"}}
        ]
    }
)

for result in results:
    print(f"Score: {result.score:.4f}")
    print(f"Text: {result.payload['text']}")
    print(f"Category: {result.payload['category']}\n")
```

---

## Embedding Models

### Model Selection Guide

```python
# Embedding Model Comparison

| Model | Dimension | Performance | Cost | Use Case |
|-------|-----------|-------------|------|----------|
| **OpenAI text-embedding-3-large** | 3072 | Excellent | $0.13/1M tokens | Production, best quality |
| **OpenAI text-embedding-3-small** | 1536 | Very Good | $0.02/1M tokens | Cost-effective production |
| **Cohere embed-v3** | 1024 | Excellent | $0.10/1M tokens | Multilingual |
| **all-mpnet-base-v2** | 768 | Good | Free (self-host) | General purpose |
| **all-MiniLM-L6-v2** | 384 | Fair | Free (self-host) | Fast, low-resource |
| **BGE-large-en** | 1024 | Excellent | Free (self-host) | Open-source SOTA |
| **E5-large-v2** | 1024 | Excellent | Free (self-host) | Research-backed |
```

### Implementation Examples

```python
# Example: Multi-Model Embeddings

from typing import List, Literal
import numpy as np

class EmbeddingService:
    """Unified embedding service supporting multiple models"""
    
    def __init__(self, model_type: Literal["openai", "cohere", "local"]):
        self.model_type = model_type
        self._init_model()
    
    def _init_model(self):
        if self.model_type == "openai":
            from openai import OpenAI
            self.client = OpenAI()
            self.model_name = "text-embedding-3-small"
            self.dimension = 1536
            
        elif self.model_type == "cohere":
            import cohere
            self.client = cohere.Client("your-api-key")
            self.model_name = "embed-english-v3.0"
            self.dimension = 1024
            
        elif self.model_type == "local":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-mpnet-base-v2')
            self.dimension = 768
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts"""
        
        if self.model_type == "openai":
            response = self.client.embeddings.create(
                model=self.model_name,
                input=texts
            )
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings)
        
        elif self.model_type == "cohere":
            response = self.client.embed(
                texts=texts,
                model=self.model_name,
                input_type="search_document"  # or "search_query"
            )
            return np.array(response.embeddings)
        
        elif self.model_type == "local":
            return self.model.encode(texts)
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a search query (may differ from documents)"""
        
        if self.model_type == "cohere":
            # Cohere has different input types for queries
            response = self.client.embed(
                texts=[query],
                model=self.model_name,
                input_type="search_query"
            )
            return np.array(response.embeddings[0])
        else:
            return self.embed([query])[0]

# Usage
embedder = EmbeddingService("local")  # or "openai", "cohere"
texts = ["Document 1", "Document 2"]
embeddings = embedder.embed(texts)
query_embedding = embedder.embed_query("search query")
```

### Chunking Strategies

```python
# Document Chunking for Optimal Embeddings

from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

class SmartChunker:
    """Intelligent document chunking for embeddings"""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        model: str = "text-embedding-3-small"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.encoding_for_model(model)
        
        # Different strategies for different content types
        self.splitters = {
            "code": RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\nclass ", "\n\ndef ", "\n\n", "\n", " "]
            ),
            "markdown": RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n## ", "\n### ", "\n\n", "\n", " "]
            ),
            "general": RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        }
    
    def chunk_document(
        self,
        text: str,
        doc_type: str = "general",
        metadata: Dict = None
    ) -> List[Dict]:
        """Chunk document with metadata preservation"""
        
        splitter = self.splitters.get(doc_type, self.splitters["general"])
        chunks = splitter.split_text(text)
        
        result = []
        for i, chunk in enumerate(chunks):
            # Calculate token count
            tokens = len(self.tokenizer.encode(chunk))
            
            chunk_metadata = {
                **(metadata or {}),
                "chunk_index": i,
                "total_chunks": len(chunks),
                "token_count": tokens,
                "char_count": len(chunk)
            }
            
            result.append({
                "content": chunk,
                "metadata": chunk_metadata
            })
        
        return result
    
    def chunk_by_semantic_similarity(
        self,
        text: str,
        embedder,
        threshold: float = 0.5
    ) -> List[str]:
        """Chunk based on semantic similarity (advanced)"""
        
        sentences = text.split(". ")
        if len(sentences) < 2:
            return [text]
        
        # Embed all sentences
        embeddings = embedder.embed(sentences)
        
        # Group by similarity
        chunks = []
        current_chunk = [sentences[0]]
        
        for i in range(1, len(sentences)):
            # Calculate similarity with previous sentence
            similarity = np.dot(
                embeddings[i],
                embeddings[i-1]
            )
            
            if similarity < threshold or len(" ".join(current_chunk)) > self.chunk_size:
                # Start new chunk
                chunks.append(". ".join(current_chunk) + ".")
                current_chunk = [sentences[i]]
            else:
                current_chunk.append(sentences[i])
        
        if current_chunk:
            chunks.append(". ".join(current_chunk) + ".")
        
        return chunks

# Usage
chunker = SmartChunker(chunk_size=512, chunk_overlap=50)

# For code
code = "class MyClass:\n    def method1(self):\n        pass..."
code_chunks = chunker.chunk_document(
    code,
    doc_type="code",
    metadata={"file": "myclass.py", "language": "python"}
)

# For markdown
markdown = "# Title\n\n## Section 1\n\nContent..."
md_chunks = chunker.chunk_document(
    markdown,
    doc_type="markdown",
    metadata={"file": "docs.md"}
)
```

---

## Production Architecture

### Architecture Pattern 1: Simple RAG

```python
# Simple RAG with Vector Database

import asyncio
from typing import List, Dict
import numpy as np

class ProductionRAG:
    """Production-ready RAG system"""
    
    def __init__(
        self,
        vector_db_client,
        embedder,
        llm_client,
        collection_name: str
    ):
        self.vector_db = vector_db_client
        self.embedder = embedder
        self.llm = llm_client
        self.collection_name = collection_name
    
    async def index_documents(
        self,
        documents: List[Dict],
        batch_size: int = 100
    ):
        """Index documents with batching"""
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # Generate embeddings
            texts = [doc["content"] for doc in batch]
            embeddings = self.embedder.embed(texts)
            
            # Prepare vectors
            vectors = []
            for j, (doc, emb) in enumerate(zip(batch, embeddings)):
                vectors.append({
                    "id": doc.get("id", f"doc_{i+j}"),
                    "vector": emb.tolist(),
                    "payload": {
                        "content": doc["content"],
                        "metadata": doc.get("metadata", {})
                    }
                })
            
            # Upsert to vector DB
            await self._upsert_batch(vectors)
            
            print(f"Indexed {i + len(batch)}/{len(documents)} documents")
    
    async def _upsert_batch(self, vectors: List[Dict]):
        """Upsert vectors to database"""
        # Implementation depends on vector DB
        pass
    
    async def query(
        self,
        question: str,
        k: int = 5,
        rerank: bool = True
    ) -> Dict:
        """Query with optional reranking"""
        
        # Step 1: Generate query embedding
        query_embedding = self.embedder.embed_query(question)
        
        # Step 2: Vector search
        results = await self._vector_search(
            query_embedding,
            k=k*2 if rerank else k
        )
        
        # Step 3: Rerank (optional)
        if rerank:
            results = await self._rerank(question, results, k)
        
        # Step 4: Build context
        context = self._build_context(results)
        
        # Step 5: Generate answer
        answer = await self._generate_answer(question, context)
        
        return {
            "answer": answer,
            "sources": [r["payload"] for r in results],
            "confidence": self._calculate_confidence(results)
        }
    
    async def _vector_search(
        self,
        embedding: np.ndarray,
        k: int
    ) -> List[Dict]:
        """Perform vector similarity search"""
        # Implementation depends on vector DB
        pass
    
    async def _rerank(
        self,
        query: str,
        results: List[Dict],
        k: int
    ) -> List[Dict]:
        """Rerank results using cross-encoder or LLM"""
        
        # Simple reranking with LLM
        texts = [r["payload"]["content"] for r in results]
        scores = []
        
        for text in texts:
            prompt = f"""Rate relevance 0-10:
Query: {query}
Text: {text[:500]}
Relevance score:"""
            
            score = await self.llm.score(prompt)
            scores.append(score)
        
        # Sort by score
        ranked = sorted(
            zip(results, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [r for r, s in ranked[:k]]
    
    def _build_context(self, results: List[Dict]) -> str:
        """Build context from search results"""
        context_parts = []
        for i, result in enumerate(results, 1):
            content = result["payload"]["content"]
            context_parts.append(f"[{i}] {content}")
        return "\n\n".join(context_parts)
    
    async def _generate_answer(
        self,
        question: str,
        context: str
    ) -> str:
        """Generate answer using LLM"""
        
        prompt = f"""Answer based on context only:

Context:
{context}

Question: {question}

Answer:"""
        
        return await self.llm.complete(prompt)
    
    def _calculate_confidence(self, results: List[Dict]) -> float:
        """Calculate answer confidence from scores"""
        if not results:
            return 0.0
        
        scores = [r.get("score", 0) for r in results]
        return float(np.mean(scores))
```

### Architecture Pattern 2: Hybrid Search

```python
# Hybrid Search (Dense + Sparse)

from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
import numpy as np

class HybridSearchRAG:
    """RAG with hybrid search (vector + keyword)"""
    
    def __init__(
        self,
        vector_db_client,
        embedder,
        llm_client,
        alpha: float = 0.5  # Vector weight (1-alpha for BM25)
    ):
        self.vector_db = vector_db_client
        self.embedder = embedder
        self.llm = llm_client
        self.alpha = alpha
        self.bm25 = None
        self.documents = []
    
    def index_documents(self, documents: List[Dict]):
        """Index for both vector and keyword search"""
        
        # Store documents
        self.documents = documents
        
        # Build BM25 index
        tokenized_docs = [
            doc["content"].lower().split()
            for doc in documents
        ]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        # Build vector index
        texts = [doc["content"] for doc in documents]
        embeddings = self.embedder.embed(texts)
        
        self.vector_db.upsert(
            embeddings=embeddings,
            documents=documents
        )
    
    def hybrid_search(
        self,
        query: str,
        k: int = 10
    ) -> List[Tuple[Dict, float]]:
        """Combine vector and keyword search"""
        
        # Vector search
        query_embedding = self.embedder.embed_query(query)
        vector_results = self.vector_db.search(query_embedding, k=k)
        vector_scores = {
            r["id"]: r["score"]
            for r in vector_results
        }
        
        # BM25 search
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # Normalize scores
        vector_norm = self._normalize_scores(list(vector_scores.values()))
        bm25_norm = self._normalize_scores(bm25_scores)
        
        # Combine scores
        combined_scores = {}
        for i, doc in enumerate(self.documents):
            doc_id = doc["id"]
            v_score = vector_norm.get(doc_id, 0)
            b_score = bm25_norm[i]
            
            combined_scores[doc_id] = (
                self.alpha * v_score +
                (1 - self.alpha) * b_score
            )
        
        # Sort and return top-k
        top_docs = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]
        
        return [
            (self._get_doc_by_id(doc_id), score)
            for doc_id, score in top_docs
        ]
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range"""
        scores = np.array(scores)
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return ((scores - min_score) / (max_score - min_score)).tolist()
    
    def _get_doc_by_id(self, doc_id: str) -> Dict:
        """Get document by ID"""
        for doc in self.documents:
            if doc["id"] == doc_id:
                return doc
        return None
```

---

## Performance Optimization

### Indexing Optimization

```python
# Optimize Vector Database Indexing

class OptimizedIndexer:
    """Optimized batch indexing"""
    
    def __init__(
        self,
        vector_db,
        embedder,
        batch_size: int = 100,
        num_workers: int = 4
    ):
        self.vector_db = vector_db
        self.embedder = embedder
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    async def index_documents_parallel(
        self,
        documents: List[Dict]
    ):
        """Parallel document indexing"""
        
        from concurrent.futures import ThreadPoolExecutor
        import asyncio
        
        # Split into batches
        batches = [
            documents[i:i + self.batch_size]
            for i in range(0, len(documents), self.batch_size)
        ]
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(self._process_batch, batch)
                for batch in batches
            ]
            
            for future in futures:
                await asyncio.wrap_future(future)
    
    def _process_batch(self, batch: List[Dict]):
        """Process single batch"""
        texts = [doc["content"] for doc in batch]
        embeddings = self.embedder.embed(texts)
        
        self.vector_db.upsert(
            embeddings=embeddings,
            documents=batch
        )

# Usage
indexer = OptimizedIndexer(
    vector_db=qdrant_client,
    embedder=embedding_service,
    batch_size=100,
    num_workers=4
)

await indexer.index_documents_parallel(documents)
```

### Query Optimization

```python
# Optimize Query Performance

class QueryOptimizer:
    """Optimize vector search queries"""
    
    def __init__(self, vector_db, cache_ttl: int = 3600):
        self.vector_db = vector_db
        self.cache = {}  # Use Redis in production
        self.cache_ttl = cache_ttl
    
    async def search_with_cache(
        self,
        query_embedding: np.ndarray,
        k: int = 10
    ) -> List[Dict]:
        """Search with semantic caching"""
        
        # Generate cache key (quantized embedding)
        cache_key = self._generate_cache_key(query_embedding)
        
        # Check cache
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Perform search
        results = await self.vector_db.search(
            vector=query_embedding,
            limit=k
        )
        
        # Cache results
        self.cache[cache_key] = results
        
        return results
    
    def _generate_cache_key(
        self,
        embedding: np.ndarray,
        precision: int = 2
    ) -> str:
        """Generate cache key from embedding"""
        # Quantize embedding for similar queries
        quantized = np.round(embedding, decimals=precision)
        return hash(quantized.tobytes())
    
    async def search_with_filters(
        self,
        query_embedding: np.ndarray,
        filters: Dict,
        k: int = 10
    ) -> List[Dict]:
        """Search with efficient filtering"""
        
        # Use native filter support when available
        results = await self.vector_db.search(
            vector=query_embedding,
            limit=k,
            filter=filters  # Push filter to DB level
        )
        
        return results
```

---

## Cost Analysis

### Cost Comparison (10M vectors, 1M queries/month)

```python
# Monthly Cost Breakdown

Pinecone:
  Storage: $200 (2 pods)
  Queries: Included
  Total: ~$200-300/month

Weaviate (Self-Hosted):
  Infrastructure: $150 (3x t3.large EC2)
  Storage: $50 (500GB EBS)
  Ops Time: $200 (10 hours @$20/hr)
  Total: ~$400/month

Qdrant (Self-Hosted):
  Infrastructure: $100 (2x t3.large EC2)
  Storage: $40 (400GB EBS)
  Ops Time: $100 (5 hours @$20/hr)
  Total: ~$240/month

Recommendation:
- Quick start: Pinecone
- Best value: Qdrant self-hosted
- Enterprise: Weaviate (hybrid search)
```

---

## Migration Strategies

```python
# Migrate Between Vector Databases

class VectorDBMigrator:
    """Migrate data between vector databases"""
    
    def __init__(self, source_db, target_db):
        self.source = source_db
        self.target = target_db
    
    async def migrate(
        self,
        batch_size: int = 1000,
        validate: bool = True
    ):
        """Migrate all data"""
        
        total_migrated = 0
        
        async for batch in self._iter_source_batches(batch_size):
            # Write to target
            await self.target.upsert(batch)
            
            total_migrated += len(batch)
            print(f"Migrated {total_migrated} vectors")
            
            # Validate (optional)
            if validate:
                await self._validate_batch(batch)
    
    async def _iter_source_batches(self, batch_size: int):
        """Iterate source database in batches"""
        offset = 0
        while True:
            batch = await self.source.fetch(
                limit=batch_size,
                offset=offset
            )
            if not batch:
                break
            yield batch
            offset += batch_size
    
    async def _validate_batch(self, batch: List[Dict]):
        """Validate migrated data"""
        for item in batch[:10]:  # Sample validation
            source_result = await self.source.get(item["id"])
            target_result = await self.target.get(item["id"])
            
            # Compare embeddings
            similarity = np.dot(
                source_result["vector"],
                target_result["vector"]
            )
            
            if similarity < 0.99:
                raise ValueError(f"Migration validation failed for {item['id']}")
```

---

## Production Checklist

```markdown
✅ Vector Database Selection
  □ Evaluated 3+ options
  □ Cost analysis completed
  □ Proof of concept tested
  □ Scaling plan defined

✅ Embedding Model
  □ Model selected and tested
  □ Benchmark performance measured
  □ Cost per embed calculated
  □ Fallback model defined

✅ Indexing
  □ Chunking strategy finalized
  □ Metadata schema defined
  □ Batch indexing implemented
  □ Update/delete logic ready

✅ Search
  □ Hybrid search considered
  □ Filtering requirements met
  □ Reranking strategy chosen
  □ Performance benchmarked

✅ Monitoring
  □ Query latency tracked
  □ Error rate monitored
  □ Cost per query tracked
  □ Index size monitored

✅ Operations
  □ Backup strategy defined
  □ Disaster recovery tested
  □ Scaling triggers set
  □ On-call runbook created
```

---

*This guide covers production vector database deployment. For complete RAG systems, see [LLM Operations Guide](../ai-ml-frameworks/llm-operations-guide.md).*
