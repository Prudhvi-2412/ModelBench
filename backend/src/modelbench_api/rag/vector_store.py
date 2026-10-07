import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path

from qdrant_client import QdrantClient, models

from ..embedding_settings import EmbeddingSettings
from ..settings import ROOT
from .chunking import DocumentChunk


class LocalVectorStore:
    """Persistent local Qdrant store; collection is isolated per embedding space."""

    def __init__(self, path: str) -> None:
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = ROOT / resolved
        resolved.parent.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(resolved))

    @staticmethod
    def collection_for(settings: EmbeddingSettings) -> str:
        key = f"{settings.model_id}:{settings.dimension}:{settings.provider}"
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
        return f"modelbench_{digest}"

    def upsert_document(
        self,
        *,
        document_id: str,
        filename: str,
        chunks: list[DocumentChunk],
        vectors: list[list[float]],
        embedding_settings: EmbeddingSettings,
    ) -> None:
        collection = self.collection_for(embedding_settings)
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(
                    size=len(vectors[0]), distance=models.Distance.COSINE
                ),
            )
        timestamp = datetime.now(timezone.utc).isoformat()
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "document_id": document_id,
                    "chunk_id": f"{document_id}:{chunk.index}",
                    "filename": filename,
                    "page_number": chunk.page_number,
                    "chunk_text": chunk.text,
                    "embedding_model": embedding_settings.model_id,
                    "created_at": timestamp,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(collection_name=collection, points=points, wait=True)

    def search(
        self,
        *,
        vector: list[float],
        embedding_settings: EmbeddingSettings,
        top_k: int,
        document_id: str | None = None,
    ) -> list[dict]:
        collection = self.collection_for(embedding_settings)
        if not self.client.collection_exists(collection):
            return []
        query_filter = None
        if document_id:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id", match=models.MatchValue(value=document_id)
                    )
                ]
            )
        response = self.client.query_points(
            collection_name=collection,
            query=vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        return [
            {
                **(point.payload or {}),
                "score": float(point.score),
            }
            for point in response.points
        ]

    def close(self) -> None:
        self.client.close()
