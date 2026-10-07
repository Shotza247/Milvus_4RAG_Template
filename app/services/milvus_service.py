import json
from typing import Any, Dict, List, Optional
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    utility,
)
from app.core.config import settings
from app.core.logging import logger
from app.core.milvus import MilvusConnectionManager
from app.services.interfaces import BaseVectorStore, DocumentChunk


class MilvusService(BaseVectorStore):
    """
    Concrete implementation of BaseVectorStore using PyMilvus.
    """

    def __init__(self):
        self._collections: Dict[str, Collection] = {}

    async def connect(self) -> bool:
        return MilvusConnectionManager.connect()

    async def disconnect(self) -> None:
        MilvusConnectionManager.disconnect()

    async def is_healthy(self) -> Dict[str, Any]:
        return MilvusConnectionManager.health_check()

    def _ensure_connected(self):
        if not MilvusConnectionManager.is_connected():
            MilvusConnectionManager.connect()

    async def has_collection(self, collection_name: str) -> bool:
        try:
            self._ensure_connected()
            return utility.has_collection(collection_name)
        except Exception as e:
            logger.error(f"Error checking collection {collection_name}: {e}")
            return False

    async def create_collection(
        self,
        collection_name: str,
        dimension: int = 384,
        metric_type: str = "COSINE",
        auto_id: bool = True,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Creates a collection with standard RAG fields:
        - id (INT64, Primary Key)
        - chunk_id (VARCHAR)
        - doc_id (VARCHAR)
        - text (VARCHAR)
        - metadata_json (VARCHAR)
        - vector (FLOAT_VECTOR)
        """
        try:
            self._ensure_connected()
            if utility.has_collection(collection_name):
                logger.info(f"Collection '{collection_name}' already exists.")
                col = Collection(collection_name)
                col.load()
                self._collections[collection_name] = col
                return True

            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="metadata_json", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension),
            ]

            schema = CollectionSchema(
                fields=fields,
                description=f"RAG vector collection: {collection_name}",
                enable_dynamic_field=True,
            )

            collection = Collection(name=collection_name, schema=schema)
            logger.info(f"Created collection schema for '{collection_name}'. Creating index...")

            # Create default HNSW or IVF index
            index_params = {
                "metric_type": metric_type.upper(),
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 200},
            }
            collection.create_index(field_name="vector", index_params=index_params)
            collection.load()
            self._collections[collection_name] = collection
            logger.info(f"Collection '{collection_name}' initialized and loaded into memory.")
            return True
        except Exception as e:
            logger.error(f"Failed to create collection '{collection_name}': {e}")
            raise e

    async def drop_collection(self, collection_name: str) -> bool:
        try:
            self._ensure_connected()
            if utility.has_collection(collection_name):
                utility.drop_collection(collection_name)
                self._collections.pop(collection_name, None)
                logger.info(f"Dropped collection '{collection_name}'.")
                return True
            return False
        except Exception as e:
            logger.error(f"Error dropping collection '{collection_name}': {e}")
            raise e

    async def list_collections(self) -> List[str]:
        try:
            self._ensure_connected()
            return utility.list_collections()
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []

    async def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        try:
            self._ensure_connected()
            if not utility.has_collection(collection_name):
                return {"error": f"Collection '{collection_name}' does not exist"}

            col = Collection(collection_name)
            col.load()
            indexes = [idx.to_dict() for idx in col.indexes]
            return {
                "collection_name": collection_name,
                "num_entities": col.num_entities,
                "description": col.description,
                "schema": col.schema.to_dict(),
                "indexes": indexes,
            }
        except Exception as e:
            logger.error(f"Error getting stats for '{collection_name}': {e}")
            return {"error": str(e)}

    async def insert_chunks(
        self, collection_name: str, chunks: List[DocumentChunk]
    ) -> int:
        if not chunks:
            return 0

        try:
            self._ensure_connected()
            if not utility.has_collection(collection_name):
                dim = len(chunks[0].embedding) if chunks[0].embedding else settings.EMBEDDING_DIMENSION
                await self.create_collection(collection_name=collection_name, dimension=dim)

            col = Collection(collection_name)

            chunk_ids = [c.chunk_id for c in chunks]
            doc_ids = [c.doc_id for c in chunks]
            texts = [c.text for c in chunks]
            metas = [json.dumps(c.metadata) for c in chunks]
            vectors = [c.embedding for c in chunks]

            entities = [
                chunk_ids,
                doc_ids,
                texts,
                metas,
                vectors,
            ]

            insert_result = col.insert(entities)
            col.flush()
            logger.info(f"Inserted {len(chunks)} chunks into '{collection_name}'.")
            return len(insert_result.primary_keys)
        except Exception as e:
            logger.error(f"Error inserting chunks into '{collection_name}': {e}")
            raise e

    async def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 10,
        filter_expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
    ) -> List[DocumentChunk]:
        try:
            self._ensure_connected()
            if not utility.has_collection(collection_name):
                logger.warning(f"Collection '{collection_name}' does not exist.")
                return []

            col = Collection(collection_name)
            col.load()

            fields = output_fields or ["chunk_id", "doc_id", "text", "metadata_json"]
            search_params = {"metric_type": "COSINE", "params": {"ef": 64}}

            res = col.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                expr=filter_expr,
                output_fields=fields,
            )

            hits = res[0]
            results: List[DocumentChunk] = []

            for hit in hits:
                meta = {}
                try:
                    meta_raw = hit.entity.get("metadata_json")
                    if meta_raw:
                        meta = json.loads(meta_raw)
                except Exception:
                    pass

                # Convert distance/similarity score
                score = round(float(hit.distance), 4)

                results.append(
                    DocumentChunk(
                        chunk_id=hit.entity.get("chunk_id") or str(hit.id),
                        doc_id=hit.entity.get("doc_id") or "",
                        text=hit.entity.get("text") or "",
                        metadata=meta,
                        similarity_score=score,
                    )
                )

            return results
        except Exception as e:
            logger.error(f"Search failed on '{collection_name}': {e}")
            raise e


milvus_service = MilvusService()
