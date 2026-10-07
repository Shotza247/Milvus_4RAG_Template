from typing import Dict, Any
from pymilvus import connections, utility
from app.core.config import settings
from app.core.logging import logger


class MilvusConnectionManager:
    _connected: bool = False

    @classmethod
    def connect(cls) -> bool:
        """Connect to Milvus instance."""
        try:
            logger.info(
                f"Connecting to Milvus standalone at {settings.MILVUS_HOST}:{settings.MILVUS_PORT}..."
            )
            connections.connect(
                alias="default",
                host=settings.MILVUS_HOST,
                port=settings.MILVUS_PORT,
                user=settings.MILVUS_USER,
                password=settings.MILVUS_PASSWORD,
                secure=settings.MILVUS_SECURE,
            )
            cls._connected = True
            server_version = utility.get_server_version()
            logger.info(f"Connected to Milvus successfully. Server Version: {server_version}")
            return True
        except Exception as e:
            logger.warning(f"Could not connect to Milvus ({e}). Running in detached or mock-ready mode.")
            cls._connected = False
            return False

    @classmethod
    def disconnect(cls) -> None:
        """Disconnect from Milvus instance."""
        if cls._connected:
            try:
                connections.disconnect(alias="default")
                cls._connected = False
                logger.info("Disconnected from Milvus.")
            except Exception as e:
                logger.error(f"Error disconnecting from Milvus: {e}")

    @classmethod
    def is_connected(cls) -> bool:
        """Check if connection is alive."""
        if not cls._connected:
            return False
        try:
            return connections.has_connection("default")
        except Exception:
            return False

    @classmethod
    def health_check(cls) -> Dict[str, Any]:
        """Perform a deep health check on Milvus."""
        try:
            if not cls.is_connected():
                # Attempt lazy reconnect
                connected = cls.connect()
                if not connected:
                    return {
                        "status": "degraded",
                        "connected": False,
                        "host": settings.MILVUS_HOST,
                        "port": settings.MILVUS_PORT,
                        "error": "Milvus unreachable",
                    }

            server_ver = utility.get_server_version()
            collections = utility.list_collections()
            return {
                "status": "healthy",
                "connected": True,
                "server_version": server_ver,
                "host": settings.MILVUS_HOST,
                "port": settings.MILVUS_PORT,
                "collections_count": len(collections),
                "collections": collections,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "connected": False,
                "host": settings.MILVUS_HOST,
                "port": settings.MILVUS_PORT,
                "error": str(e),
            }
