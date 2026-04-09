from sqlmodel import SQLModel, Field, Column, Index
from pgvector.sqlalchemy import Vector # pgvector's SQLAlchemy integration


class UserBiometrics(SQLModel, table=True):
    __tablename__ = "user_biometrics"
    __table_args__ = (
        # HNSW Index for ultra-fast Cosine Similarity searches
        # This translates to: USING hnsw (embedding vector_cosine_ops)
        Index(
            "ix_user_biometrics_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    # 1. Primary Key
    id: int | None = Field(default=None, primary_key=True)

    # 2. Foreign ID from your main application (Indexed for fast lookups)
    external_user_id: str = Field(index=True, unique=True, nullable=False)

    # 3. Vector Column (Using sa_column for deep SQLAlchemy integration)
    # Dimensions fixed to 512 for your EfficientNet/FaceNet model
    embedding: list[float] = Field(
        sa_column=Column(Vector(512), nullable=False)
    )

    # 4. Metadata
    model_version: str = Field(default="efficientnet_v1")
    is_active: bool = Field(default=True)
