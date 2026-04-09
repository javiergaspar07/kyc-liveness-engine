from core.config import settings

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

engine = None
async_session = None

def init_db():
    global engine, async_session

    engine = create_async_engine(settings.database_url, echo=False, future=True)

    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

async def get_session():
    async with async_session() as session:
        yield session