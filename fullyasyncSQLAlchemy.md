Migrating to a fully-async SQLAlchemy 2.0 stack is a sizeable, mechanical job but it follows a repeatable recipe.
Below is a battle-tested blueprint you can apply module-by-module until the synchronous ORM is gone.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    1. Prerequisites

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

• Dependencies
  pip install "sqlalchemy>=2.0" asyncpg aiosqlite

• Database URL
  postgresql+asyncpg://… or sqlite+aiosqlite://…

• Core scaffolding (already introduced earlier)

  from sqlalchemy.ext.asyncio import (
      AsyncSession, async_sessionmaker, create_async_engine
  )

  engine = create_async_engine(DATABASE_URL, echo=False)
  SessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False)

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    1. Generic rewrite pattern

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Sync ORM                                   →  Async 2.0

db.query(Model).filter(…).first()           stmt = select(Model).where(…)
                                            result = await db.scalar(stmt)

db.query(func.count()).scalar()             stmt = select(func.count()).select_from(Model)
                                            total = await db.scalar(stmt)

for row in db.query(Model)…                 result = await db.scalars(stmt)
                                            async for row in result: …

db.add(obj);  db.commit()                   db.add(obj);  await db.commit()

db.execute(raw_sql)                         await db.execute(text(raw_sql))

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    1. Dependency helper

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        yield session

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    1. Concrete examples from the codebase

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## A) /backend/app/routes/auth.py  – registration-available

# Before

user_count = db.query(User).count()

# After

stmt = select(func.count()).select_from(User)
user_count = await db.scalar(stmt)

## B) ChatService.get_thread

# Before

return self.db.query(ChatThread).filter(
    ChatThread.id == thread_id,
    ChatThread.user_id == user_id,
    ChatThread.is_archived == False
).first()

# After

stmt = (
    select(ChatThread)
    .where(
        ChatThread.id == thread_id,
        ChatThread.user_id == user_id,
        ChatThread.is_archived.is_(False),
    )
)
return await self.db.scalar(stmt)

## C) Pagination with order_by / limit / offset

stmt = (
    select(ChatThread)
    .where(ChatThread.project_id == project_id,
           ChatThread.user_id == user_id,
           ChatThread.is_archived.is_(include_archived is False))
    .order_by(desc(ChatThread.last_activity_at))
    .offset(offset)
    .limit(limit)
)
threads = (await self.db.scalars(stmt)).all()

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    1. Transaction helpers

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

For multi-step operations that must succeed or roll back together use the
session as an async-context manager:

async with db.begin():
    db.add(obj1)
    db.add(obj2)

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    1. Migration workflow

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    1. Replace get_db() provider with the async version and update FastAPI type-hints to AsyncSession.
    2. Touch each service / route:
       a. import select, text, func from sqlalchemy.future (or sqlalchemy).
       b. Convert query chain → select().
       c. Replace .first() / .all() with await db.scalar() / (await db.scalars()).all().
       d. Add await db.commit(), await db.flush() etc.
    3. Remove run_in_thread() fallbacks once the file no longer contains blocking calls.
    4. Run test-suite; fix leftover sync calls detected by “greenlet error: IO in wrong thread”.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    1. Tooling tips

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

• ripgrep + sed one-liners can automate 70-80 %:
  rg -l "db.query" | xargs sed -i 's/db.query(/select(/g'   (manual follow-up still needed)

• Enable mypy “sqlalchemy2-stubs”; it flags sync/async mismatches.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    1. Gradual rollout

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Because each request gets its own AsyncSession you can incrementally
migrate: convert one router at a time and keep run_in_thread() as a
fallback for untouched modules.  Nothing breaks mid-migration.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    1. Summary

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    1. Introduce async engine + session (done).
    2. Replace .query chains with select() + await pattern.
    3. Commit / flush / execute must be awaited.
    4. Remove thread-offloading once a module is fully async.

Following the template above you can refactor the repository in several
small PRs without a big-bang rewrite.
