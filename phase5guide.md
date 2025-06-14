Let me review and correct my Phase 5 implementation guide based on the actual codebase structure.You're right. Let me provide corrected instructions based on the actual codebase structure.

## **Phase 5 Implementation Guide - Corrected for Existing Codebase**

### **Part 1: Unified Search System**

#### **Step 1: Create Search Service**

**File: `backend/app/services/search_service.py`**
```python
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_, and_, func
import logging

from ..models import Document, Project, ChatThread, ChatMessage
from .vector_db_service import VectorDBService
from ..core.config import settings

logger = logging.getLogger(__name__)

class SearchService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.vector_service = VectorDBService()

    async def unified_search(
        self,
        query: str,
        user_id: str,
        project_id: Optional[str] = None,
        limit: int = 20
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Perform unified search across documents, chats, and vector embeddings.
        """
        results = {
            "documents": [],
            "chats": [],
            "semantic_matches": []
        }

        # 1. Document search
        doc_stmt = select(Document).join(Project).where(
            Project.user_id == user_id
        )

        if project_id:
            doc_stmt = doc_stmt.where(Document.project_id == project_id)

        doc_stmt = doc_stmt.where(
            or_(
                Document.title.ilike(f"%{query}%"),
                Document.content.ilike(f"%{query}%")
            )
        ).limit(limit)

        docs = (await self.db.scalars(doc_stmt)).all()

        results["documents"] = [
            {
                "id": str(doc.id),
                "title": doc.title,
                "preview": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                "project_id": str(doc.project_id),
                "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
                "type": "document"
            }
            for doc in docs
        ]

        # 2. Chat message search
        msg_stmt = select(ChatMessage).join(ChatThread).where(
            and_(
                ChatThread.user_id == user_id,
                ChatMessage.content.ilike(f"%{query}%")
            )
        )

        if project_id:
            msg_stmt = msg_stmt.where(ChatThread.project_id == project_id)

        msg_stmt = msg_stmt.limit(limit)
        messages = (await self.db.scalars(msg_stmt)).all()

        results["chats"] = [
            {
                "id": str(msg.id),
                "thread_id": str(msg.thread_id),
                "preview": msg.content[:200] + "..." if len(msg.content) > 200 else msg.content,
                "created_at": msg.created_at.isoformat(),
                "type": "chat"
            }
            for msg in messages
        ]

        # 3. Vector search
        try:
            collection_name = f"project_{project_id}" if project_id else f"user_{user_id}"

            semantic_results = await self.vector_service.search_similar(
                collection_name=collection_name,
                query_text=query,
                n_results=limit
            )

            if semantic_results:
                results["semantic_matches"] = [
                    {
                        "id": res.get("id"),
                        "content": res.get("content", "")[:200] + "...",
                        "score": res.get("score", 0),
                        "metadata": res.get("metadata", {}),
                        "type": "semantic"
                    }
                    for res in semantic_results
                ]
        except Exception as e:
            logger.error(f"Vector search failed: {e}")

        return results
```

#### **Step 2: Create Search Routes**

**File: `backend/app/routes/search.py`**
```python
from fastapi import APIRouter, Depends, Query, HTTPException
from typing import Optional
import uuid

from ..dependencies.auth import get_current_user, get_async_db
from ..services.search_service import SearchService
from ..models import User
from ..schemas.search import SearchResponse

router = APIRouter(prefix="/search", tags=["search"])

@router.get("", response_model=SearchResponse)
async def search(
    q: str = Query(..., min_length=1, max_length=200, description="Search query"),
    project_id: Optional[uuid.UUID] = Query(None, description="Filter by project"),
    limit: int = Query(20, ge=1, le=100, description="Max results per category"),
    current_user: User = Depends(get_current_user),
    db = Depends(get_async_db)
):
    """
    Unified search across documents, chats, and semantic content.
    """
    search_service = SearchService(db)

    results = await search_service.unified_search(
        query=q,
        user_id=str(current_user.id),
        project_id=str(project_id) if project_id else None,
        limit=limit
    )

    return SearchResponse(
        query=q,
        results=results,
        total_results=sum(len(v) for v in results.values())
    )
```

#### **Step 3: Create Search Schemas**

**File: `backend/app/schemas/search.py`**
```python
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class SearchResultItem(BaseModel):
    id: str
    type: str
    preview: str
    score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    # Type-specific fields
    title: Optional[str] = None
    project_id: Optional[str] = None
    thread_id: Optional[str] = None
    updated_at: Optional[datetime] = None
    created_at: Optional[datetime] = None

class SearchResults(BaseModel):
    documents: List[SearchResultItem] = Field(default_factory=list)
    chats: List[SearchResultItem] = Field(default_factory=list)
    semantic_matches: List[SearchResultItem] = Field(default_factory=list)

class SearchResponse(BaseModel):
    query: str
    results: SearchResults
    total_results: int
```

#### **Step 4: Update Routes Init**

**File: `backend/app/routes/__init__.py`**
- Add this line below the others:
  ```python
  from .search import router as search_router
  ```
- Also, add `"search_router",` to the `__all__` list.

#### **Step 5: Include Search Router in Main**

**File: `backend/app/main.py`**
- Add `search_router` to the tuple of imports:
  ```python
  from .routes import (
      auth_router,
      users_router,
      projects_router,
      templates_router,
      tags_router,
      activities_router,
      documents_router,
      search_router,  # Add this
  )
  ```
- And add `app.include_router(search_router)` after the other routers.

### **Part 2: Frontend Search Implementation**

#### **Step 6: Create SearchResults Page**

**File: `frontend/src/pages/SearchResults.tsx`**
```tsx
import React, { useState, useEffect } from 'react';
import { useSearchParams, Link, useNavigate } from 'react-router-dom';
import { Search, FileText, MessageSquare, Brain, ChevronRight, ArrowLeft } from 'lucide-react';
import { searchApi } from '../services/api';
import { Card } from '../components/common/Card';
import { Button } from '../components/common/Button';

interface SearchResult {
  documents: Array<{
    id: string;
    title: string;
    preview: string;
    project_id: string;
    updated_at: string;
    type: string;
  }>;
  chats: Array<{
    id: string;
    thread_id: string;
    preview: string;
    created_at: string;
    type: string;
  }>;
  semantic_matches: Array<{
    id: string;
    content: string;
    score: number;
    metadata: any;
    type: string;
  }>;
}

export default function SearchResults() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const query = searchParams.get('q') || '';
  const projectId = searchParams.get('project') || undefined;

  const [results, setResults] = useState<SearchResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (query) {
      performSearch();
    }
  }, [query, projectId]);

  const performSearch = async () => {
    setLoading(true);
    setError(null);

    try {
      const data = await searchApi.search(query, projectId);
      setResults(data.results);
    } catch (err) {
      setError('Failed to perform search. Please try again.');
      console.error('Search error:', err);
    } finally {
      setLoading(false);
    }
  };

  const renderDocumentResult = (doc: any) => (
    <Link
      key={doc.id}
      to={`/projects/${doc.project_id}/documents/${doc.id}`}
      className="block"
    >
      <Card className="p-4 hover:shadow-md transition-shadow">
        <div className="flex items-start space-x-3">
          <FileText className="w-5 h-5 text-blue-600 mt-1 flex-shrink-0" />
          <div className="flex-1 min-w-0">
            <h4 className="font-medium text-gray-900 truncate">{doc.title}</h4>
            <p className="text-sm text-gray-600 mt-1 line-clamp-2">{doc.preview}</p>
            <p className="text-xs text-gray-400 mt-2">
              Updated {new Date(doc.updated_at).toLocaleDateString()}
            </p>
          </div>
          <ChevronRight className="w-5 h-5 text-gray-400 flex-shrink-0" />
        </div>
      </Card>
    </Link>
  );

  const renderChatResult = (chat: any) => (
    <Link
      key={chat.id}
      to={`/chat?thread=${chat.thread_id}`}
      className="block"
    >
      <Card className="p-4 hover:shadow-md transition-shadow">
        <div className="flex items-start space-x-3">
          <MessageSquare className="w-5 h-5 text-green-600 mt-1 flex-shrink-0" />
          <div className="flex-1 min-w-0">
            <p className="text-sm text-gray-600 line-clamp-3">{chat.preview}</p>
            <p className="text-xs text-gray-400 mt-2">
              {new Date(chat.created_at).toLocaleDateString()}
            </p>
          </div>
          <ChevronRight className="w-5 h-5 text-gray-400 flex-shrink-0" />
        </div>
      </Card>
    </Link>
  );

  const renderSemanticResult = (match: any) => (
    <Card key={match.id} className="p-4">
      <div className="flex items-start space-x-3">
        <Brain className="w-5 h-5 text-purple-600 mt-1 flex-shrink-0" />
        <div className="flex-1">
          <p className="text-sm text-gray-600 line-clamp-3">{match.content}</p>
          <div className="flex items-center mt-2 text-xs text-gray-400">
            <span>Relevance: {(match.score * 100).toFixed(0)}%</span>
            {match.metadata?.source && (
              <span className="ml-4">Source: {match.metadata.source}</span>
            )}
          </div>
        </div>
      </div>
    </Card>
  );

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="max-w-5xl mx-auto p-6">
      <div className="mb-8">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => navigate(-1)}
          className="mb-4"
        >
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back
        </Button>

        <h1 className="text-2xl font-bold text-gray-900 mb-2">
          Search Results
        </h1>
        <p className="text-gray-600">
          Showing results for "{query}"
          {projectId && <span className="text-sm"> in current project</span>}
        </p>
      </div>

      {error && (
        <Card className="mb-6 p-4 bg-red-50 border-red-200">
          <p className="text-red-700">{error}</p>
        </Card>
      )}

      {results && (
        <div className="space-y-8">
          {/* Documents Section */}
          {results.documents.length > 0 && (
            <section>
              <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <FileText className="w-5 h-5 mr-2" />
                Documents ({results.documents.length})
              </h2>
              <div className="space-y-3">
                {results.documents.map(renderDocumentResult)}
              </div>
            </section>
          )}

          {/* Chat Messages Section */}
          {results.chats.length > 0 && (
            <section>
              <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <MessageSquare className="w-5 h-5 mr-2" />
                Chat Messages ({results.chats.length})
              </h2>
              <div className="space-y-3">
                {results.chats.map(renderChatResult)}
              </div>
            </section>
          )}

          {/* Semantic Matches Section */}
          {results.semantic_matches.length > 0 && (
            <section>
              <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <Brain className="w-5 h-5 mr-2" />
                Related Content ({results.semantic_matches.length})
              </h2>
              <div className="space-y-3">
                {results.semantic_matches.map(renderSemanticResult)}
              </div>
            </section>
          )}

          {/* No Results */}
          {Object.values(results).every(arr => arr.length === 0) && (
            <Card className="text-center py-12">
              <Search className="w-12 h-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-600">No results found for your search.</p>
              <p className="text-sm text-gray-500 mt-2">
                Try different keywords or search in all projects.
              </p>
            </Card>
          )}
        </div>
      )}
    </div>
  );
}
```

#### **Step 7: Add Search API Service**

**File: `frontend/src/utils/api.ts`** (Add search function; **note actual path differs from guide**)
```typescript
// In frontend/src/utils/api.ts:
export const searchApi = {
  search: async (query: string, projectId?: string): Promise<any> => {
    const params = new URLSearchParams({ q: query });
    if (projectId) params.append('project_id', projectId);
    const response = await api.get(`/search?${params}`);
    return response.data;
  }
};
```

#### **Step 8: Update Main Layout Search**

**File: `frontend/src/components/layout/MainLayout.tsx`** (Update search functionality)
```tsx
// Add these imports
import { useNavigate } from 'react-router-dom';

// Inside MainLayout component
const navigate = useNavigate();
const [searchQuery, setSearchQuery] = useState('');

const handleSearch = (e: React.FormEvent) => {
  e.preventDefault();
  if (searchQuery.trim()) {
    navigate(`/search?q=${encodeURIComponent(searchQuery.trim())}`);
    setSearchQuery('');
  }
};

// Update the search form in the header
<form onSubmit={handleSearch} className="flex-1 max-w-2xl mx-auto">
  <div className="relative">
    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
    <input
      type="text"
      value={searchQuery}
      onChange={(e) => setSearchQuery(e.target.value)}
      placeholder="Search documents, chats, and more..."
      className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg
                 focus:outline-none focus:ring-2 focus:ring-blue-500
                 focus:border-transparent"
    />
  </div>
</form>
```

#### **Step 9: Add Search Route to App**

**File: `frontend/src/App.tsx`** (Add route)
```tsx
// Add import
import SearchResults from './pages/SearchResults';

// Add route in the Routes section
<Route path="/search" element={
  <ProtectedRoute>
    <MainLayout>
      <SearchResults />
    </MainLayout>
  </ProtectedRoute>
} />
```

### **Part 3: Production Deployment**

#### **Step 10: Production Nginx Config**

**File: `nginx/nginx.prod.conf`**
```nginx
events {
    worker_connections 1024;
}

http {
    upstream backend {
        server backend:8000;
    }

    upstream frontend {
        server frontend:3000;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=ws:10m rate=5r/s;

    # Redirect HTTP to HTTPS
    server {
        listen 80;
        server_name _;
        return 301 https://$host$request_uri;
    }

    # HTTPS Server
    server {
        listen 443 ssl http2;
        server_name _;

        # SSL Configuration
        ssl_certificate /etc/nginx/ssl/fullchain.pem;
        ssl_certificate_key /etc/nginx/ssl/privkey.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;
        ssl_prefer_server_ciphers off;
        ssl_session_cache shared:SSL:10m;
        ssl_session_timeout 10m;

        # Security Headers
        add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;
        add_header X-Frame-Options "DENY" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header X-XSS-Protection "1; mode=block" always;
        add_header Referrer-Policy "strict-origin-when-cross-origin" always;

        # API endpoints
        location /api {
            limit_req zone=api burst=20 nodelay;

            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;

            # Timeouts for long-running requests
            proxy_read_timeout 300s;
            proxy_connect_timeout 75s;
        }

        # WebSocket
        location /ws {
            limit_req zone=ws burst=10 nodelay;

            proxy_pass http://backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;

            # WebSocket timeouts
            proxy_read_timeout 3600s;
            proxy_send_timeout 3600s;
        }

        # Frontend
        location / {
            proxy_pass http://frontend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
```

#### **Step 11: Production Docker Compose**

**File: `docker-compose.prod.yml`**
```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped
    networks:
      - app-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER}"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    command: redis-server --requirepass ${REDIS_PASSWORD}
    restart: unless-stopped
    networks:
      - app-network
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    environment:
      DATABASE_URL: postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
      REDIS_URL: redis://:${REDIS_PASSWORD}@redis:6379/0
      SECRET_KEY: ${SECRET_KEY}
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      ENVIRONMENT: production
      CORS_ORIGINS: ${CORS_ORIGINS}
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    restart: unless-stopped
    networks:
      - app-network

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
      args:
        VITE_API_URL: ${FRONTEND_API_URL}
        VITE_WS_URL: ${FRONTEND_WS_URL}
    restart: unless-stopped
    networks:
      - app-network

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.prod.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - backend
      - frontend
    restart: unless-stopped
    networks:
      - app-network

volumes:
  postgres_data:
    driver: local

networks:
  app-network:
    driver: bridge
```

#### **Step 12: Comprehensive README**

**File: `README.md`**
```markdown
# AI-Powered Document & Chat Management System

A production-ready document management and AI chat application with RAG (Retrieval-Augmented Generation) capabilities, built for single-user productivity.

## Features

- 📄 **Document Management**: Upload, version, and organize documents with AI-suggested tags
- 💬 **AI Chat**: Context-aware conversations using your documents via RAG
- 🔍 **Unified Search**: Search across documents, chats, and semantic content
- 🚀 **Real-time Updates**: WebSocket-powered live notifications
- 🔐 **Secure**: JWT authentication, encrypted connections, rate limiting
- 📊 **Project Organization**: Organize work into projects with isolated contexts

## Architecture

### Backend
- **Framework**: FastAPI with async SQLAlchemy
- **Database**: PostgreSQL with full-text search
- **Vector Store**: ChromaDB for semantic search
- **Cache**: Redis for session management
- **AI**: OpenAI GPT integration with streaming

### Frontend
- **Framework**: React 18 with TypeScript
- **State**: Zustand for global state management
- **Styling**: Tailwind CSS
- **Build**: Vite for fast development

## Quick Start

### Prerequisites
- Docker & Docker Compose
- OpenAI API key
- 8GB RAM minimum
- Linux/macOS/WSL2 (for production)

### Development Setup

1. **Clone the repository**
   \`\`\`bash
   git clone <repository-url>
   cd ai-productivity-app
   \`\`\`

2. **Configure environment**
   \`\`\`bash
   cp .env.example .env
   \`\`\`

   Edit `.env` with your settings:
   \`\`\`env
   # Database
   POSTGRES_USER=appuser
   POSTGRES_PASSWORD=securepwd123
   POSTGRES_DB=aiproductivity

   # Redis
   REDIS_PASSWORD=redispwd123

   # Application
   SECRET_KEY=your-secret-key-min-32-chars
   OPENAI_API_KEY=sk-your-openai-key

   # URLs (development)
   FRONTEND_API_URL=http://localhost/api
   FRONTEND_WS_URL=ws://localhost/ws
   \`\`\`

3. **Start services**
   \`\`\`bash
   docker-compose up -d
   \`\`\`

4. **Access the application**
   - Frontend: http://localhost
   - API Docs: http://localhost/api/docs

## Production Deployment

### SSL Certificate Setup

1. **Using Let's Encrypt (recommended)**
   \`\`\`bash
   sudo certbot certonly --standalone -d yourdomain.com
   sudo cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem ./ssl/
   sudo cp /etc/letsencrypt/live/yourdomain.com/privkey.pem ./ssl/
   \`\`\`

2. **Set permissions**
   \`\`\`bash
   sudo chown $USER:$USER ./ssl/*
   chmod 600 ./ssl/privkey.pem
   \`\`\`

### Environment Configuration

Create production `.env.prod`:
\`\`\`env
# Database (use strong passwords!)
POSTGRES_USER=produser
POSTGRES_PASSWORD=$(openssl rand -base64 32)
POSTGRES_DB=aiproductivity_prod

# Redis
REDIS_PASSWORD=$(openssl rand -base64 32)

# Security
SECRET_KEY=$(openssl rand -hex 32)

# OpenAI
OPENAI_API_KEY=sk-your-production-key

# Frontend URLs
FRONTEND_API_URL=https://yourdomain.com/api
FRONTEND_WS_URL=wss://yourdomain.com/ws

# CORS
CORS_ORIGINS=["https://yourdomain.com"]
\`\`\`

### Deploy to Production

\`\`\`bash
# Deploy with production config
docker-compose -f docker-compose.prod.yml --env-file .env.prod up -d

# Check logs
docker-compose -f docker-compose.prod.yml logs -f

# Scale if needed
docker-compose -f docker-compose.prod.yml up -d --scale backend=3
\`\`\`

### Database Encryption

For production, enable encryption at rest:

**PostgreSQL on AWS RDS:**
- Enable encryption when creating the instance
- Use AWS KMS for key management

**Self-hosted PostgreSQL:**
\`\`\`bash
# Install PostgreSQL TDE extension
# Configure in postgresql.conf:
shared_preload_libraries = 'pg_tde'
pg_tde.keyring_type = 'file'
pg_tde.keyring_file_path = '/secure/keys/pg_tde_keyring'
\`\`\`

## API Documentation

Interactive API documentation available at:
- Swagger UI: `/api/docs`
- ReDoc: `/api/redoc`

### Key Endpoints

- `POST /auth/register` - Register (single user only)
- `POST /auth/login` - Login
- `GET /projects` - List projects
- `POST /projects/{id}/documents` - Upload document
- `GET /threads` - List chat threads
- `GET /search` - Unified search

## Maintenance

### Backup

\`\`\`bash
# Database backup
docker-compose exec postgres pg_dump -U $POSTGRES_USER $POSTGRES_DB | \
  gzip > backup_$(date +%Y%m%d_%H%M%S).sql.gz

# Full data backup
docker run --rm -v aiproductivity_postgres_data:/data \
  -v $(pwd):/backup alpine tar czf /backup/data_backup.tar.gz /data
\`\`\`

### Monitoring

- Health check: `GET /api/health`
- Metrics: Monitor via `docker stats`
- Logs: `docker-compose logs -f [service]`

### Updates

\`\`\`bash
# Pull latest changes
git pull origin main

# Rebuild and restart
docker-compose -f docker-compose.prod.yml build
docker-compose -f docker-compose.prod.yml up -d
\`\`\`

## Security Considerations

1. **Environment Variables**
   - Never commit `.env` files
   - Use strong, unique passwords
   - Rotate keys regularly

2. **Network Security**
   - Use firewall rules (allow only 80/443)
   - Enable fail2ban for SSH
   - Regular security updates

3. **Data Protection**
   - Enable database encryption
   - Regular automated backups
   - Test restore procedures

4. **Application Security**
   - Rate limiting enabled
   - CORS properly configured
   - Security headers in Nginx

## Troubleshooting

### Common Issues

**WebSocket Connection Failed**
- Check Nginx upgrade headers
- Verify CORS includes your domain
- Check firewall allows WebSocket

**Database Connection Error**
- Verify PostgreSQL is running: `docker-compose ps`
- Check credentials in `.env`
- Ensure database exists

**Search Not Working**
- Check ChromaDB container logs
- Verify collections are created
- Rebuild vector indices if needed

**High Memory Usage**
- Adjust Docker memory limits
- Reduce ChromaDB cache size
- Scale horizontally instead

## Performance Tuning

### PostgreSQL
\`\`\`sql
-- Adjust in postgresql.conf
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
\`\`\`

### Redis
\`\`\`conf
# In redis.conf
maxmemory 512mb
maxmemory-policy allkeys-lru
\`\`\`

### Application
- Enable response caching
- Optimize vector search parameters
- Use connection pooling

## License

[Your License Here]

## Support

For issues and questions:
- GitHub Issues: [repository-url]/issues
- Documentation: [repository-url]/wiki
```

### **Final Integration Steps**

#### **Step 13: Update Environment Template**

**File: `.env.example`**
```env
# Database
POSTGRES_USER=appuser
POSTGRES_PASSWORD=changeme
POSTGRES_DB=aiproductivity

# Redis
REDIS_PASSWORD=changeme

# Application
SECRET_KEY=your-secret-key-minimum-32-characters
OPENAI_API_KEY=sk-...

# Frontend URLs (development)
FRONTEND_API_URL=http://localhost/api
FRONTEND_WS_URL=ws://localhost/ws

# CORS (production only)
CORS_ORIGINS=["http://localhost:3000"]

# Email (optional)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
```

#### **Step 14: Add Search Activity Logging**

**File: `backend/app/services/search_service.py`** (Add at the end of unified_search)
```python
# Log search activity
from ..services.activity_logger import ActivityLogger
activity_logger = ActivityLogger(self.db)

await activity_logger.log_activity(
    user_id=user_id,
    activity_type="search_performed",
    details={
        "query": query,
        "project_id": project_id,
        "result_counts": {
            "documents": len(results["documents"]),
            "chats": len(results["chats"]),
            "semantic": len(results["semantic_matches"])
        }
    }
)
```

This completes Phase 5 implementation that works with your existing codebase structure, avoiding duplication and maintaining consistency with established patterns.
