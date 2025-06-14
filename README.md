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
- **Vector Store**: ChromaDB for semantic search (currently disabled)
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
   ```bash
   git clone <repository-url>
   cd ai-productivity-app
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   ```

   Edit `.env` with your settings:
   ```env
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
   ```

3. **Start services**
   ```bash
   docker-compose up -d
   ```

4. **Access the application**
   - Frontend: http://localhost
   - API Docs: http://localhost/api/docs

## Production Deployment

### SSL Certificate Setup

1. **Using Let's Encrypt (recommended)**
   ```bash
   sudo certbot certonly --standalone -d yourdomain.com
   sudo cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem ./ssl/
   sudo cp /etc/letsencrypt/live/yourdomain.com/privkey.pem ./ssl/
   ```

2. **Set permissions**
   ```bash
   sudo chown $USER:$USER ./ssl/*
   chmod 600 ./ssl/privkey.pem
   ```

### Environment Configuration

Create production `.env.prod`:
```env
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
```

### Deploy to Production

```bash
# Deploy with production config
docker-compose -f docker-compose.prod.yml --env-file .env.prod up -d

# Check logs
docker-compose -f docker-compose.prod.yml logs -f

# Scale if needed
docker-compose -f docker-compose.prod.yml up -d --scale backend=3
```

### Database Encryption

For production, enable encryption at rest:

**PostgreSQL on AWS RDS:**
- Enable encryption when creating the instance
- Use AWS KMS for key management

**Self-hosted PostgreSQL:**
```bash
# Install PostgreSQL TDE extension
# Configure in postgresql.conf:
shared_preload_libraries = 'pg_tde'
pg_tde.keyring_type = 'file'
pg_tde.keyring_file_path = '/secure/keys/pg_tde_keyring'
```

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

```bash
# Database backup
docker-compose exec postgres pg_dump -U $POSTGRES_USER $POSTGRES_DB | \
  gzip > backup_$(date +%Y%m%d_%H%M%S).sql.gz

# Full data backup
docker run --rm -v aiproductivity_postgres_data:/data \
  -v $(pwd):/backup alpine tar czf /backup/data_backup.tar.gz /data
```

### Monitoring

- Health check: `GET /api/health`
- Metrics: Monitor via `docker stats`
- Logs: `docker-compose logs -f [service]`

### Updates

```bash
# Pull latest changes
git pull origin main

# Rebuild and restart
docker-compose -f docker-compose.prod.yml build
docker-compose -f docker-compose.prod.yml up -d
```

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
- Check ChromaDB container logs (currently disabled)
- Verify collections are created
- Rebuild vector indices if needed

**High Memory Usage**
- Adjust Docker memory limits
- Reduce ChromaDB cache size
- Scale horizontally instead

## Performance Tuning

### PostgreSQL
```sql
-- Adjust in postgresql.conf
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
```

### Redis
```conf
# In redis.conf
maxmemory 512mb
maxmemory-policy allkeys-lru
```

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
