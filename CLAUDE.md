# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Backend (FastAPI)
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend (React + Vite)
```bash
cd frontend
npm install
npm run dev
npm run build
npm run lint
npm test
```

### Database Migrations
```bash
cd backend
alembic upgrade head
alembic revision --autogenerate -m "Description of changes"
```

### Docker Development
```bash
docker-compose up -d
docker-compose down
```

## Architecture Overview

This is a single-user AI productivity application split into backend and frontend services:

### Backend Architecture
- **FastAPI** application with SQLAlchemy ORM and PostgreSQL
- **Authentication**: JWT tokens stored in httpOnly cookies, single-user constraint enforced
- **Structure**: 
  - `app/core/` - Configuration and database setup
  - `app/models/` - SQLAlchemy models (User, PasswordReset)
  - `app/routes/` - API endpoints (auth, users)
  - `app/services/` - Business logic (email, security)
  - `app/dependencies/` - FastAPI dependencies for auth
- **Rate limiting** using slowapi on auth endpoints
- **Email service** for password reset functionality

### Frontend Architecture
- **React 18** with TypeScript and Vite build system
- **State Management**: Zustand with persistence for auth state
- **Routing**: React Router with protected routes
- **Styling**: Tailwind CSS with component structure:
  - `components/auth/` - Authentication forms and modals
  - `components/common/` - Reusable UI components (Button, Card, Input)
  - `components/layout/` - App layout components (Sidebar, UserProfile)
  - `pages/` - Page components (Dashboard, Projects, Chats, Settings)
- **Form handling**: React Hook Form with Zod validation
- **API client**: Axios with auth interceptors in `utils/api.ts`

### Key Integration Points
- Frontend auth store (`stores/authStore.ts`) manages user state with Zustand persistence
- API communication uses cookies for JWT tokens (no localStorage)
- Protected routes check authentication status before rendering
- Single-user constraint enforced at backend registration endpoint

### Environment Setup
- Backend requires `.env` file with DATABASE_URL, SECRET_KEY, SMTP_* settings
- Frontend connects to backend at localhost:8000 in development
- Docker Compose orchestrates PostgreSQL, backend, and frontend services

## Testing
- Backend: pytest with async support (`pytest`, `pytest-asyncio`)
- Frontend: Vitest with Testing Library (`npm test`, `npm run test:ui`)