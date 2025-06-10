# AI Productivity App - Phase 1

A private, single-user AI productivity application with secure authentication, responsive UI, and foundation for future AI-powered features.

## Features Implemented (Phase 1)

### Authentication & Security
- ✅ User registration (single-user constraint enforced)
- ✅ Secure login with JWT tokens stored in httpOnly cookies
- ✅ Password reset functionality via email
- ✅ Profile management and password change
- ✅ Rate limiting on authentication endpoints
- ✅ Input validation and security headers

### User Interface
- ✅ Responsive layout with collapsible sidebar
- ✅ Clean, modern design using Tailwind CSS
- ✅ Mobile-friendly responsive behavior
- ✅ Keyboard shortcuts (Cmd/Ctrl + B to toggle sidebar)
- ✅ Loading states and error handling
- ✅ Form validation with real-time feedback

### Technical Foundation
- ✅ FastAPI backend with PostgreSQL database
- ✅ React frontend with TypeScript and Vite
- ✅ Zustand for state management
- ✅ Database migrations with Alembic
- ✅ Docker containerization
- ✅ RESTful API design

## Technology Stack

### Backend
- **FastAPI** - Modern Python web framework
- **PostgreSQL** - Reliable SQL database
- **SQLAlchemy** - Database ORM
- **Alembic** - Database migrations
- **Pydantic** - Data validation
- **bcrypt** - Password hashing
- **JWT** - Secure authentication tokens

### Frontend
- **React 18** - UI library
- **TypeScript** - Type safety
- **Vite** - Fast build tool
- **Tailwind CSS** - Utility-first styling
- **Zustand** - Lightweight state management
- **React Hook Form** - Form handling
- **Zod** - Schema validation
- **Axios** - HTTP client

### Infrastructure
- **Docker & Docker Compose** - Containerization
- **Nginx** - Reverse proxy and static file serving
- **PostgreSQL** - Database

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Git

### 1. Clone and Setup
```bash
git clone <repository-url>
cd ai-productivity-app
```

### 2. Environment Configuration
Copy the example environment file and configure your settings:
```bash
cp backend/.env.example backend/.env
```

Edit `backend/.env` with your configuration:
```env
SECRET_KEY=your-super-secret-key-change-this-in-production
DATABASE_URL=postgresql://postgres:postgres@db:5432/ai_productivity_app
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
```

### 3. Launch the Application
```bash
docker-compose up -d
```

The application will be available at:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

### 4. Create Your Account
1. Navigate to http://localhost:3000
2. Click "Sign up" to create the first (and only) user account
3. Complete the registration form
4. Sign in with your credentials

## Development

### Backend Development
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend Development
```bash
cd frontend
npm install
npm run dev
```

### Database Migrations
```bash
cd backend
alembic upgrade head
```

To create a new migration:
```bash
alembic revision --autogenerate -m "Description of changes"
```

## Project Structure

```
ai-productivity-app/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── core/           # Core configuration and database
│   │   ├── models/         # SQLAlchemy models
│   │   ├── routes/         # API endpoints
│   │   ├── services/       # Business logic
│   │   └── dependencies/   # FastAPI dependencies
│   ├── alembic/            # Database migrations
│   └── requirements.txt
├── frontend/               # React frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── pages/          # Page components
│   │   ├── stores/         # Zustand stores
│   │   ├── types/          # TypeScript types
│   │   └── utils/          # Utility functions
│   └── package.json
└── docker-compose.yml      # Container orchestration
```

## Security Considerations

- **Single-User Enforcement**: Registration is automatically disabled after the first user is created
- **Password Security**: Passwords are hashed using bcrypt with automatic salt generation
- **JWT Security**: Tokens are stored in httpOnly cookies with secure flags
- **Rate Limiting**: Login attempts are rate-limited to prevent brute force attacks
- **Input Validation**: All inputs are validated both client and server-side
- **HTTPS Ready**: Application is configured for SSL/TLS in production

## Next Steps (Phase 2)

The next phase will implement:
- Project management system
- Empty state dashboard with project templates
- Activity logging and timeline
- Project creation and organization workflows

## Contributing

1. Follow the existing code style and patterns
2. Add appropriate tests for new features
3. Update documentation as needed
4. Ensure all linting and type checks pass

## License

This project is private and proprietary.