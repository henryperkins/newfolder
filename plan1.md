# Phase 1: Foundation & Core Layout - Detailed Implementation Specification

## Phase Summary
Establish the application shell, secure user authentication, and build the static layout elements from the mockup. This phase creates the foundational infrastructure upon which all subsequent features will be built.

## 1. User Interface Components

### 1.1 Core Layout Components

#### **AppLayout Component**
- **Purpose**: Root layout container providing consistent structure across all authenticated views
- **Props**:
  ```typescript
  interface AppLayoutProps {
    children: React.ReactNode;
  }
  ```
- **State Management**:
  - `isSidebarCollapsed: boolean` (managed in Zustand store)
  - `currentUser: User | null` (from auth store)
- **Responsibilities**:
  - Render fixed left sidebar (280px width when expanded, 60px when collapsed)
  - Provide main content area with responsive padding
  - Handle responsive behavior (auto-collapse sidebar on mobile < 768px)
  - Implement keyboard shortcuts (Cmd/Ctrl + B to toggle sidebar)
- **CSS Grid Structure**:
  ```css
  display: grid;
  grid-template-columns: var(--sidebar-width) 1fr;
  min-height: 100vh;
  ```

#### **Sidebar Component**
- **Purpose**: Primary navigation and user context display
- **Props**:
  ```typescript
  interface SidebarProps {
    isCollapsed: boolean;
    onToggleCollapse: () => void;
  }
  ```
- **Sub-components**:
  1. **UserProfile** (top section)
  2. **PrimaryActions** (New Chat button)
  3. **WorkspaceNav** (Projects, Settings links)
  4. **RecentActivity** (placeholder for Phase 3)
- **Behaviors**:
  - Smooth CSS transition on collapse/expand (300ms ease-in-out)
  - Tooltips on hover when collapsed
  - Active route highlighting with left border accent

#### **UserProfile Component**
- **Purpose**: Display current user info and quick actions
- **Props**:
  ```typescript
  interface UserProfileProps {
    user: User;
    isCollapsed: boolean;
  }
  ```
- **Display Elements**:
  - Avatar (initials-based if no profile picture)
  - Username (truncated with ellipsis if > 20 chars)
  - Email (shown only when expanded)
  - Dropdown menu trigger (gear icon)
- **Dropdown Menu Items**:
  - "Account Settings" → navigates to /settings
  - "Sign Out" → calls logout API and redirects

#### **ContentArea Component**
- **Purpose**: Main content wrapper with consistent padding and max-width
- **Props**:
  ```typescript
  interface ContentAreaProps {
    children: React.ReactNode;
    maxWidth?: 'sm' | 'md' | 'lg' | 'xl' | 'full';
    padding?: boolean;
  }
  ```
- **Default Styles**:
  - Max-width: 1200px (xl default)
  - Padding: 2rem (32px) on all sides
  - Background: #FAFAFA or CSS variable --bg-primary

### 1.2 Authentication Components

#### **LoginPage Component**
- **Purpose**: User authentication entry point
- **State**:
  ```typescript
  interface LoginState {
    email: string;
    password: string;
    isLoading: boolean;
    error: string | null;
    showPassword: boolean;
  }
  ```
- **Form Elements**:
  - Email input with validation (regex: valid email format)
  - Password input with visibility toggle
  - "Remember me" checkbox (stores email in localStorage)
  - "Forgot password?" link
  - Submit button with loading state
- **Validation Rules**:
  - Email: Required, valid format
  - Password: Required, min 8 characters
- **Success Flow**: Store JWT in httpOnly cookie, redirect to dashboard
- **Error States**:
  - Invalid credentials: "Email or password is incorrect"
  - Server error: "Unable to sign in. Please try again."
  - Rate limited: "Too many attempts. Please try again in X minutes."

#### **RegistrationPage Component**
- **Purpose**: New user registration (only works if no users exist)
- **State**:
  ```typescript
  interface RegistrationState {
    username: string;
    email: string;
    password: string;
    confirmPassword: string;
    isLoading: boolean;
    error: string | null;
    validationErrors: Record<string, string>;
  }
  ```
- **Special Behavior**:
  - On mount, check if registration is available via `GET /auth/registration-available`
  - If unavailable (403), show message: "This instance already has a registered user. Please contact the administrator."
- **Validation Rules**:
  - Username: Required, 3-30 chars, alphanumeric + underscore only
  - Email: Required, valid format, will be checked for uniqueness
  - Password: Required, min 8 chars, must contain uppercase, lowercase, number
  - Confirm Password: Must match password
- **Password Strength Indicator**: Visual bar showing weak/medium/strong

#### **ForgotPasswordModal Component**
- **Purpose**: Initiate password reset flow
- **Props**:
  ```typescript
  interface ForgotPasswordModalProps {
    isOpen: boolean;
    onClose: () => void;
  }
  ```
- **Flow**:
  1. User enters email
  2. Submit triggers `POST /auth/forgot-password`
  3. Show success message regardless of email existence (security)
  4. Email contains reset link with token valid for 1 hour

#### **ResetPasswordPage Component**
- **Purpose**: Complete password reset with token
- **Route**: `/reset-password?token=xxx`
- **Validation**: Same password rules as registration
- **Error States**:
  - Invalid/expired token: "This reset link is invalid or has expired."
  - Password mismatch: "Passwords do not match"

### 1.3 Settings Components

#### **SettingsPage Component**
- **Purpose**: User profile and app configuration management
- **Layout**: Two-column with sidebar navigation
- **Sections**:
  1. **Profile Settings**
     - Username (editable)
     - Email (editable with re-authentication)
     - Avatar upload (Phase 1.5 enhancement)
  2. **Security Settings**
     - Change password (requires current password)
     - Active sessions list (future enhancement)
  3. **Preferences** (placeholder for future)

#### **ChangePasswordForm Component**
- **State**:
  ```typescript
  interface ChangePasswordState {
    currentPassword: string;
    newPassword: string;
    confirmPassword: string;
    isLoading: boolean;
    success: boolean;
    error: string | null;
  }
  ```
- **Validation**: Current password verified server-side
- **Success Behavior**: Show toast notification, clear form

## 2. Backend Services

### 2.1 Security Service
```python
# services/security.py
class SecurityService:
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt with automatic salt generation"""
        return self.pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return self.pwd_context.verify(plain_password, hashed_password)

    def create_access_token(self, data: dict, expires_delta: timedelta = None) -> str:
        """Create JWT token with expiration"""
        to_encode = data.copy()
        expire = datetime.utcnow() + (expires_delta or timedelta(hours=24))
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)

    def decode_token(self, token: str) -> dict:
        """Decode and validate JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except JWTError:
            raise ValueError("Invalid token")

    def create_password_reset_token(self, email: str) -> str:
        """Create a password reset token valid for 1 hour"""
        data = {"email": email, "type": "password_reset"}
        return self.create_access_token(data, timedelta(hours=1))
```

### 2.2 Email Service
```python
# services/email.py
class EmailService:
    def __init__(self, smtp_host: str, smtp_port: int, username: str, password: str):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.sender_email = username
        self.app_name = "AI Productivity App"

    async def send_password_reset_email(self, to_email: str, reset_token: str, base_url: str):
        """Send password reset email with secure token"""
        reset_link = f"{base_url}/reset-password?token={reset_token}"

        subject = f"{self.app_name} - Password Reset Request"
        body = f"""
        <html>
            <body>
                <h2>Password Reset Request</h2>
                <p>You requested a password reset for your {self.app_name} account.</p>
                <p>Click the link below to reset your password:</p>
                <p><a href="{reset_link}">Reset Password</a></p>
                <p>This link will expire in 1 hour.</p>
                <p>If you didn't request this, please ignore this email.</p>
            </body>
        </html>
        """

        await self._send_email(to_email, subject, body)

    async def _send_email(self, to_email: str, subject: str, html_body: str):
        """Internal method to send emails via SMTP"""
        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = f"{self.app_name} <{self.sender_email}>"
        message["To"] = to_email

        html_part = MIMEText(html_body, "html")
        message.attach(html_part)

        async with aiosmtplib.SMTP(hostname=self.smtp_host, port=self.smtp_port) as server:
            await server.starttls()
            await server.login(self.username, self.password)
            await server.send_message(message)
```

## 3. Database Models

### 3.1 User Model
```python
# models/user.py
class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(30), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    last_login_at = Column(DateTime(timezone=True), nullable=True)

    # Constraints
    __table_args__ = (
        CheckConstraint("char_length(username) >= 3", name="username_min_length"),
        CheckConstraint("email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}$'", name="valid_email"),
    )
```

### 3.2 Password Reset Token Model
```python
# models/password_reset.py
class PasswordResetToken(Base):
    __tablename__ = "password_reset_tokens"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    token = Column(String(255), unique=True, nullable=False, index=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    used = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    user = relationship("User", backref="password_reset_tokens")

    # Constraints
    __table_args__ = (
        CheckConstraint("expires_at > created_at", name="valid_expiration"),
    )
```

## 4. API Routes and Specifications

### 4.1 Authentication Routes

#### **POST /auth/register**
- **Purpose**: Create the first and only user account
- **Request Body**:
  ```json
  {
    "username": "john_doe",
    "email": "john@example.com",
    "password": "SecurePass123!"
  }
  ```
- **Success Response** (201):
  ```json
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "username": "john_doe",
    "email": "john@example.com",
    "created_at": "2024-01-15T10:30:00Z"
  }
  ```
- **Error Responses**:
  - 403: `{"detail": "Registration is closed. This instance already has a user."}`
  - 422: `{"detail": [{"loc": ["body", "email"], "msg": "Invalid email format"}]}`
- **Implementation Note**: Check `SELECT COUNT(*) FROM users` before allowing registration

#### **POST /auth/login**
- **Purpose**: Authenticate user and create session
- **Request Body**:
  ```json
  {
    "email": "john@example.com",
    "password": "SecurePass123!"
  }
  ```
- **Success Response** (200):
  ```json
  {
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
    "token_type": "bearer",
    "user": {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "username": "john_doe",
      "email": "john@example.com"
    }
  }
  ```
- **Response Headers**: Set httpOnly cookie with JWT token
- **Error Responses**:
  - 401: `{"detail": "Invalid email or password"}`
  - 429: `{"detail": "Too many login attempts. Try again in 5 minutes."}`

#### **POST /auth/logout**
- **Purpose**: Invalidate current session
- **Authorization**: Bearer token required
- **Success Response** (200): `{"message": "Successfully logged out"}`
- **Side Effects**: Clear httpOnly cookie, optionally blacklist token

#### **GET /auth/registration-available**
- **Purpose**: Check if registration is open (no users exist)
- **Success Response** (200): `{"available": true}`
- **Error Response** (403): `{"available": false, "message": "Registration closed"}`

#### **POST /auth/forgot-password**
- **Purpose**: Initiate password reset flow
- **Request Body**:
  ```json
  {
    "email": "john@example.com"
  }
  ```
- **Success Response** (200): `{"message": "If the email exists, a reset link has been sent"}`
- **Security Note**: Always return 200 to prevent email enumeration

#### **POST /auth/reset-password**
- **Purpose**: Complete password reset with token
- **Request Body**:
  ```json
  {
    "token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
    "new_password": "NewSecurePass456!"
  }
  ```
- **Success Response** (200): `{"message": "Password successfully reset"}`
- **Error Responses**:
  - 400: `{"detail": "Invalid or expired reset token"}`
  - 422: `{"detail": "Password does not meet requirements"}`

### 4.2 User Routes

#### **GET /users/me**
- **Purpose**: Get current user profile
- **Authorization**: Bearer token required
- **Success Response** (200):
  ```json
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "username": "john_doe",
    "email": "john@example.com",
    "created_at": "2024-01-15T10:30:00Z",
    "last_login_at": "2024-01-20T15:45:00Z"
  }
  ```

#### **PATCH /users/me**
- **Purpose**: Update user profile
- **Authorization**: Bearer token required
- **Request Body** (partial update):
  ```json
  {
    "username": "john_doe_updated",
    "email": "newemail@example.com"
  }
  ```
- **Validation**: Email change requires password confirmation
- **Success Response** (200): Updated user object

#### **POST /users/me/change-password**
- **Purpose**: Change password for authenticated user
- **Authorization**: Bearer token required
- **Request Body**:
  ```json
  {
    "current_password": "OldPass123!",
    "new_password": "NewPass456!"
  }
  ```
- **Success Response** (200): `{"message": "Password successfully changed"}`
- **Error Response** (401): `{"detail": "Current password is incorrect"}`

## 5. Authentication Flow & Security

### 5.1 JWT Token Strategy
- **Token Lifetime**: 24 hours for access tokens
- **Storage**: httpOnly, secure, sameSite cookies
- **Refresh Strategy**: Silent refresh before expiration
- **Claims Structure**:
  ```json
  {
    "sub": "550e8400-e29b-41d4-a716-446655440000",
    "email": "john@example.com",
    "exp": 1705584000,
    "iat": 1705497600
  }
  ```

### 5.2 Security Middleware
```python
# dependencies/auth.py
async def get_current_user(
    request: Request,
    db: Session = Depends(get_db),
    security_service: SecurityService = Depends(get_security_service)
) -> User:
    """Extract and validate user from JWT token"""
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        payload = security_service.decode_token(token)
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.query(User).filter(User.id == user_id).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")

    return user
```

## 6. UI/UX Edge Cases

### 6.1 Loading States
- **Login/Register Forms**: Disable all inputs and show spinner in submit button
- **Page Transitions**: Show top-loading bar (like YouTube)
- **Data Fetching**: Skeleton screens for user profile, settings sections

### 6.2 Error States
- **Network Errors**: Toast notification with retry action
- **Form Validation**: Inline error messages below fields
- **404 Pages**: Custom "Page not found" with navigation back to dashboard
- **Session Expiry**: Modal prompt to re-login without losing current page context

### 6.3 Empty States
- **First Login**: Redirect to empty dashboard (Phase 2)
- **No User Photo**: Show initials-based avatar with random background color

### 6.4 Responsive Behaviors
- **Mobile (<768px)**:
  - Sidebar starts collapsed
  - Login/register forms go full-width
  - Settings page becomes single column
- **Tablet (768-1024px)**:
  - Sidebar collapsible via hamburger menu
  - Two-column layouts maintain structure
- **Accessibility**:
  - All interactive elements keyboard navigable
  - ARIA labels on icon-only buttons
  - Focus management on modal open/close

## 7. Technology Decisions

### 7.1 Frontend Libraries
- **Form Management**: React Hook Form (lightweight, good TS support)
- **Validation**: Zod (schema validation with TS inference)
- **HTTP Client**: Axios with request/response interceptors
- **CSS Framework**: Tailwind CSS with custom design tokens
- **Icons**: Lucide React (consistent, tree-shakeable)
- **Date Handling**: date-fns (modular, tree-shakeable)

### 7.2 Backend Libraries
- **CORS**: fastapi-cors with credentials support
- **Rate Limiting**: slowapi (FastAPI-compatible)
- **Background Tasks**: FastAPI BackgroundTasks for email sending
- **Environment Config**: pydantic-settings for type-safe config

### 7.3 Development Tools
- **API Documentation**: FastAPI's built-in Swagger UI
- **Type Checking**: mypy for Python, built-in TS for frontend
- **Code Formatting**: Black (Python), Prettier (JS/TS)
- **Pre-commit Hooks**: Husky + lint-staged

## 8. Testing Strategy

### 8.1 Unit Tests

#### Frontend (Vitest + React Testing Library)
```typescript
// __tests__/components/LoginPage.test.tsx
describe('LoginPage', () => {
  it('validates email format before submission', async () => {
    render(<LoginPage />);
    const emailInput = screen.getByLabelText(/email/i);
    const submitButton = screen.getByRole('button', { name: /sign in/i });

    await userEvent.type(emailInput, 'invalid-email');
    await userEvent.click(submitButton);

    expect(screen.getByText(/invalid email format/i)).toBeInTheDocument();
    expect(mockLogin).not.toHaveBeenCalled();
  });

  it('shows loading state during authentication', async () => {
    mockLogin.mockImplementation(() => new Promise(resolve => setTimeout(resolve, 100)));
    render(<LoginPage />);

    // Fill valid form
    await userEvent.type(screen.getByLabelText(/email/i), 'test@example.com');
    await userEvent.type(screen.getByLabelText(/password/i), 'password123');
    await userEvent.click(screen.getByRole('button', { name: /sign in/i }));

    expect(screen.getByRole('button', { name: /signing in/i })).toBeDisabled();
  });
});
```

#### Backend (Pytest)
```python
# tests/test_auth.py
def test_registration_closed_when_user_exists(client, test_user):
    """Test that registration returns 403 when a user already exists"""
    response = client.post("/auth/register", json={
        "username": "newuser",
        "email": "new@example.com",
        "password": "NewPass123!"
    })

    assert response.status_code == 403
    assert "already has a user" in response.json()["detail"]

def test_password_hashing(security_service):
    """Test that passwords are properly hashed and verified"""
    plain_password = "TestPass123!"
    hashed = security_service.hash_password(plain_password)

    assert hashed != plain_password
    assert security_service.verify_password(plain_password, hashed)
    assert not security_service.verify_password("WrongPass", hashed)
```

### 8.2 Integration Tests

#### API Integration
```python
# tests/integration/test_auth_flow.py
async def test_complete_auth_flow(async_client, db_session):
    """Test complete registration → login → profile access flow"""
    # Register
    register_response = await async_client.post("/auth/register", json={
        "username": "testuser",
        "email": "test@example.com",
        "password": "SecurePass123!"
    })
    assert register_response.status_code == 201

    # Login
    login_response = await async_client.post("/auth/login", json={
        "email": "test@example.com",
        "password": "SecurePass123!"
    })
    assert login_response.status_code == 200
    assert "access_token" in login_response.json()

    # Access protected route
    token = login_response.json()["access_token"]
    profile_response = await async_client.get(
        "/users/me",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert profile_response.status_code == 200
    assert profile_response.json()["email"] == "test@example.com"
```

### 8.3 E2E Tests (Playwright)
```typescript
// e2e/auth.spec.ts
test.describe('Authentication Flow', () => {
  test('new user can register and access dashboard', async ({ page }) => {
    // Navigate to registration
    await page.goto('/register');

    // Fill registration form
    await page.fill('[name="username"]', 'e2e_testuser');
    await page.fill('[name="email"]', 'e2e@example.com');
    await page.fill('[name="password"]', 'E2ETestPass123!');
    await page.fill('[name="confirmPassword"]', 'E2ETestPass123!');

    // Submit and wait for redirect
    await page.click('button[type="submit"]');
    await page.waitForURL('/dashboard');

    // Verify user is logged in
    await expect(page.locator('[data-testid="user-profile"]')).toContainText('e2e_testuser');
  });

  test('prevents multiple user registration', async ({ page }) => {
    // Assuming a user already exists from previous test
    await page.goto('/register');

    // Should show "registration closed" message
    await expect(page.locator('[data-testid="registration-closed"]')).toBeVisible();
  });
});
```

## 9. Phase 1 Milestones & Acceptance Criteria

### Milestone 1: Backend Authentication Foundation
**Acceptance Criteria:**
- [x] User model created with all constraints
- [x] Security service implements password hashing and JWT creation
- [x] All auth endpoints return correct status codes and payloads
- [x] Single-user restriction enforced at registration endpoint
- [x] Rate limiting applied to login endpoint (5 attempts per minute)
- [x] 90% test coverage on auth routes and services

### Milestone 2: Frontend Shell and Routing
**Acceptance Criteria:**
- [x] AppLayout renders with collapsible sidebar
- [x] Public routes (login, register, forgot-password) accessible without auth
- [x] Protected routes redirect to login when unauthenticated
- [x] User profile displays in sidebar with dropdown menu
- [x] Responsive behavior works on mobile/tablet/desktop
- [x] Navigation highlights active route

### Milestone 3: Complete Auth UI Flow
**Acceptance Criteria:**
- [x] User can register (if no users exist)
- [x] User can login with email/password
- [x] Password reset email sends and link works
- [x] Form validation shows inline errors
- [x] Loading states display during async operations
- [x] Success/error toasts appear for all actions
- [x] Session persists across page refreshes

### Milestone 4: Settings and Profile Management
**Acceptance Criteria:**
- [x] Settings page accessible from user dropdown
- [x] User can update username
- [x] User can change password (with current password verification)
- [x] Email change requires password confirmation
- [x] All changes show success confirmation
- [x] Form states reset after successful submission

### Milestone 5: Production Readiness
**Acceptance Criteria:**
- [x] Environment variables configured for all secrets
- [x] CORS properly configured for production domain
- [x] SSL/TLS redirect enforced in production
- [x] Security headers (CSP, HSTS) configured
- [x] Comprehensive README with deployment instructions
- [x] All E2E tests pass in CI/CD pipeline

## 10. Development Execution Order

1. **Database Setup**
   - Create User and PasswordResetToken models
   - Write and run Alembic migrations
   - Seed database with test user (dev only)

2. **Backend Services**
   - Implement SecurityService with tests
   - Implement EmailService with mock SMTP for tests
   - Create authentication dependencies

3. **API Routes**
   - Implement all /auth/* endpoints
   - Implement /users/* endpoints
   - Add OpenAPI documentation

4. **Frontend Foundation**
   - Set up React + Vite + TypeScript
   - Configure Zustand stores (auth, ui)
   - Create routing structure

5. **UI Components (Bottom-up)**
   - Build atomic components (buttons, inputs, cards)
   - Build authentication forms
   - Build layout components
   - Assemble into pages

6. **Integration**
   - Connect frontend to backend
   - Implement error handling
   - Add loading states
   - Test full flows

7. **Polish**
   - Optimize bundle size
   - Add comprehensive error boundaries
   - Implement telemetry/logging
   - Write deployment documentation

This comprehensive specification for Phase 1 provides the solid foundation needed for all subsequent phases of the AI Productivity App.
