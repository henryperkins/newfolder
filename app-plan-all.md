### **FINAL Complete AI Productivity App: UI-Guided Implementation Plan**

This document represents the final, comprehensive implementation plan for the private, single-user AI Productivity App. It is the single source of truth for development, synthesizing all technical requirements, user experience goals from the UI mockup, and key architectural decisions into a cohesive roadmap.

#### **1. Core Principles & Scope**

*   **App Identity:** A private, single-user AI productivity app optimized for simplicity, ease-of-use, and resource efficiency.
*   **Deployment:** Designed for simple, self-hosted deployment via Docker.
*   **User Experience:** The UI should closely mirror the provided mockup, focusing on clarity and intuitive workflows.

##### **Remediated & Out-of-Scope Features**
To maintain focus and align with the core identity, the following features visible in the initial mockup are explicitly **excluded** from this plan:
*   `Subscription Info`
*   `Browser Extensions`
*   `Tips & Announcements`

#### **2. Technology Stack**

*   **Backend:** FastAPI (Python)
*   **Frontend:** React with Vite (TypeScript)
*   **State Management:** Zustand
*   **Structured Database:** PostgreSQL
*   **Vector Database:** ChromaDB
*   **AI & ML:** `openai`, `sentence-transformers`, `PyMuPDF`, `python-docx`
*   **Deployment:** Docker & Docker Compose with an Nginx reverse proxy

#### **3. Architecture Overview**

A decoupled, containerized architecture will be used, with a React frontend communicating via REST and WebSockets to a FastAPI backend. Data is persisted in PostgreSQL (structured) and ChromaDB (vector).

---

### **Phased Implementation Roadmap**

#### **Phase 1: Foundation & Core Layout**

*   **Goal:** Establish the application shell, secure user authentication, and build the static layout elements from the mockup.

*   **User Interface (React)**
    *   Implement the main `AppLayout` with a fixed left `Sidebar` and a main `ContentArea`.
    *   Build the `Sidebar` component, including the `UserProfile` section (top-left), the primary `+ New Chat` button, and the `Workspace` section with navigation links for "Projects" and "Settings".
    *   Create public pages for `Login` and `Registration`.
    *   Create the `Settings` page for profile management (username, email, password).

*   **Services (Backend)**
    *   **Security Service:** Implements `bcrypt` password hashing and JWT creation/decoding.
    *   **Email Service:** Integrates an SMTP client for the password reset workflow.

*   **Routes (FastAPI Endpoints)**
    *   Implement `/auth`, `/users`, and password recovery endpoints.
    *   **Decision:** The `POST /auth/register` endpoint will implement a code-level check. If `User.count() > 0`, it will return a `403 Forbidden` status, programmatically enforcing the single-user constraint.

*   **Business Logic & Data**
    *   Define the `User` SQLAlchemy model.
    *   Implement the `get_current_user` dependency to secure all non-public routes.

#### **Phase 2: Projects & The "Empty State" Dashboard**

*   **Goal:** Implement the project management lifecycle and the visually guided "first-run" experience.

*   **User Interface (React)**
    *   Build the **`EmptyDashboard`** component, which is the default view for new users. It will feature the "Organize Your Work with Projects" card and the primary "Create Your First Project" button.
    *   Build the **`ProjectTemplates`** component, displaying the predefined template cards as seen in the mockup.
    *   Implement the `ProjectCreationModal`, triggered by clicking either the main CTA or a template card.
    *   Create the `Projects` view to list all user-created projects.

*   **Services (Backend)**
    *   **Activity Logging Service:** This is a **user-facing feature service**. It will log key events (project creation, status changes) to populate the visual timeline.
    *   **Template Service:** A service to define and retrieve project template data.

*   **Routes (FastAPI Endpoints)**
    *   Implement full CRUD endpoints for `/projects`.
    *   `GET /project-templates`: An endpoint to fetch the list of available project templates.

*   **Business Logic & Data**
    *   Define `Project`, `Tag`, and junction tables for project tagging.
    *   **Decision:** For the initial version, Project Templates will be **hardcoded** in the `TemplateService`. This prioritizes development speed and can be migrated to a database later if needed.

#### **Phase 3: Core Chat Experience**

*   **Goal:** Activate the primary chat interface, connecting it to projects and enabling basic AI interaction.

*   **User Interface (React)**
    *   Build the main **`ChatView`** component, which includes the message display area, input form, and WebSocket handling.
    *   Make the "Start a new chat" input bar on the `EmptyDashboard` functional.
    *   Build the **`ExamplePrompts`** component ("Summarize," "Code," etc.) to guide the user.
    *   **Decision on UX Flow:** When a user starts a chat from the dashboard:
        1.  A `SelectProjectModal` will appear, listing existing projects and a `+ Create New Project` option.
        2.  Upon selection, the UI will transition to the `ChatView` within the chosen project's context, with the user's initial prompt pre-populated.
    *   Activate the "Recent Chats" link in the sidebar. **Decision:** This will show the most recent chat threads from **all projects**.

*   **Services (Backend)**
    *   Implement the modular **AI Provider Service**, **WebSocket Manager Service**, and **Summarization Service**.

*   **Routes (FastAPI Endpoints)**
    *   Implement all endpoints for chat history, messages (`PUT`/`DELETE`), chat threads, and on-demand summaries (`POST /chat/{id}/summarize`).

*   **Business Logic & Data**
    *   Define `ChatMessage` and `ChatThread` models. Every chat message must belong to a thread, and every thread to a project.

#### **Phase 4: Knowledge Management & RAG Integration**

*   **Goal:** Integrate version-controlled document uploads and the RAG pipeline to make the AI fully context-aware.

*   **User Interface (React)**
    *   Build the `DocumentManager` view within each project for bulk uploads (drag-and-drop).
    *   Add a paperclip/attachment icon to the `ChatView` input for in-line file uploads.
    *   Implement UI for AI-suggested tags on newly uploaded documents.
    *   Build the `VersionHistory` component to view and revert document versions.
    *   **Decision on Indexing Feedback:** A **two-part feedback mechanism** will be used:
        1.  **Contextual Status:** The `DocumentManager` will show a status indicator (`Processing...`, `Indexed`) next to each file.
        2.  **Global Notification:** A `NotificationBell` in the layout will signal when a batch upload is complete.

*   **Services (Backend)**
    *   Implement the **Vector DB Service**, **File Processor Service** (for parsing, embedding, and tag suggestion), and the core **RAG Service**.

*   **Routes (FastAPI Endpoints)**
    *   Implement `POST /projects/{project_id}/documents` for multipart file uploads.

*   **Business Logic & Data**
    *   Implement `Document` and `DocumentVersion` models to ensure all file changes are tracked non-destructively.
    *   **Decision:** The RAG Service will exclusively query the **latest (current) version** of documents by default to maintain simplicity.

#### **Phase 5: Search, Notifications & Finalization**

*   **Goal:** Implement unified search, finalize the notification system, and prepare the application for deployment.

*   **User Interface (React)**
    *   Make the main search bar fully functional, leading to a dedicated `SearchResults` page.
    *   **Decision on Search Ranking:** The `SearchResults` page will present results in **distinct, clearly labeled sections** (e.g., "Top Results from Documents" from semantic search, and "Keyword Matches in Titles" from SQL search) to provide clarity.
    *   Build the `NotificationBell` and its associated dropdown list to display real-time events.

*   **Services (Backend)**
    *   Implement the **Search Service** to orchestrate hybrid search queries.
    *   Implement the **Notification Service** to push events to the frontend via WebSockets.

*   **Routes (FastAPI Endpoints)**
    *   Implement the unified `GET /search` endpoint.

*   **Business Logic & Deployment**
    *   Finalize Dockerfiles and the `docker-compose.yml` file for all services.
    *   Configure Nginx as a reverse proxy with SSL/TLS termination.
    *   Write a comprehensive `README.md` with clear deployment instructions. **Decision:** The README will explicitly instruct users to enable **database-level encryption at rest** when deploying in a production environment.
