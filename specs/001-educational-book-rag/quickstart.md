# Quickstart Guide: Educational Book with Integrated RAG Chatbot

## Prerequisites

- Node.js 18+ with npm
- Python 3.11+
- Git
- Access to OpenAI API
- Access to Neon Serverless Postgres

## Setting up the Development Environment

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Install Frontend Dependencies
```bash
# Navigate to the project root (where package.json is located)
npm install
```

### 3. Install Backend Dependencies
```bash
# Create a Python virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install Python dependencies
pip install -r backend/requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root with the following variables:

```env
OPENAI_API_KEY=your_openai_api_key_here
NEON_DB_URL=your_neon_postgres_connection_string
SECRET_KEY=your_secret_key_for_session_management
```

### 5. Initialize the Database

Set up the database schema for storing vector embeddings and chat history:

```bash
# Make sure your virtual environment is activated
# Run database initialization script
python backend/scripts/init_db.py
```

## Running the Application

### 1. Start the Backend API Server
```bash
# Make sure your virtual environment is activated
cd backend
python -m uvicorn src.main:app --reload --port 8000
```

### 2. Start the Docusaurus Frontend
In a new terminal (with the virtual environment deactivated):
```bash
npm start
```

The application will be available at `http://localhost:3000`, with the backend API running at `http://localhost:8000`.

## Project Structure Overview

```
project-root/
├── backend/                 # FastAPI backend for RAG chatbot
│   ├── src/
│   │   ├── main.py         # Application entrypoint
│   │   ├── models/         # Pydantic models
│   │   ├── services/       # Business logic
│   │   └── routes/         # API route definitions
│   └── requirements.txt
├── docs/                   # Educational content (modules & chapters)
│   ├── module-1-the-robotic-nervous-system/
│   ├── module-2-the-digital-twin/
│   ├── module-3-the-ai-robot-brain/
│   └── module-4-vision-language-action-systems/
├── src/                    # Docusaurus React components
├── static/                 # Static assets (images)
├── docusaurus.config.js    # Docusaurus configuration
└── sidebars.ts             # Navigation sidebar configuration
```

## Adding Educational Content

### Creating a New Chapter

1. Add your chapter markdown file to the appropriate module directory:
   ```
   docs/module-{n}-{module-name}/your-chapter-name.md
   ```

2. Add chapter content using Docusaurus markdown format:
   ```markdown
   ---
   title: Your Chapter Title
   sidebar_position: 1
   ---

   # Your Chapter Title

   Your educational content here...
   ```

3. Update `sidebars.ts` to include your new chapter in the navigation.

### Adding Images to Chapters

1. Place your image in `static/img/` (or a subdirectory like `static/img/book/`)

2. Reference the image in your markdown:
   ```markdown
   ![Image Alt Text](/img/your-image-name.jpg)
   ```

## Working with the RAG Chatbot

### Ingesting Content for RAG

To make educational content available to the chatbot, it needs to be converted to vector embeddings:

```bash
# Run the content ingestion script
python backend/scripts/ingest_content.py
```

This script will:
1. Read all markdown files in the `docs/` directory
2. Split content into appropriate chunks
3. Generate embeddings using OpenAI's embedding API
4. Store embeddings in the database with references to source content

### Testing the Chatbot API

Once the backend is running, you can test the chat API:

```bash
curl -X POST http://localhost:8000/api/chat/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the difference between forward and inverse kinematics?",
    "sessionId": "test-session-123"
  }'
```

## Deployment

### Building for Production

1. Build the Docusaurus frontend:
   ```bash
   npm run build
   ```

2. The static site will be generated in the `build/` directory, ready for deployment to GitHub Pages or other static hosting.

3. Deploy the backend API to your preferred Python hosting platform (Heroku, AWS, etc.)

### GitHub Pages Deployment

The project is configured for GitHub Pages deployment. To deploy:

1. Ensure your `docusaurus.config.js` has the correct `baseUrl` and `organizationName`/`projectName` settings
2. Run the deployment script:
   ```bash
   npm run deploy
   ```

This will build the site and push the static files to the `gh-pages` branch.

## Troubleshooting

### Common Issues

1. **Port already in use**: If you get a port in use error, try changing the port in the start commands:
   ```bash
   # Backend
   python -m uvicorn src.main:app --reload --port 8001
   
   # Frontend
   npm start -- --port 3001
   ```

2. **Python module not found**: Ensure your virtual environment is activated before running Python commands.

3. **OpenAI API errors**: Verify that your `OPENAI_API_KEY` is correctly set in the environment variables.

### Development Commands

- `npm start`: Start Docusaurus development server
- `npm run build`: Build static site for production
- `npm run serve`: Serve the built site locally
- `npm run deploy`: Deploy to GitHub Pages

For backend development:
- `python -m uvicorn src.main:app --reload`: Start backend with auto-reload
- `pytest`: Run backend tests
- `python -m src.scripts.ingest_content`: Ingest content into RAG system