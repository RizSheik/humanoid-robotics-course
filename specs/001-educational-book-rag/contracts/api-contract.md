# API Contract: Educational Book with Integrated RAG Chatbot

## Content API

### Get All Modules
```
GET /api/modules
```
**Description**: Retrieves a list of all educational modules in the book

**Request**:
- Headers: None required
- Query Parameters: None

**Response**:
```
Status: 200 OK
Content-Type: application/json

{
  "modules": [
    {
      "id": "uuid",
      "title": "string",
      "description": "string",
      "order": "integer",
      "chapters": [
        {
          "id": "uuid",
          "title": "string",
          "order": "integer"
        }
      ]
    }
  ]
}
```

### Get Module by ID
```
GET /api/modules/{moduleId}
```
**Description**: Retrieves details of a specific module and its chapters

**Request**:
- Headers: None required
- Path Parameters: 
  - moduleId: UUID of the module

**Response**:
```
Status: 200 OK
Content-Type: application/json

{
  "id": "uuid",
  "title": "string",
  "description": "string",
  "order": "integer",
  "chapters": [
    {
      "id": "uuid",
      "title": "string",
      "order": "integer"
    }
  ]
}
```

### Get Chapter Content
```
GET /api/chapters/{chapterId}
```
**Description**: Retrieves the content of a specific chapter

**Request**:
- Headers: None required
- Path Parameters:
  - chapterId: UUID of the chapter

**Response**:
```
Status: 200 OK
Content-Type: application/json

{
  "id": "uuid",
  "moduleId": "uuid",
  "title": "string",
  "content": "string (Markdown format)",
  "order": "integer",
  "learningObjectives": ["string"],
  "prerequisites": ["string"],
  "learningContent": [
    {
      "id": "uuid",
      "type": "enum ['text', 'image', 'diagram', 'video', 'code', 'exercise', 'quiz']",
      "content": "string",
      "altText": "string (optional)",
      "caption": "string (optional)",
      "order": "integer"
    }
  ],
  "images": [
    {
      "id": "uuid",
      "filename": "string",
      "path": "string",
      "altText": "string",
      "caption": "string"
    }
  ]
}
```

## RAG Chatbot API

### Submit Query to RAG Chatbot
```
POST /api/chat/query
```
**Description**: Submits a query to the RAG chatbot and receives a contextual response based on educational content

**Request**:
- Headers: 
  - Content-Type: application/json
- Body:
```
{
  "query": "string (the user's question)",
  "sessionId": "string (user session identifier)"
}
```

**Response**:
```
Status: 200 OK
Content-Type: application/json

{
  "id": "uuid (of the chat query record)",
  "query": "string (echoed input query)",
  "response": "string (AI-generated response)",
  "sources": [
    {
      "chapterId": "uuid",
      "chapterTitle": "string",
      "moduleId": "uuid",
      "moduleTitle": "string",
      "contentSnippet": "string (relevant text snippet)",
      "relevanceScore": "float (0-1)"
    }
  ],
  "timestamp": "ISO 8601 datetime"
}
```

### Get Chat History
```
GET /api/chat/history
```
**Description**: Retrieves chat history for a specific session

**Request**:
- Headers: None required
- Query Parameters:
  - sessionId: string (user session identifier)

**Response**:
```
Status: 200 OK
Content-Type: application/json

{
  "sessionId": "string",
  "queries": [
    {
      "id": "uuid",
      "query": "string",
      "response": "string",
      "timestamp": "ISO 8601 datetime",
      "sources": [
        {
          "chapterId": "uuid",
          "chapterTitle": "string",
          "moduleId": "uuid",
          "moduleTitle": "string",
          "relevanceScore": "float (0-1)"
        }
      ]
    }
  ]
}
```

## Search API

### Search Educational Content
```
GET /api/search
```
**Description**: Performs semantic search across educational content

**Request**:
- Headers: None required
- Query Parameters:
  - q: string (search query)
  - limit: integer (optional, max results, default: 10)

**Response**:
```
Status: 200 OK
Content-Type: application/json

{
  "query": "string",
  "results": [
    {
      "id": "uuid",
      "type": "enum ['module', 'chapter', 'section']",
      "title": "string",
      "contentPreview": "string",
      "moduleId": "uuid",
      "moduleName": "string",
      "relevanceScore": "float (0-1)"
    }
  ]
}
```

## User Session API

### Create or Get Session
```
POST /api/session
```
**Description**: Creates a new user session or retrieves an existing one

**Request**:
- Headers:
  - Content-Type: application/json
- Body:
```
{
  "sessionId": "string (optional, client-generated)",
  "userId": "uuid (optional, for authenticated users)"
}
```

**Response**:
```
Status: 200 OK
Content-Type: application/json

{
  "sessionId": "string",
  "userId": "uuid (if authenticated)",
  "createdAt": "ISO 8601 datetime",
  "lastAccessedAt": "ISO 8601 datetime",
  "currentModule": "integer (optional)",
  "currentChapterId": "uuid (optional)"
}
```

### Update Session Progress
```
PUT /api/session/{sessionId}/progress
```
**Description**: Updates the user's learning progress

**Request**:
- Headers:
  - Content-Type: application/json
- Path Parameters:
  - sessionId: string (user session identifier)
- Body:
```
{
  "currentModule": "integer",
  "currentChapterId": "uuid",
  "progressData": "JSON (detailed progress information)"
}
```

**Response**:
```
Status: 200 OK
Content-Type: application/json

{
  "sessionId": "string",
  "currentModule": "integer",
  "currentChapterId": "uuid",
  "progressData": "JSON",
  "updatedAt": "ISO 8601 datetime"
}
```