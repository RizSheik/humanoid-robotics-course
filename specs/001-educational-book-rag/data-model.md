# Data Model: Educational Book with Integrated RAG Chatbot

## Entity: Module
**Description**: Represents one of the four main sections of the educational book, containing 4 chapters each
**Fields**:
- id: UUID (Primary Key)
- title: String (Required)
- description: String (Optional)
- order: Integer (Required, 1-4)
- createdAt: DateTime (Required)
- updatedAt: DateTime (Required)

**Relationships**:
- One-to-Many: Module → Chapter (one module contains many chapters)

## Entity: Chapter
**Description**: Individual lesson units that form part of a module, each containing text content, images, and exercises
**Fields**:
- id: UUID (Primary Key)
- moduleId: UUID (Foreign Key to Module)
- title: String (Required)
- content: String (Required, Markdown format)
- order: Integer (Required, 1-4 within module)
- learningObjectives: Array<String> (Optional)
- prerequisites: Array<String> (Optional)
- createdAt: DateTime (Required)
- updatedAt: DateTime (Required)

**Relationships**:
- Many-to-One: Chapter → Module (many chapters belong to one module)
- One-to-Many: Chapter → ContentEmbedding (one chapter can have multiple embeddings)

## Entity: LearningContent
**Description**: Educational materials including text, images, diagrams, and other media that comprise the course curriculum
**Fields**:
- id: UUID (Primary Key)
- chapterId: UUID (Foreign Key to Chapter)
- type: Enum ['text', 'image', 'diagram', 'video', 'code', 'exercise', 'quiz'] (Required)
- content: String (Required for text/code, path for media)
- altText: String (Optional, for accessibility)
- caption: String (Optional)
- order: Integer (Required, order within chapter)
- createdAt: DateTime (Required)
- updatedAt: DateTime (Required)

**Relationships**:
- Many-to-One: LearningContent → Chapter (many content items belong to one chapter)

## Entity: ChatQuery
**Description**: Questions or prompts submitted by users to the RAG chatbot for assistance with the educational content
**Fields**:
- id: UUID (Primary Key)
- userId: UUID (Optional, for logged-in users)
- sessionId: UUID (Required, for anonymous users)
- query: String (Required)
- response: String (Required)
- sources: Array<JSON> (Required, references to educational content)
- timestamp: DateTime (Required)
- isHelpful: Boolean (Optional, for feedback)
- createdAt: DateTime (Required)

**Relationships**:
- Many-to-One: ChatQuery → UserSession (many queries per session)

## Entity: ContentEmbedding
**Description**: Numerical representations of the educational content that enable semantic search capabilities
**Fields**:
- id: UUID (Primary Key)
- contentId: UUID (Foreign Key to LearningContent or Chapter)
- embedding: Array<Float> (Required, the vector representation)
- text: String (Required, the original text that was embedded)
- type: Enum ['chapter', 'section', 'paragraph', 'image_caption'] (Required)
- createdAt: DateTime (Required)

**Relationships**:
- Many-to-One: ContentEmbedding → LearningContent (many embeddings can come from one content item)

## Entity: UserSession
**Description**: Temporary data structure that maintains the state of a user's interaction with the educational platform
**Fields**:
- id: UUID (Primary Key)
- sessionId: String (Required, for anonymous users)
- userId: UUID (Optional, for logged-in users)
- currentModule: Integer (Optional, module user is currently viewing)
- currentChapter: UUID (Optional, chapter user is currently viewing)
- progress: JSON (Optional, tracking user's progress)
- lastAccessedAt: DateTime (Required)
- createdAt: DateTime (Required)

**Relationships**:
- One-to-Many: UserSession → ChatQuery (one session can have many queries)

## Entity: Image
**Description**: Images used in the educational content, stored for efficient delivery
**Fields**:
- id: UUID (Primary Key)
- filename: String (Required)
- path: String (Required, relative to src/static/img/)
- altText: String (Required, for accessibility)
- caption: String (Optional)
- associatedChapterId: UUID (Foreign Key to Chapter)
- uploadDate: DateTime (Required)

**Relationships**:
- Many-to-One: Image → Chapter (many images can be associated with one chapter)