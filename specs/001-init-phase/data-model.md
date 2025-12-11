# Data Model: Project Initialization

## Entities

### Module

- **Description**: A top-level organizational unit for grouping related chapters within the textbook.
- **Attributes**:
    - `id`: Unique identifier for the module (derived from folder name).
    - `label`: Display name for the module (from `category.json`).
    - `title`: Title for the module overview page (from `category.json`).
    - `description`: Short description for the module overview page (from `category.json`).
    - `slug`: URL slug for the module overview page (from `category.json`).
    - `items`: List of chapter IDs belonging to this module (from `sidebars.ts`).
- **Relationships**: Contains multiple Chapters.

### Chapter

- **Description**: A single Markdown file representing a section or topic within a module of the textbook.
- **Attributes**:
    - `id`: Unique identifier for the chapter (derived from filename).
    - `title`: Display title for the chapter (from frontmatter in `.md` file).
    - `sidebar_label`: Label used in the sidebar navigation (from frontmatter in `.md` file).
    - `content`: Markdown content of the chapter.
- **Relationships**: Belongs to one Module.
