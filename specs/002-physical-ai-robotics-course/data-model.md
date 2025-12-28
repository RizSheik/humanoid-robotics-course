# Data Model: Physical AI & Humanoid Robotics Course - Phase 3

## Entities

### Module
- **Description**: A top-level organizational unit containing related chapters and learning objectives for the robotics course
- **Attributes**:
  - `id`: Unique identifier for the module (derived from directory name)
  - `title`: Display title for the module (e.g., "Physical AI & Embodied Intelligence")
  - `description`: Brief description of the module's content and objectives
  - `learningOutcomes`: List of specific learning outcomes students should achieve
  - `durationWeeks`: Estimated duration to complete the module in weeks
  - `prerequisites`: List of prerequisite modules or knowledge areas
  - `resources`: List of required resources (software, hardware, reading materials)
- **Relationships**: Contains multiple Chapters; belongs to the overall Course
- **Validation**: Must have at least one Chapter, valid learning outcomes, and achievable duration

### Chapter
- **Description**: A focused section within a module that covers a specific topic or skill in robotics
- **Attributes**:
  - `id`: Unique identifier for the chapter (derived from filename)
  - `title`: Display title for the chapter
  - `learningObjectives`: Specific objectives students should achieve after completing the chapter
  - `content`: Markdown content following formal textbook structure with sections and subsections
  - `practicalLab`: Practical exercises and lab activities (if applicable)
  - `simulationComponent`: Simulation environment or scenario (if applicable)
  - `assignment`: Assignment or project related to chapter content
  - `quiz`: Assessment questions to verify understanding
- **Relationships**: Belongs to one Module; contains multiple Sections
- **Validation**: Must have learning objectives and follow textbook structure requirements

### Section
- **Description**: A subdivision of a chapter that focuses on a specific concept or topic
- **Attributes**:
  - `id`: Unique identifier for the section (derived from heading)
  - `title`: Display title for the section
  - `content`: Detailed content explaining the topic
  - `examples`: Code snippets, diagrams, equations, or practical examples
- **Relationships**: Belongs to one Chapter; may have multiple Subsections
- **Validation**: Must have a clear focus and contribute to chapter objectives

### LearningOutcome
- **Description**: A specific, measurable skill or knowledge that students should acquire
- **Attributes**:
  - `id`: Unique identifier for the learning outcome
  - `description`: Clear description of what students should be able to do
  - `level`: Cognitive difficulty level (e.g., "remember", "understand", "apply", "analyze", "evaluate", "create")
  - `assessment`: How the outcome will be assessed (quiz, assignment, practical exercise)
  - `moduleId`: Reference to the module this outcome belongs to
- **Relationships**: Associated with a Module; evaluated through Assessments
- **Validation**: Must be measurable and aligned with module objectives

### Assessment
- **Description**: Evaluation mechanism to verify student understanding and skills
- **Attributes**:
  - `id`: Unique identifier for the assessment
  - `type`: Type of assessment (quiz, assignment, practical lab, project)
  - `difficulty`: Difficulty level (beginner, intermediate, advanced)
  - `questions`: List of questions or tasks to evaluate students
  - `rubric`: Grading criteria and expected outcomes
  - `chapterId`: Reference to the chapter this assessment relates to
- **Relationships**: Associated with a Chapter; evaluates related LearningOutcomes
- **Validation**: Questions must align with chapter objectives and learning outcomes

### SimulationEnvironment
- **Description**: Virtual environment for testing and validating robotics concepts
- **Attributes**:
  - `id`: Unique identifier for the simulation environment
  - `name`: Name of the environment (e.g., "Isaac Sim", "Gazebo", "Unity")
  - `description`: Overview of what the environment is used for
  - `scenarios`: List of specific scenarios that can be simulated
  - `assets`: Required models, robots, or scenes for the environment
  - `chapterId`: Reference to the chapter that uses this simulation
- **Relationships**: Associated with Chapters requiring practical labs or simulation components
- **Validation**: Must have compatible assets and scenarios that match learning objectives

## Relationships

```
[Course] 1 -- * [Module]
[Module] 1 -- * [Chapter]
[Chapter] 1 -- * [Section]
[Module] 1 -- * [LearningOutcome]
[Chapter] 1 -- * [Assessment]
[Chapter] 0..1 -- 1 [SimulationEnvironment]
[Chapter] 0..1 -- 1 [PracticalLab]
```

## Validation Rules

1. Each Module must have at least 3 distinct Learning Outcomes
2. Each Chapter must have Learning Objectives that align with the Module's Learning Outcomes
3. All Assessments must be linked to specific Learning Outcomes they evaluate
4. SimulationEnvironments must have corresponding practical components in chapters
5. Content must follow formal textbook structure with sections, subsections, and exercises
6. All learning outcomes must be achievable within the estimated duration
7. Prerequisites must be properly defined and linked between modules
8. Assessment difficulty must match the cognitive level of learning outcomes