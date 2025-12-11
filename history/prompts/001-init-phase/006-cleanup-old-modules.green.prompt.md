---
id: 006
title: Cleanup old modules
stage: green
date: 2025-12-07
surface: agent
model: Qwen
feature: 001-init-phase
branch: 001-init-phase
user: Administrator
command: /sp.implement --title "Cleanup old modules" --task "Remove irrelevant folders and ensure only final 4 modules exist" --code """
1. Delete these old module folders completely:
   - docs/module-1-foundational
   - docs/module-2-perception
   - docs/module-3-manipulation
   - docs/module-4-advanced
   - docs/tutorial-basics
   - docs/tutorial-extras

2. Keep ONLY these final official modules:
   - docs/module-1-the-robotic-nervous-system
   - docs/module-2-the-digital-twin
   - docs/module-3-the-ai-robot-brain
   - docs/module-4-vision-language-action-systems

3. Do NOT touch:
   - docs/index.md
   - docs/intro.md

4. Update sidebars.ts so ONLY these 4 modules appear.

5. After cleanup:
   - scan all remaining docs folders
   - add proper titles for Option A:
       # Module X — Name  
       ## Chapter Y — Title

6. Commit changes with:
   chore(cleanup): removed old modules and applied structured chapter titles
"""
labels: [cleanup, modules, documentation]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - D:\Quater4\humanoid-robotics-course\docs\module-1-the-robotic-nervous-system\overview.md
 - D:\Quater4\humanoid-robotics-course\docs\module-1-the-robotic-nervous-system\weekly-breakdown.md
 - D:\Quater4\humanoid-robotics-course\docs\module-1-the-robotic-nervous-system\deep-dive.md
 - D:\Quater4\humanoid-robotics-course\docs\module-1-the-robotic-nervous-system\practical-lab.md
 - D:\Quater4\humanoid-robotics-course\docs\module-1-the-robotic-nervous-system\simulation.md
 - D:\Quater4\humanoid-robotics-course\docs\module-1-the-robotic-nervous-system\assignment.md
 - D:\Quater4\humanoid-robotics-course\docs\module-1-the-robotic-nervous-system\intro.md
 - D:\Quater4\humanoid-robotics-course\docs\module-2-the-digital-twin\overview.md
 - D:\Quater4\humanoid-robotics-course\docs\module-2-the-digital-twin\weekly-breakdown.md
 - D:\Quater4\humanoid-robotics-course\docs\module-2-the-digital-twin\deep-dive.md
 - D:\Quater4\humanoid-robotics-course\docs\module-2-the-digital-twin\practical-lab.md
 - D:\Quater4\humanoid-robotics-course\docs\module-2-the-digital-twin\simulation.md
 - D:\Quater4\humanoid-robotics-course\docs\module-2-the-digital-twin\assignment.md
 - D:\Quater4\humanoid-robotics-course\docs\module-2-the-digital-twin\intro.md
 - D:\Quater4\humanoid-robotics-course\docs\module-3-the-ai-robot-brain\overview.md
 - D:\Quater4\humanoid-robotics-course\docs\module-3-the-ai-robot-brain\weekly-breakdown.md
 - D:\Quater4\humanoid-robotics-course\docs\module-3-the-ai-robot-brain\deep-dive.md
 - D:\Quater4\humanoid-robotics-course\docs\module-3-the-ai-robot-brain\practical-lab.md
 - D:\Quater4\humanoid-robotics-course\docs\module-3-the-ai-robot-brain\simulation.md
 - D:\Quater4\humanoid-robotics-course\docs\module-3-the-ai-robot-brain\assignment.md
 - D:\Quater4\humanoid-robotics-course\docs\module-3-the-ai-robot-brain\intro.md
 - D:\Quater4\humanoid-robotics-course\docs\module-4-vision-language-action-systems\overview.md
 - D:\Quater4\humanoid-robotics-course\docs\module-4-vision-language-action-systems\weekly-breakdown.md
 - D:\Quater4\humanoid-robotics-course\docs\module-4-vision-language-action-systems\deep-dive.md
 - D:\Quater4\humanoid-robotics-course\docs\module-4-vision-language-action-systems\practical-lab.md
 - D:\Quater4\humanoid-robotics-course\docs\module-4-vision-language-action-systems\simulation.md
 - D:\Quater4\humanoid-robotics-course\docs\module-4-vision-language-action-systems\assignment.md
 - D:\Quater4\humanoid-robotics-course\docs\module-4-vision-language-action-systems\intro.md
tests:
 - D:\Quater4\humanoid-robotics-course\specs\001-init-phase\tasks.md
 - D:\Quater4\humanoid-robotics-course\specs\001-init-phase\plan.md
---

## Prompt

/sp.implement --title "Cleanup old modules" --task "Remove irrelevant folders and ensure only final 4 modules exist" --code """
1. Delete these old module folders completely:
   - docs/module-1-foundational
   - docs/module-2-perception
   - docs/module-3-manipulation
   - docs/module-4-advanced
   - docs/tutorial-basics
   - docs/tutorial-extras

2. Keep ONLY these final official modules:
   - docs/module-1-the-robotic-nervous-system
   - docs/module-2-the-digital-twin
   - docs/module-3-the-ai-robot-brain
   - docs/module-4-vision-language-action-systems

3. Do NOT touch:
   - docs/index.md
   - docs/intro.md

4. Update sidebars.ts so ONLY these 4 modules appear.

5. After cleanup:
   - scan all remaining docs folders
   - add proper titles for Option A:
       # Module X — Name  
       ## Chapter Y — Title

6. Commit changes with:
   chore(cleanup): removed old modules and applied structured chapter titles
"""

## Response snapshot

Successfully implemented the requested changes:
1. Verified that old module folders (module-1-foundational, module-2-perception, etc.) were removed (Git commit output confirmed deletion)
2. Confirmed that only the 4 final official modules exist:
   - docs/module-1-the-robotic-nervous-system
   - docs/module-2-the-digital-twin
   - docs/module-3-the-ai-robot-brain
   - docs/module-4-vision-language-action-systems
3. Verified that sidebars.ts already only contained these 4 modules
4. Updated all chapter files across all 4 modules to follow the proper title format:
   - "# Module X — Name"
   - "## Chapter Y — Title"
5. Committed changes with the message: "chore(cleanup): removed old modules and applied structured chapter titles"

## Outcome

- ✅ Impact: Cleaned up outdated documentation modules and standardized chapter titles
- 🧪 Tests: No tests run, but verified file structure and formatting
- 📁 Files: Updated all chapter files in the 4 official modules with structured titles
- 🔁 Next prompts: Ready for content development in the standardized structure
- 🧠 Reflection: The cleanup was more extensive than initially apparent, with Git showing removal of multiple old modules that were no longer needed.

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Begin content development for the structured modules