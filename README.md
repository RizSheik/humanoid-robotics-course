# Physical AI & Humanoid Robotics Course

A comprehensive textbook on Physical AI and Humanoid Robotics, built with Docusaurus.

## Overview

This repository contains a complete textbook covering the fundamentals and advanced topics in humanoid robotics, including:
- The robotic nervous system (ROS 2)
- Digital twin technologies (Gazebo, Unity)
- AI for robot brains (NVIDIA Isaac)
- Vision-language-action systems
- Capstone project on autonomous humanoids

## Getting Started

### Prerequisites

- Node.js (version 20 or higher)
- npm or yarn

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/RizSheik/humanoid-robotics-course.git
   ```

2. Navigate to the project directory:
   ```bash
   cd humanoid-robotics-course
   ```

3. Install dependencies:
   ```bash
   npm install
   ```

### Development

To start a local development server with live reloading:
```bash
npm start
```

To build the static site for production:
```bash
npm run build
```

To serve the built site locally for preview:
```bash
npm run serve
```

## Project Structure

- `docs/` - Contains all textbook modules and chapters
  - `module-1-the-robotic-nervous-system/` - ROS 2 and communication fundamentals
  - `module-2-the-digital-twin/` - Simulation and modeling
  - `module-3-the-ai-robot-brain/` - AI and decision making
  - `module-4-vision-language-action-systems/` - VLA and multimodal systems
  - `capstone-the-autonomous-humanoid/` - Capstone project
  - `appendices/` - Reference materials
- `src/` - Custom React components and CSS
- `static/` - Static assets like images
- `docusaurus.config.js` - Site configuration
- `sidebars.ts` - Navigation structure

## Contributing

We welcome contributions to improve the content. To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

Please ensure your changes align with the textbook's educational objectives and maintain the formal technical tone.

## Deployment

This site is automatically deployed to GitHub Pages using GitHub Actions. When changes are pushed to the `main` branch, the site is rebuilt and published.

The GitHub Actions workflow is defined in `.github/workflows/deploy.yml`.

## License

This textbook content is licensed under [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) (CC BY-NC 4.0).

## Support

If you encounter any issues with the textbook content or the website, please open an issue in this repository.