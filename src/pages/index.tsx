import type {ReactNode} from 'react';
import React, { useState, useEffect } from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

// Robot image slider component
function ImageSlider() {
  const images = [
    { src: 'img/hero/Hero Section Cover The_Course_DetailsPhysical_AI_Humanoid_0.jpg', alt: 'Humanoid Robot Standing' },
    { src: 'img/Architecture_diagram_cloud_workstation_A_0.jpg', alt: 'ROS 2 Architecture' },
    { src: 'img/Closeup_illustration_of_humanoid_robot_h_0.jpg', alt: 'AI Brain Neural Network' },
    { src: 'img/Ultrarealistic_Gazebo_simulation_scene_w_0.jpg', alt: 'Vision-Language-Action System' },
    { src: 'img/Ultrarealistic_textbook_cover_design_for_0.jpg', alt: 'Vision-Language-Action System' },
  ];

  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentIndex((prevIndex) =>
        prevIndex === images.length - 1 ? 0 : prevIndex + 1
      );
    }, 3000); // Change image every 3 seconds

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="imageSlider">
      {images.map((image, index) => (
        <div
          key={index}
          className={`slide ${index === currentIndex ? 'active' : ''}`}
        >
          <img
            src={image.src}
            alt={image.alt}
            style={{ width: '200%', height: '200%', objectFit: 'contain' }}
          />
        </div>
      ))}
    </div>
  );
}

type FeatureItem = {
  title: string;
  Svg: React.ComponentType<React.SVGProps<SVGSVGElement>>;
  description: ReactNode;
  to: string;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Module 1: Physical AI Foundations',
    Svg: () => (
      <img src="img/robotic-nervous-system.svg" className="featureSvg" alt="Physical AI Foundations" />
    ),
    description: (
      <>
        Learn about the foundational elements of Physical AI and embodied intelligence concepts
        that form the basis of humanoid robotics.
      </>
    ),
    to: '/docs/module-1-physical-ai-foundations/overview'
  },
  {
    title: 'Module 2: ROS 2 Fundamentals',
    Svg: () => (
      <img src="img/digital-twin.svg" className="featureSvg" alt="ROS 2 Fundamentals" />
    ),
    description: (
      <>
        Explore Robot Operating System 2 (ROS 2) fundamentals including topics, services,
        actions, and distributed systems for robotic communication.
      </>
    ),
    to: '/docs/module-2-ros-2-fundamentals/overview'
  },
  {
    title: 'Module 3: Digital Twin Simulation',
    Svg: () => (
      <img src="img/ai-robot-brain.svg" className="featureSvg" alt="Digital Twin Simulation" />
    ),
    description: (
      <>
        Discover simulation environments including Gazebo and Unity for robotics development
        and testing in virtual environments.
      </>
    ),
    to: '/docs/module-3-digital-twin-simulation/overview'
  },
  {
    title: 'Module 4: AI Robot Brain',
    Svg: () => (
      <img src="img/vision-language-action.svg" className="featureSvg" alt="AI Robot Brain" />
    ),
    description: (
      <>
        Understand AI algorithms and NVIDIA Isaac Platform for robot decision-making
        and intelligence in autonomous systems.
      </>
    ),
    to: '/docs/module-4-ai-robot-brain/overview'
  },
];

function Feature({title, Svg, description, to}: FeatureItem) {
  return (
    <div className="col col--6">
      <Link to={to}>
        <div className="featureCard">
          <div className="text--center">
            <Svg role="img" />
          </div>
          <div className="text--center padding-horiz--md">
            <Heading as="h3">{title}</Heading>
            <p>{description}</p>
          </div>
        </div>
      </Link>
    </div>
  );
}

function HomepageFeatures() {
  return (
    <section className="features">
      <div className="container">
        <Heading as="h2">Textbook Modules</Heading>
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', 'heroBanner')}>
      <div className="heroSplitContainer">
        <div className="heroTextContent">
          <Heading as="h1" className="hero__title">
            Physical AI & Humanoid Robotics
          </Heading>
          <p className="hero__subtitle">
            Advanced Robotics Textbook — From Theory to Practice
          </p>
          <div className="buttons">
            <Link
              className="button button--secondary button--lg"
              to="/docs/intro">
              Start Reading 📖
            </Link>
          </div>
        </div>
        <div className="heroImageContent">
          <div className="robotFloating">
            <ImageSlider />
          </div>
        </div>
      </div>
    </header>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();

  return (
    <Layout
      title={`Humanoid Robotics Course`}
      description="Advanced Robotics Textbook — From Theory to Practice">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}
