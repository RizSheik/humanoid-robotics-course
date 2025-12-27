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
    title: 'Module 1 – Robotic Nervous System (ROS 2)',
    Svg: () => (
      <img src="img/robotic-nervous-system.svg" className="featureSvg" alt="Robotic Nervous System" />
    ),
    description: (
      <>
        Learn about Robot Operating System 2 (ROS 2) - the middleware that enables communication between
        different components of a robot system.
      </>
    ),
    to: '/docs/module-1-the-robotic-nervous-system/module-1-intro'
  },
  {
    title: 'Module 2 – Digital Twin (Gazebo & Unity)',
    Svg: () => (
      <img src="img/digital-twin.svg" className="featureSvg" alt="Digital Twin" />
    ),
    description: (
      <>
        Explore digital twin technologies using Gazebo and Unity for simulating and testing robotic systems
        in virtual environments.
      </>
    ),
    to: '/docs/module-2-the-digital-twin/module-2-intro'
  },
  {
    title: 'Module 3 – AI Robot Brain (NVIDIA Isaac)',
    Svg: () => (
      <img src="img/ai-robot-brain.svg" className="featureSvg" alt="AI Robot Brain" />
    ),
    description: (
      <>
        Discover AI and machine learning frameworks for robotic perception, planning, and control
        using NVIDIA Isaac platform.
      </>
    ),
    to: '/docs/module-3-the-ai-robot-brain/module-3-intro'
  },
  {
    title: 'Module 4 – Vision-Language-Action (VLA)',
    Svg: () => (
      <img src="img/vision-language-action.svg" className="featureSvg" alt="Vision-Language-Action" />
    ),
    description: (
      <>
        Understand Vision-Language-Action models that enable robots to perceive, understand,
        and interact with the world through multimodal AI.
      </>
    ),
    to: '/docs/module-4-vision-language-action-systems/module-4-intro'
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
