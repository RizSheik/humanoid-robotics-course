import React, { useState, useEffect } from 'react';
import clsx from 'clsx';
import styles from './HeroSlider.module.css';

const slides = [
  {
    title: "Physical AI & Humanoid Robotics",
    subtitle: "AI Systems in the Physical World",
    image: "/static/img/hero/Hero Section Cover The_Course_DetailsPhysical_AI_Humanoid_0.jpg",
    buttonText: "Start Reading",
    buttonLink: "/docs/introduction"
  },
  {
    title: "Module 1: Fundamentals of Physical AI",
    subtitle: "Understanding humanoid robot design principles",
    image: "/static/img/hero/Closeup_illustration_of_humanoid_robot_h_0.jpg",
    buttonText: "Explore Module",
    buttonLink: "/docs/Module-1/overview"
  },
  {
    title: "Advanced Control Systems",
    subtitle: "From perception to action in physical systems",
    image: "/static/img/hero/Ultrarealistic_Gazebo_simulation_scene_w_0.jpg",
    buttonText: "Begin Learning",
    buttonLink: "/docs/Module-3/overview"
  }
];

function Slide({ slide, isActive }) {
  return (
    <div className={clsx(styles.slide, isActive && styles.active)}>
      <div className={styles.textContent}>
        <h1>{slide.title}</h1>
        <p>{slide.subtitle}</p>
        <a href={slide.buttonLink} className={styles.primaryButton}>
          {slide.buttonText}
        </a>
      </div>
      <div className={styles.imageContent}>
        <img src={slide.image} alt={slide.title} />
      </div>
    </div>
  );
}

export default function HeroSlider() {
  const [activeIndex, setActiveIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setActiveIndex((prevIndex) => (prevIndex + 1) % slides.length);
    }, 5000); // Change slide every 5 seconds

    return () => clearInterval(interval);
  }, []);

  return (
    <section className={styles.heroSection}>
      <div className={styles.sliderContainer}>
        {slides.map((slide, index) => (
          <Slide key={index} slide={slide} isActive={index === activeIndex} />
        ))}
      </div>
      <div className={styles.indicators}>
        {slides.map((_, index) => (
          <button
            key={index}
            className={clsx(styles.indicator, index === activeIndex && styles.active)}
            onClick={() => setActiveIndex(index)}
          />
        ))}
      </div>
    </section>
  );
}