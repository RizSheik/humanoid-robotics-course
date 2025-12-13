import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import styles from './styles.module.css';

interface ModuleCardItem {
  title: string;
  description: string;
  to: string;
  icon: string; // emoji or short name for icon
}

interface ModuleOverviewCardsProps {
  moduleCards: ModuleCardItem[];
}

export default function ModuleOverviewCards({ moduleCards }: ModuleOverviewCardsProps): JSX.Element {
  return (
    <section className={styles.moduleCards}>
      <div className="container">
        <div className="row">
          {moduleCards.map((item, idx) => (
            <div key={idx} className={clsx('col col--3', styles.moduleCardCol)}>
              <Link to={item.to} className={styles.moduleCardLink}>
                <div className={styles.moduleCard}>
                  <div className={styles.moduleCardHeader}>
                    <h3 className={styles.moduleCardTitle}>
                      {item.icon} {item.title}
                    </h3>
                  </div>
                  <div className={styles.moduleCardBody}>
                    <p className={styles.moduleCardDescription}>
                      {item.description}
                    </p>
                  </div>
                  <div className={styles.moduleCardFooter}>
                    <span className={styles.moduleCardLinkText}>
                      Explore →
                    </span>
                  </div>
                </div>
              </Link>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}