/* ─── Chart.js Defaults ──────────────────────────────────── */
Chart.defaults.font.family = '"Sora", "PingFang SC", system-ui, sans-serif';
Chart.defaults.font.size   = 12;
Chart.defaults.color       = '#8b8680';

/* ─── Scheme 03: Two-Tone Focus ──────────────────────────── */
// Baselines: single muted warm tone
// HiMe (Ours): amber pop
// Human: near-black warm dark
const MUTED     = '#c8b8a8';
const HIGHLIGHT = '#bb8963';
const DARK      = '#4a3e34';

const darkTooltip = {
  backgroundColor: '#1a1a1a',
  titleColor: '#fff',
  bodyColor:  '#e8e4de',
  titleFont:  { family: '"Sora",sans-serif', size: 12 },
  bodyFont:   { family: '"Sora",sans-serif', size: 12 },
  padding: 12,
  cornerRadius: 8,
};

const scaleY = {
  min: 0, max: 100,
  grid:   { display: false },
  border: { display: false },
  ticks:  { color: '#8b8680', font: { size: 11 }, callback: v => v + '%' },
};

const scaleX = {
  grid:   { display: false },
  border: { display: false },
  ticks:  { color: '#2c251e', font: { size: 11 }, maxRotation: 0 },
};

const sharedBar = {
  borderRadius: { topLeft: 4, topRight: 4, bottomLeft: 0, bottomRight: 0 },
  borderSkipped: false,
};

const sharedLegend = {
  position: 'bottom',
  labels: {
    boxWidth: 12,
    boxHeight: 12,
    borderRadius: 3,
    padding: 14,
    color: '#8b8680',
    font: { size: 11 },
  },
};

/* ─── Chart 1: Main Results ──────────────────────────────── */
new Chart(document.getElementById('chartMain'), {
  type: 'bar',
  data: {
    labels: ['Object Search', 'Counting', 'Rearrangement', 'Average'],
    datasets: [
      {
        label: 'Transient Memory',
        data: [12, 8, 22, 14],
        backgroundColor: MUTED,
        ...sharedBar,
      },
      {
        label: 'Transient + Sentry',
        data: [27, 23, 28, 26],
        backgroundColor: MUTED,
        ...sharedBar,
      },
      {
        label: 'Flat Memory',
        data: [64, 58, 73, 65],
        backgroundColor: MUTED,
        ...sharedBar,
      },
      {
        label: 'HiMe w/o Sentry',
        data: [70, 65, 69, 68],
        backgroundColor: MUTED,
        ...sharedBar,
      },
      {
        label: 'HiMe (Ours)',
        data: [92, 92, 87, 90],
        backgroundColor: HIGHLIGHT,
        ...sharedBar,
      },
      {
        label: 'Human High-level',
        data: [95, 97, 93, 95],
        backgroundColor: DARK,
        ...sharedBar,
      },
    ],
  },
  options: {
    responsive: true,
    aspectRatio: 2.4,
    plugins: {
      legend: sharedLegend,
      tooltip: {
        ...darkTooltip,
        callbacks: { label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y}%` },
      },
    },
    scales: { x: scaleX, y: scaleY },
  },
});

/* ─── Chart 2: Memory Management Ablation ────────────────── */
new Chart(document.getElementById('chartManagement'), {
  type: 'bar',
  data: {
    labels: ['Object Search', 'Counting', 'Rearrangement'],
    datasets: [
      {
        label: 'FIFO',
        data: [70, 62, 72],
        backgroundColor: MUTED,
        ...sharedBar,
      },
      {
        label: 'No Management',
        data: [88, 85, 85],
        backgroundColor: MUTED,
        ...sharedBar,
      },
      {
        label: 'HiMe (Ours)',
        data: [92, 92, 87],
        backgroundColor: HIGHLIGHT,
        ...sharedBar,
      },
    ],
  },
  options: {
    responsive: true,
    plugins: {
      legend: sharedLegend,
      tooltip: {
        ...darkTooltip,
        callbacks: { label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y}%` },
      },
    },
    scales: { x: scaleX, y: scaleY },
  },
});

/* ─── Chart 3: Memory Modality Ablation ─────────────────── */
new Chart(document.getElementById('chartModality'), {
  type: 'bar',
  data: {
    labels: ['Object Search', 'Counting', 'Rearrangement'],
    datasets: [
      {
        label: 'Text Only',
        data: [74, 91, 80],
        backgroundColor: MUTED,
        ...sharedBar,
      },
      {
        label: 'Image Only',
        data: [86, 78, 85],
        backgroundColor: MUTED,
        ...sharedBar,
      },
      {
        label: 'Cross-Modal (HiMe)',
        data: [92, 92, 87],
        backgroundColor: HIGHLIGHT,
        ...sharedBar,
      },
    ],
  },
  options: {
    responsive: true,
    plugins: {
      legend: sharedLegend,
      tooltip: {
        ...darkTooltip,
        callbacks: { label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y}%` },
      },
    },
    scales: { x: scaleX, y: scaleY },
  },
});

/* ─── Chart 4: Sentry Window Analysis ───────────────────── */
new Chart(document.getElementById('chartSentry'), {
  type: 'line',
  data: {
    labels: ['1', '2', '3', '4', '5', '6', '7', '8'],
    datasets: [
      {
        label: 'Precision',
        data: [76, 77.5, 78.8, 79.8, 80.6, 81.2, 81.7, 82],
        borderColor: HIGHLIGHT,
        backgroundColor: 'rgba(187,137,99,0.12)',
        pointBackgroundColor: HIGHLIGHT,
        pointRadius: 5,
        borderWidth: 2.5,
        fill: true,
        tension: 0.4,
      },
      {
        label: 'Recall',
        data: [22, 25, 27.5, 29.5, 31, 32.5, 33.8, 35],
        borderColor: MUTED,
        backgroundColor: 'rgba(200,184,168,0.12)',
        pointBackgroundColor: MUTED,
        pointRadius: 4,
        borderWidth: 1.5,
        fill: true,
        tension: 0.4,
      },
    ],
  },
  options: {
    responsive: true,
    plugins: {
      legend: {
        position: 'bottom',
        labels: {
          boxWidth: 10, boxHeight: 10, borderRadius: 2,
          color: '#8b8680', font: { size: 11 }, padding: 10,
        },
      },
      tooltip: {
        ...darkTooltip,
        callbacks: { label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(1)}%` },
      },
    },
    scales: {
      x: {
        ...scaleX,
        title: {
          display: true,
          text: 'Window Size (frames)',
          color: '#8b8680',
          font: { size: 11 },
          padding: { top: 6 },
        },
      },
      y: { ...scaleY },
    },
  },
});

/* ─── Reveal on Scroll ───────────────────────────────────── */
const revealEls = document.querySelectorAll('.reveal');

const revealObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('is-visible');
        revealObserver.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.08 }
);

revealEls.forEach((el) => revealObserver.observe(el));

/* ─── Sidebar Active Link ────────────────────────────────── */
const sections = document.querySelectorAll('section[id], header[id]');
const navLinks = document.querySelectorAll('.sidebar-link');

const navObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        navLinks.forEach((link) => {
          link.classList.toggle(
            'active',
            link.getAttribute('href') === `#${entry.target.id}`
          );
        });
      }
    });
  },
  { rootMargin: '-30% 0px -60% 0px' }
);

sections.forEach((section) => navObserver.observe(section));
