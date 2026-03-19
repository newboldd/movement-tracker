/**
 * Tutorials index page — renders card grid from TUTORIALS data,
 * grouped by series (beginner / advanced).
 */
(function () {
    'use strict';

    function renderSection(container, title, description, tutorials) {
        const heading = document.createElement('div');
        heading.className = 'tut-section-header';
        heading.innerHTML = `
            <h3 style="margin:0 0 4px;font-size:18px;font-weight:600;">${title}</h3>
            <p style="margin:0;font-size:14px;color:var(--text-muted,#aaa);">${description}</p>
        `;
        container.appendChild(heading);

        const grid = document.createElement('div');
        grid.className = 'tut-grid';
        tutorials.forEach(t => {
            const a = document.createElement('a');
            a.className = 'tut-card';
            a.href = `/tutorial?id=${t.id}`;
            a.innerHTML = `
                <div class="tut-card-header">
                    <span class="tut-num">${t.id}</span>
                    <span class="tut-card-title">${t.title}</span>
                </div>
                <div class="tut-card-desc">${t.subtitle}</div>
                <div class="tut-card-footer">
                    <span class="time">${t.time}</span>
                    <span class="arrow">Start &rarr;</span>
                </div>
            `;
            grid.appendChild(a);
        });
        container.appendChild(grid);
    }

    function init() {
        const container = document.getElementById('tutGrid');
        if (!container || !window.TUTORIALS) return;

        const beginner = window.TUTORIALS.filter(t => t.series === 'beginner');
        const advanced = window.TUTORIALS.filter(t => t.series === 'advanced');

        renderSection(container, 'Getting Started',
            'Learn the basics: from raw video to trained DLC model.',
            beginner);

        if (advanced.length) {
            const spacer = document.createElement('div');
            spacer.style.height = '32px';
            container.appendChild(spacer);

            renderSection(container, 'Advanced',
                'Refinement, event analysis, results, and 3D hand fitting.',
                advanced);
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
