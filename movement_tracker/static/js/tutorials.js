/**
 * Tutorials index page — renders card grid from TUTORIALS data.
 */
(function () {
    'use strict';

    function init() {
        const grid = document.getElementById('tutGrid');
        if (!grid || !window.TUTORIALS) return;

        window.TUTORIALS.forEach(t => {
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
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
