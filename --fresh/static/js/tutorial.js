/**
 * Tutorial viewer — renders a single tutorial from TUTORIALS data,
 * driven by ?id= query parameter.
 */
(function () {
    'use strict';

    const $ = id => document.getElementById(id);

    function init() {
        const tutorials = window.TUTORIALS;
        if (!tutorials) return;

        const params = new URLSearchParams(window.location.search);
        const id = parseInt(params.get('id')) || 1;
        const tut = tutorials.find(t => t.id === id);
        if (!tut) {
            $('tutTitle').textContent = 'Tutorial not found';
            return;
        }

        // Header
        $('tutTitle').textContent = `${tut.id}. ${tut.title}`;
        $('tutSubtitle').textContent = tut.subtitle;
        $('breadcrumbTitle').textContent = tut.title;
        document.title = `${tut.title} — Movement Tracker`;

        // Sidebar TOC
        const tocList = $('tocList');
        tut.steps.forEach((step, i) => {
            const link = document.createElement('a');
            link.className = 'toc-link';
            link.href = `#step-${i + 1}`;
            link.innerHTML = `<span class="toc-num">${i + 1}</span><span>${step.title}</span>`;
            link.addEventListener('click', e => {
                e.preventDefault();
                const el = document.getElementById(`step-${i + 1}`);
                if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
            });
            tocList.appendChild(link);
        });

        // Steps
        const container = $('stepsContainer');
        tut.steps.forEach((step, i) => {
            const div = document.createElement('div');
            div.className = 'tut-step';
            div.id = `step-${i + 1}`;

            let html = `
                <div class="tut-step-header">
                    <span class="tut-step-num">${i + 1}</span>
                    <span class="tut-step-title">${step.title}</span>
                </div>
                <div class="tut-step-body">${step.body}</div>
            `;

            if (step.tips && step.tips.length) {
                step.tips.forEach(tip => {
                    html += `<div class="tut-tip">${tip}</div>`;
                });
            }

            div.innerHTML = html;
            container.appendChild(div);
        });

        // Bottom navigation
        const navBottom = $('tutNavBottom');
        const prev = tutorials.find(t => t.id === id - 1);
        const next = tutorials.find(t => t.id === id + 1);

        const prevBtn = prev
            ? `<a class="tut-nav-btn" href="/tutorial?id=${prev.id}">&larr; ${prev.title}</a>`
            : `<span class="tut-nav-btn disabled">&larr; Previous</span>`;

        const nextBtn = next
            ? `<a class="tut-nav-btn" href="/tutorial?id=${next.id}">${next.title} &rarr;</a>`
            : `<a class="tut-nav-btn" href="/tutorials">Back to Tutorials</a>`;

        navBottom.innerHTML = prevBtn + nextBtn;

        // Scroll spy — highlight active TOC entry
        const tocLinks = tocList.querySelectorAll('.toc-link');
        const stepEls = container.querySelectorAll('.tut-step');
        const observer = new IntersectionObserver(entries => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const idx = Array.from(stepEls).indexOf(entry.target);
                    tocLinks.forEach((l, j) => l.classList.toggle('active', j === idx));
                }
            });
        }, { rootMargin: '-20% 0px -70% 0px' });
        stepEls.forEach(el => observer.observe(el));

        // Activate first by default
        if (tocLinks.length) tocLinks[0].classList.add('active');
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
