// Apply a saved palette (CSS custom-property overrides) on every
// page load.  Runs as soon as the <head> includes this file so the
// overrides take effect before the page paints.
//
// Storage: localStorage key "mt_palette" → JSON dict of CSS var
// names → color strings.  Set / cleared by the palette editor on
// the Subjects dashboard.  Per-browser only (no backend round-trip).
(() => {
    try {
        const raw = localStorage.getItem('mt_palette');
        if (!raw) return;
        const palette = JSON.parse(raw);
        if (!palette || typeof palette !== 'object') return;
        const root = document.documentElement;
        Object.keys(palette).forEach(k => {
            if (k.startsWith('--')) root.style.setProperty(k, palette[k]);
        });
    } catch (_) { /* ignore */ }
})();
