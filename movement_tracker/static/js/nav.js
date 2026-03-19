/* Update "DLC" nav link to return to the last active session. */
(function() {
    const sid = sessionStorage.getItem('dlc_lastSessionId');
    if (!sid) return;
    const link = document.getElementById('labelingLink')
              || document.querySelector('nav a[href*="labeling-select"]');
    if (link) link.href = '/labeling?session=' + sid;
})();

/* Conditionally show/hide Tutorials nav link based on setting. */
(function() {
    const tutLink = document.querySelector('nav a[href="/tutorials"]');
    if (!tutLink) return;
    fetch('/api/settings').then(r => r.json()).then(cfg => {
        if (cfg.show_tutorials === false) tutLink.style.display = 'none';
    }).catch(() => {});
})();
