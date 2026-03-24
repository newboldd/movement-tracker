/* Global: track most recently viewed subject across all pages.
   Any page that loads a subject should call setLastSubject(id).
   Any page that needs a default subject should call getLastSubject(). */
function setLastSubject(id) {
    if (id) localStorage.setItem('mt_lastSubjectId', String(id));
}
function getLastSubject() {
    return parseInt(localStorage.getItem('mt_lastSubjectId')) || null;
}

/* Update nav links to include last viewed subject where applicable. */
(function() {
    const lastSubj = getLastSubject();
    if (!lastSubj) return;

    // MediaPipe link
    const mpLink = document.querySelector('nav a[href*="mediapipe-select"]');
    if (mpLink && !mpLink.href.includes('subject=')) {
        mpLink.href = '/mediapipe-select?subject=' + lastSubj;
    }

    // Deidentify link
    const deidLink = document.querySelector('nav a[href="/deidentify"]');
    if (deidLink) {
        deidLink.href = '/deidentify?subject=' + lastSubj;
    }

    // Results link
    const resLink = document.querySelector('nav a[href="/results"]');
    if (resLink && !resLink.href.includes('subject=')) {
        resLink.href = '/results?subject=' + lastSubj;
    }

    // Videos link
    const vidLink = document.querySelector('nav a[href="/videos"]');
    if (vidLink && !vidLink.href.includes('subject=')) {
        vidLink.href = '/videos?subject=' + lastSubj;
    }

    // MANO link
    const manoLink = document.querySelector('nav a[href="/mano"]');
    if (manoLink && !manoLink.href.includes('subject=')) {
        manoLink.href = '/mano?subject=' + lastSubj;
    }

    // DLC labeling link
    const dlcLink = document.querySelector('nav a[href*="labeling-select"]');
    if (dlcLink && !dlcLink.href.includes('subject=')) {
        dlcLink.href = '/labeling-select?subject=' + lastSubj;
    }
})();

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
