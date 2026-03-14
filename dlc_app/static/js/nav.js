/* Update "Labeling" nav link to return to the last active session. */
(function() {
    const sid = sessionStorage.getItem('dlc_lastSessionId');
    if (!sid) return;
    const link = document.getElementById('labelingLink')
              || document.querySelector('nav a[href*="labeling-select"]');
    if (link) link.href = '/labeling?session=' + sid;
})();
