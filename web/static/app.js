// BigBrother — Frontend JavaScript
// Shared utilities and common functionality

(function() {
    'use strict';

    // Add active state visual feedback
    document.querySelectorAll('.nav-link').forEach(link => {
        if (link.href === window.location.href) {
            link.classList.add('active');
        }
    });

    // Console branding
    console.log(
        '%c⊕ BIGBROTHER %cv0.1 — Always Watching.',
        'color: #ff4444; font-size: 16px; font-weight: bold;',
        'color: #888; font-size: 12px;'
    );

})();
