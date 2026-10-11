document.addEventListener('DOMContentLoaded', function () {
    // Navbar scroll shadow
    var navbar = document.querySelector('.navbar');
    if (navbar) {
        window.addEventListener('scroll', function () {
            navbar.classList.toggle('scrolled', window.scrollY > 10);
        }, { passive: true });
    }

    // KaTeX auto-render on all markdown content
    if (typeof renderMathInElement !== 'undefined') {
        document.querySelectorAll('.markdown-content').forEach(function (el) {
            renderMathInElement(el, {
                delimiters: [
                    { left: '$$', right: '$$', display: true },
                    { left: '$', right: '$', display: false },
                    { left: '\\[', right: '\\]', display: true },
                    { left: '\\(', right: '\\)', display: false }
                ],
                throwOnError: false
            });
        });
    }

    // Close the mobile menu before moving to an anchor on the same page.
    document.querySelectorAll('a[href*="#"]').forEach(function (a) {
        a.addEventListener('click', function (e) {
            if (this.origin !== location.origin || this.pathname !== location.pathname || !this.hash) return;
            var hash = this.hash;
            var target = document.getElementById(decodeURIComponent(hash.slice(1)));
            if (!target) return;
            e.preventDefault();
            function moveToTarget() {
                var offset = (navbar ? navbar.offsetHeight : 68) + 24;
                var top = target.getBoundingClientRect().top + window.scrollY - offset;
                var reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
                window.scrollTo({ top: top, behavior: reduced ? 'auto' : 'smooth' });
                history.pushState(null, null, hash);
            }
            var menu = document.getElementById('navbarNav');
            if (menu && menu.classList.contains('show') && window.bootstrap) {
                menu.addEventListener('hidden.bs.collapse', moveToTarget, { once: true });
                bootstrap.Collapse.getOrCreateInstance(menu).hide();
            } else {
                moveToTarget();
            }
        });
    });

    // Only auto-resize small form textareas. The markdown editor stays fixed-height
    // so typing does not keep reflowing the whole page.
    function resizeTextarea(ta) {
        var minHeight = ta.dataset.minHeight || 120;
        ta.style.height = 'auto';
        ta.style.height = Math.max(parseInt(minHeight, 10), ta.scrollHeight) + 'px';
    }

    document.querySelectorAll('textarea.form-control:not(.markdown-editor)').forEach(function (ta) {
        ta.style.overflowY = 'hidden';
        resizeTextarea(ta);

        ta.addEventListener('input', function () {
            resizeTextarea(this);
        });
    });
});
