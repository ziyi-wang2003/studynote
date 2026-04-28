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

    // Smooth scroll for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(function (a) {
        a.addEventListener('click', function (e) {
            var target = document.querySelector(this.getAttribute('href'));
            if (target) {
                e.preventDefault();
                var offset = 80;
                var top = target.getBoundingClientRect().top + window.scrollY - offset;
                window.scrollTo({ top: top, behavior: 'smooth' });
                history.pushState(null, null, this.getAttribute('href'));
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
