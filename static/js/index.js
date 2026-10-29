// Crafter project page — minimal interactivity (no dependencies)
(function () {
  // Mobile nav burger
  document.querySelectorAll('.navbar-burger').forEach(function (burger) {
    burger.addEventListener('click', function () {
      var target = document.getElementById(burger.dataset.target);
      burger.classList.toggle('is-active');
      if (target) target.classList.toggle('is-active');
      burger.setAttribute('aria-expanded', burger.classList.contains('is-active'));
    });
  });
  // Close the mobile menu after choosing an anchor
  document.querySelectorAll('.navbar-menu .navbar-item').forEach(function (link) {
    link.addEventListener('click', function () {
      var menu = link.closest('.navbar-menu');
      var burger = document.querySelector('.navbar-burger');
      if (menu) menu.classList.remove('is-active');
      if (burger) burger.classList.remove('is-active');
    });
  });

  // Copy buttons (BibTeX, quick start)
  document.querySelectorAll('.copy-btn[data-copy-target]').forEach(function (btn) {
    btn.addEventListener('click', function () {
      var src = document.getElementById(btn.dataset.copyTarget);
      if (!src) return;
      var text = src.innerText;
      var done = function () {
        var original = btn.innerHTML;
        btn.classList.add('copied');
        btn.innerHTML = '<i class="fas fa-check"></i> Copied';
        setTimeout(function () { btn.classList.remove('copied'); btn.innerHTML = original; }, 1600);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done).catch(function () { fallback(text, done); });
      } else {
        fallback(text, done);
      }
    });
  });
  function fallback(text, done) {
    var ta = document.createElement('textarea');
    ta.value = text; ta.setAttribute('readonly', '');
    ta.style.position = 'absolute'; ta.style.left = '-9999px';
    document.body.appendChild(ta); ta.select();
    try { document.execCommand('copy'); done(); } catch (e) {}
    document.body.removeChild(ta);
  }

  // Animate result bars on first view
  if ('IntersectionObserver' in window) {
    var fills = document.querySelectorAll('.bar-fill');
    fills.forEach(function (f) { f.dataset.v = f.style.getPropertyValue('--v'); f.style.setProperty('--v', 0); });
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) {
          e.target.querySelectorAll('.bar-fill').forEach(function (f) { f.style.setProperty('--v', f.dataset.v); });
          io.unobserve(e.target);
        }
      });
    }, { threshold: 0.3 });
    document.querySelectorAll('.bars-group').forEach(function (g) { io.observe(g); });
  }
})();
