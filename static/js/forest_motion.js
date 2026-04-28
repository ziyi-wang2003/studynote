/* Forest Motion — ambient creature layer
   Usage:
     <div id="forest-layer" aria-hidden="true"></div>
     <link rel="stylesheet" href="forest_motion.css">
     <script src="forest_motion.js"></script>
   Options (set on window before load):
     window.FOREST_OPTIONS = {
       fireflies: 10, leaves: 5, motes: 14, butterflyEveryMs: 60000,
       foxPeek: true, leafAssetBase: '/static/creatures/'
     };
*/
(function(){
  const opts = Object.assign({
    fireflies: 10,
    leaves: 4,
    motes: 14,
    butterflyEveryMs: 55000,
    foxPeek: false,
    leafAssetBase: '/static/creatures/'
  }, window.FOREST_OPTIONS || {});

  function rand(min, max){ return min + Math.random() * (max - min); }
  function pick(arr){ return arr[Math.floor(Math.random()*arr.length)]; }

  function mount(){
    let layer = document.getElementById('forest-layer');
    if (!layer) {
      layer = document.createElement('div');
      layer.id = 'forest-layer';
      layer.setAttribute('aria-hidden', 'true');
      document.body.appendChild(layer);
    }
    layer.innerHTML = '';

    // Fireflies — drifting via JS setTransform for organic paths
    const flies = [];
    for (let i = 0; i < opts.fireflies; i++) {
      const el = document.createElement('div');
      el.className = 'fl-firefly';
      el.style.setProperty('--wink', rand(2.2, 4.6) + 's');
      el.style.animationDelay = rand(0, 4) + 's';
      layer.appendChild(el);
      flies.push({
        el,
        ax: rand(5, 95), ay: rand(15, 85),
        rx: rand(40, 120), ry: rand(20, 80),
        sx: rand(0.00004, 0.00012), sy: rand(0.00006, 0.00014),
        ox: rand(0, Math.PI*2), oy: rand(0, Math.PI*2)
      });
    }

    // Falling leaves
    const leafAssets = ['leaf-oak.svg', 'leaf-maple.svg'];
    for (let i = 0; i < opts.leaves; i++) {
      const el = document.createElement('div');
      el.className = 'fl-leaf';
      el.style.left = rand(0, 100) + 'vw';
      el.style.top = rand(-20, -5) + 'vh';
      el.style.setProperty('--dx', rand(-180, 180) + 'px');
      el.style.setProperty('--dur', rand(26, 46) + 's');
      el.style.backgroundImage = `url('${opts.leafAssetBase}${pick(leafAssets)}')`;
      el.style.animationDelay = rand(-30, 0) + 's';
      el.style.width = rand(22, 32) + 'px';
      el.style.height = el.style.width;
      layer.appendChild(el);
    }

    // Pollen / dust motes
    for (let i = 0; i < opts.motes; i++) {
      const el = document.createElement('div');
      el.className = 'fl-mote';
      el.style.left = rand(0, 100) + 'vw';
      el.style.top = rand(60, 110) + 'vh';
      el.style.setProperty('--mdx', rand(-50, 50) + 'px');
      el.style.setProperty('--mdur', rand(18, 34) + 's');
      el.style.animationDelay = rand(-20, 0) + 's';
      layer.appendChild(el);
    }

    // Fox peek
    if (opts.foxPeek) {
      const fox = document.createElement('div');
      fox.className = 'fl-fox-peek';
      document.body.appendChild(fox);
    }

    // Butterfly — spawn on interval
    function spawnButterfly(){
      const el = document.createElement('div');
      el.className = 'fl-butterfly';
      el.style.backgroundImage = `url('${opts.leafAssetBase}butterfly.svg')`;
      el.style.setProperty('--bdur', rand(18, 30) + 's');
      layer.appendChild(el);
      setTimeout(() => el.remove(), 32000);
    }
    if (opts.butterflyEveryMs) {
      setTimeout(spawnButterfly, 3000);
      setInterval(spawnButterfly, opts.butterflyEveryMs);
    }

    // Animate fireflies on requestAnimationFrame
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    let t0 = performance.now();
    function tick(now){
      const dt = now - t0;
      for (const f of flies) {
        const x = f.ax + Math.sin(f.ox + dt * f.sx) * (f.rx/10);
        const y = f.ay + Math.cos(f.oy + dt * f.sy) * (f.ry/10);
        f.el.style.left = x + 'vw';
        f.el.style.top = y + 'vh';
      }
      requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', mount);
  } else {
    mount();
  }
})();
