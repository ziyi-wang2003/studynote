#!/usr/bin/env python3
"""
Local dev server: build the site and serve it with live-reload on file changes.

Usage:
    python3 serve.py          # default port 8080
    python3 serve.py 3000     # custom port
"""

import sys
import time
import threading
import http.server
import functools
from pathlib import Path
from build import build, CONTENT_DIR, TEMPLATE_DIR, STATIC_DIR, OUTPUT_DIR

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
WATCH_DIRS = [CONTENT_DIR, TEMPLATE_DIR, STATIC_DIR]


def get_mtime_map():
    mtimes = {}
    for d in WATCH_DIRS:
        if not d.exists():
            continue
        for f in d.rglob('*'):
            if f.is_file():
                mtimes[f] = f.stat().st_mtime
    return mtimes


def watcher():
    """Poll for file changes and rebuild."""
    last = get_mtime_map()
    while True:
        time.sleep(1)
        current = get_mtime_map()
        if current != last:
            changed = [
                str(f.relative_to(Path(__file__).parent))
                for f in set(current) ^ set(last)
            ]
            if not changed:
                changed = [
                    str(f.relative_to(Path(__file__).parent))
                    for f in current
                    if current.get(f) != last.get(f)
                ]
            print(f'\n  Changed: {", ".join(changed[:5])}')
            print('  Rebuilding...', end=' ', flush=True)
            try:
                build()
                print('done.')
            except Exception as e:
                print(f'ERROR: {e}')
            last = get_mtime_map()


def main():
    # Initial build (local, no base URL prefix)
    print('Building site...')
    build()
    print(f'\n  Serving at http://127.0.0.1:{PORT}/')
    print('  Watching for changes... (Ctrl+C to stop)\n')

    # Start file watcher in background
    t = threading.Thread(target=watcher, daemon=True)
    t.start()

    # Serve docs/
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler,
        directory=str(OUTPUT_DIR),
    )
    server = http.server.HTTPServer(('127.0.0.1', PORT), handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nStopped.')


if __name__ == '__main__':
    main()
