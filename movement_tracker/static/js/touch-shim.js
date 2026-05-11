/**
 * Touch-to-mouse shim + pinch-zoom for iPad/touch devices.
 *
 * Usage: enableTouch(element, { onPinch: (scale, centerX, centerY) => {...} })
 *
 * - Forwards single-touch gestures to synthetic mousedown/mousemove/mouseup
 *   events so the existing mouse handlers Just Work.
 * - Captures pinch gestures (two fingers) and fires onPinch callback.
 * - Requires the element to have `touch-action: none` in CSS so the browser
 *   doesn't pre-empt our gestures for native scroll/zoom.
 *
 * Notes:
 * - We use pointer events under the hood so this works for stylus too.
 * - A single-tap (no movement) also fires a synthetic click.
 */
(function () {
    'use strict';

    window.enableTouch = function enableTouch(el, opts = {}) {
        if (!el) return;
        if (el.__touchShimInstalled) return;
        el.__touchShimInstalled = true;

        // Ensure CSS allows us to handle all touches
        if (!el.style.touchAction) el.style.touchAction = 'none';

        const activePointers = new Map();  // pointerId → { x, y, startX, startY, time }
        let pinchState = null;              // { startDist, lastDist, startCenter, lastCenter }
        const TAP_MAX_MOVE = 10;            // px
        const TAP_MAX_TIME = 500;           // ms

        function dispatchMouseLike(type, ev, overrides = {}) {
            const init = {
                bubbles: true,
                cancelable: true,
                view: window,
                button: 0,
                buttons: type === 'mouseup' ? 0 : 1,
                clientX: ev.clientX,
                clientY: ev.clientY,
                screenX: ev.screenX,
                screenY: ev.screenY,
                shiftKey: ev.shiftKey || false,
                ctrlKey: ev.ctrlKey || false,
                metaKey: ev.metaKey || false,
                altKey: ev.altKey || false,
                ...overrides,
            };
            // Dispatch on the actual target under the pointer so hit-testing is right.
            const target = (type === 'mousemove' || type === 'mouseup')
                ? document.elementFromPoint(ev.clientX, ev.clientY) || el
                : ev.target || el;
            target.dispatchEvent(new MouseEvent(type, init));
        }

        function dist(p1, p2) {
            const dx = p1.x - p2.x, dy = p1.y - p2.y;
            return Math.sqrt(dx * dx + dy * dy);
        }
        function center(p1, p2) {
            return { x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2 };
        }

        el.addEventListener('pointerdown', (ev) => {
            if (ev.pointerType !== 'touch' && ev.pointerType !== 'pen') return;
            activePointers.set(ev.pointerId, {
                x: ev.clientX, y: ev.clientY,
                startX: ev.clientX, startY: ev.clientY,
                time: performance.now(),
            });

            if (activePointers.size === 1) {
                // Start single-touch drag — forward as mousedown
                dispatchMouseLike('mousedown', ev);
            } else if (activePointers.size === 2) {
                // End any in-progress single-touch drag
                dispatchMouseLike('mouseup', ev);
                // Start pinch
                const pts = Array.from(activePointers.values());
                pinchState = {
                    startDist: dist(pts[0], pts[1]),
                    lastDist: dist(pts[0], pts[1]),
                    startCenter: center(pts[0], pts[1]),
                    lastCenter: center(pts[0], pts[1]),
                };
            }
            try { el.setPointerCapture(ev.pointerId); } catch {}
            ev.preventDefault();
        });

        el.addEventListener('pointermove', (ev) => {
            if (ev.pointerType !== 'touch' && ev.pointerType !== 'pen') return;
            if (!activePointers.has(ev.pointerId)) return;
            const p = activePointers.get(ev.pointerId);
            p.x = ev.clientX; p.y = ev.clientY;

            if (activePointers.size === 1) {
                dispatchMouseLike('mousemove', ev);
            } else if (activePointers.size === 2 && pinchState) {
                const pts = Array.from(activePointers.values());
                const d = dist(pts[0], pts[1]);
                const c = center(pts[0], pts[1]);
                const scale = d / pinchState.lastDist;
                if (opts.onPinch) {
                    opts.onPinch(scale, c.x, c.y, {
                        dx: c.x - pinchState.lastCenter.x,
                        dy: c.y - pinchState.lastCenter.y,
                    });
                }
                pinchState.lastDist = d;
                pinchState.lastCenter = c;
            }
            ev.preventDefault();
        });

        function endPointer(ev) {
            if (ev.pointerType !== 'touch' && ev.pointerType !== 'pen') return;
            const p = activePointers.get(ev.pointerId);
            if (!p) return;
            activePointers.delete(ev.pointerId);

            if (activePointers.size === 0) {
                // Final touch released → mouseup, and synth click if it was a tap
                dispatchMouseLike('mouseup', ev);
                const dt = performance.now() - p.time;
                const dx = ev.clientX - p.startX;
                const dy = ev.clientY - p.startY;
                if (dt < TAP_MAX_TIME && dx * dx + dy * dy < TAP_MAX_MOVE * TAP_MAX_MOVE) {
                    dispatchMouseLike('click', ev);
                }
                pinchState = null;
            } else if (activePointers.size === 1) {
                // Exited pinch — re-establish single-touch drag from the
                // remaining finger so the user can keep panning.
                pinchState = null;
                const remain = Array.from(activePointers.values())[0];
                dispatchMouseLike('mousedown', { clientX: remain.x, clientY: remain.y, target: el });
            }
            try { el.releasePointerCapture(ev.pointerId); } catch {}
        }
        el.addEventListener('pointerup', endPointer);
        el.addEventListener('pointercancel', endPointer);

        // Block native gestures (prevent page scroll/zoom while interacting)
        el.addEventListener('gesturestart', e => e.preventDefault());
        el.addEventListener('gesturechange', e => e.preventDefault());
        el.addEventListener('gestureend', e => e.preventDefault());
    };

    /** Shim wheel events from pinch gestures for browsers that don't emit them.
     *  Dispatches a synthetic WheelEvent based on pinch delta. */
    window.pinchAsWheel = function pinchAsWheel(el) {
        if (!el || el.__pinchAsWheelInstalled) return;
        el.__pinchAsWheelInstalled = true;

        enableTouch(el, {
            onPinch: (scale, cx, cy) => {
                // Convert pinch scale to wheel deltaY: bigger scale = zoom in = negative deltaY
                const deltaY = (1 - scale) * 100;
                const wheelEv = new WheelEvent('wheel', {
                    bubbles: true, cancelable: true,
                    clientX: cx, clientY: cy,
                    deltaY, deltaMode: 0,
                });
                el.dispatchEvent(wheelEv);
            },
        });
    };
})();
