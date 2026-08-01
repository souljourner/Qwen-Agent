/**
 * ui-unstick.js — resume Chainlit's scroll/paint loop when the browser stalls it.
 *
 * Why: in Chainlit 2.11.1 both autoscroll and streamed-text painting piggyback
 * on the per-token React setState — nothing re-kicks them after the browser
 * suspends the loop (mobile Safari does this during touch momentum / when the
 * page loses its interaction heartbeat). Symptom: chat frozen until the user
 * taps the chat area; refresh always renders correctly (reload is a one-shot
 * REST fetch). This shim adds the missing "re-kick" listeners plus a
 * conservative stale-state recovery after connectivity loss.
 *
 * Debug: localStorage.setItem('uiUnstickDebug', '1')
 */
(function () {
  "use strict";

  var AT_BOTTOM_SLOP = 10;        // mirror the SPA's own at-bottom check
  var WATCHDOG_MS = 1500;         // follow-bottom cadence while streaming
  var STREAMING_WINDOW_MS = 3000; // recent-mutation window = "streaming"
  var STALE_AFTER_RECONNECT_MS = 10000;
  var MIN_OUTAGE_MS = 5000;

  var lastMutationAt = 0;
  var outageStartedAt = 0;
  var reloadedThisSession = false;

  function debug() {
    try {
      if (localStorage.getItem("uiUnstickDebug")) {
        console.log.apply(console, ["[ui-unstick]"].concat([].slice.call(arguments)));
      }
    } catch (e) { /* private mode */ }
  }

  function scroller() {
    // Chainlit's message ScrollContainer: the scrollable ancestor of the
    // messages list. Structure-based, so warn loudly if it disappears.
    var el = document.querySelector("#message-list, .messages-container, main [data-radix-scroll-area-viewport], main .overflow-y-auto");
    if (!el) {
      var candidates = document.querySelectorAll("main div");
      for (var i = 0; i < candidates.length; i++) {
        var c = candidates[i];
        if (c.scrollHeight > c.clientHeight + 50 &&
            /auto|scroll/.test(getComputedStyle(c).overflowY)) {
          return c;
        }
      }
    }
    return el;
  }

  function isAtBottom(el) {
    return el.scrollTop + el.clientHeight >= el.scrollHeight - AT_BOTTOM_SLOP;
  }

  var wasAtBottom = true;

  function nudge(reason) {
    var el = scroller();
    if (!el) {
      debug("no scroll container found (", reason, ") — selector drift?");
      return;
    }
    // A real scroll event re-evaluates the SPA's own autoScrollRef and, more
    // importantly, is a user-interaction signal that resumes the suspended
    // render loop on mobile Safari.
    el.dispatchEvent(new Event("scroll", { bubbles: true }));
    if (wasAtBottom) {
      el.scrollTop = el.scrollHeight;
    }
    debug("nudge:", reason, "atBottom=", wasAtBottom);
  }

  function trackScrollPosition() {
    var el = scroller();
    if (el) wasAtBottom = isAtBottom(el);
  }

  // --- mutation observer: streaming + stall detector -----------------------
  var observer = new MutationObserver(function () {
    lastMutationAt = Date.now();
  });

  function armObserver() {
    var target = document.querySelector("main") || document.body;
    observer.observe(target, { childList: true, subtree: true, characterData: true });
  }

  // --- resume kicks --------------------------------------------------------
  document.addEventListener("visibilitychange", function () {
    if (document.visibilityState === "visible") nudge("visible");
  });
  window.addEventListener("focus", function () { nudge("focus"); });
  window.addEventListener("pageshow", function () { nudge("pageshow"); });
  window.addEventListener("online", function () { nudge("online"); });
  document.addEventListener("touchend", function () {
    trackScrollPosition();
    nudge("touchend");
  }, { passive: true });
  document.addEventListener("pointerup", function () {
    trackScrollPosition();
  }, { passive: true });

  // --- streaming watchdog --------------------------------------------------
  setInterval(function () {
    trackScrollPosition();
    var streaming = Date.now() - lastMutationAt < STREAMING_WINDOW_MS;
    if (streaming && wasAtBottom) {
      var el = scroller();
      if (el) el.scrollTop = el.scrollHeight;
    }
  }, WATCHDOG_MS);

  // --- stale-state recovery after connectivity loss ------------------------
  function noteDisconnect() {
    if (!outageStartedAt) outageStartedAt = Date.now();
  }

  function maybeRecover() {
    if (!outageStartedAt || reloadedThisSession) return;
    var outageMs = Date.now() - outageStartedAt;
    outageStartedAt = 0;
    if (outageMs < MIN_OUTAGE_MS) return;
    // Wait for things to settle; reload ONLY if nothing is painting (a live
    // stream repaints via socket events and must never be interrupted).
    setTimeout(function () {
      var sinceMutation = Date.now() - lastMutationAt;
      if (sinceMutation > STALE_AFTER_RECONNECT_MS && !reloadedThisSession) {
        reloadedThisSession = true;
        debug("stale after", outageMs, "ms outage — soft reload");
        location.reload();
      } else {
        debug("reconnected but content is live — no reload");
      }
    }, STALE_AFTER_RECONNECT_MS);
  }

  window.addEventListener("offline", noteDisconnect);
  window.addEventListener("online", maybeRecover);

  // socket.io may run over WebSocket; observe closures too (polling mode is
  // covered by the offline/online pair above).
  var NativeWS = window.WebSocket;
  if (NativeWS) {
    window.WebSocket = function (url, protocols) {
      var ws = protocols !== undefined ? new NativeWS(url, protocols) : new NativeWS(url);
      ws.addEventListener("close", noteDisconnect);
      ws.addEventListener("open", maybeRecover);
      return ws;
    };
    window.WebSocket.prototype = NativeWS.prototype;
    Object.defineProperty(window.WebSocket, "CONNECTING", { value: NativeWS.CONNECTING });
    Object.defineProperty(window.WebSocket, "OPEN", { value: NativeWS.OPEN });
    Object.defineProperty(window.WebSocket, "CLOSING", { value: NativeWS.CLOSING });
    Object.defineProperty(window.WebSocket, "CLOSED", { value: NativeWS.CLOSED });
  }

  // --- boot ----------------------------------------------------------------
  function boot() {
    armObserver();
    trackScrollPosition();
    if (!scroller()) {
      console.warn("[ui-unstick] scroll container not found — Chainlit DOM may have changed; shim inactive until it appears");
    }
    debug("armed");
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
