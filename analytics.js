(function () {
// Replaced at build time with GA_MEASUREMENT_ID from .env (see scripts/build-analytics.js)
var MEASUREMENT_ID = '__GA_MEASUREMENT_ID__'; // e.g., 'G-XXXXXXXXXX'

// Analytics is disabled when no measurement ID was configured
if (!/^G-[A-Z0-9]+$/.test(MEASUREMENT_ID)) return;

// Prevent double-initialization
if (window.__gaInitialized) return;
window.__gaInitialized = true;

// Make the ID available as ANALYTICS_ID too (for legacy code that expects it)
if (typeof window.ANALYTICS_ID === 'undefined') {
window.ANALYTICS_ID = MEASUREMENT_ID;
}

// Setup dataLayer and gtag
window.dataLayer = window.dataLayer || [];
window.gtag = window.gtag || function () { window.dataLayer.push(arguments); };

// Minimal UA-style ga shim that forwards to gtag
if (!window.ga) {
(function () {
var defaults = {};      // fields set via ga('set', ...)
var createdIds = {};    // track ids passed to ga('create', ...)

  function mapFields(obj) {
    var out = {};
    if (!obj) return out;
    if (obj.page) out.page_path = obj.page;
    if (obj.location) out.page_location = obj.location;
    if (obj.title) out.page_title = obj.title;
    return out;
  }

  window.ga = function () {
    var args = Array.prototype.slice.call(arguments);
    var cmd = args[0];

    // ga(function(){ ... }) — run callback soon as possible
    if (typeof cmd === 'function') { try { cmd(); } catch (e) {} return; }

    // ga('create', 'ID', 'auto' [, name])
    if (cmd === 'create') {
      var id = args[1];
      if (id && !createdIds[id]) {
        createdIds[id] = true;
        // Avoid duplicate initial page_view from additional configs
        window.gtag('config', id, { send_page_view: false });
      }
      return;
    }

    // ga('set', fieldName, value) or ga('set', { ... })
    if (cmd === 'set') {
      var a1 = args[1], a2 = args[2];
      if (a1 && typeof a1 === 'object') {
        for (var k in a1) defaults[k] = a1[k];
      } else if (a1) {
        defaults[a1] = a2;
      }
      return;
    }

    // ga('send', 'pageview' [, page|fields])
    // ga('send', 'event', category, action [, label, value, fields])
    if (cmd === 'send') {
      var hitType = args[1];

      if (hitType === 'pageview') {
        var fields = {};
        if (typeof args[2] === 'string') fields.page = args[2];
        else if (args[2] && typeof args[2] === 'object') fields = args[2];

        var params = Object.assign({}, mapFields(defaults), mapFields(fields));
        window.gtag('event', 'page_view', params);
        return;
      }

      if (hitType === 'event') {
        var category = args[2];
        var action   = args[3];
        var label    = args[4];
        var value    = args[5];
        var extra    = (args[6] && typeof args[6] === 'object') ? args[6] : {};

        var params = Object.assign({}, defaults, extra);
        if (category != null && params.event_category == null) params.event_category = category;
        if (label != null && params.event_label == null)       params.event_label = label;
        if (typeof value !== 'undefined' && params.value == null) params.value = value;

        window.gtag('event', action || 'event', params);
        return;
      }

      // Other hit types are no-ops in this shim
      return;
    }

    // ga('require', ...) — no-op in this shim
    if (cmd === 'require') return;
  };

  // Match analytics.js timestamp property
  window.ga.l = +new Date();
})();
}

// Queue initial events
window.gtag('js', new Date());
window.gtag('config', MEASUREMENT_ID); // sends initial page_view by default

// Load the GA library
var s = document.createElement('script');
s.async = true;
s.src = 'https://www.googletagmanager.com/gtag/js?id=' + encodeURIComponent(MEASUREMENT_ID);
(document.head || document.body || document.documentElement).appendChild(s);
})();