// Appends a content hash (?v=...) to the local asset URLs in dist/index.html.
// index.html is never cached, so each deploy makes browsers and the Cloudflare
// cache fetch the new bundle.css/bundle.js instead of serving stale copies.
var fs = require('fs');
var path = require('path');
var crypto = require('crypto');

var dist = path.join(__dirname, '..', 'dist');
var assets = ['bundle.css', 'bundle.js', 'lib.js', 'analytics.js'];

var htmlFile = path.join(dist, 'index.html');
var html = fs.readFileSync(htmlFile, 'utf8');

assets.forEach(function (name) {
  var hash = crypto.createHash('sha256')
    .update(fs.readFileSync(path.join(dist, name)))
    .digest('hex').slice(0, 10);
  var pattern = new RegExp('((?:href|src)=")' + name.replace('.', '\\.') + '"', 'g');
  if (!pattern.test(html)) {
    console.error('error: no reference to ' + name + ' found in index.html');
    process.exit(1);
  }
  html = html.replace(pattern, '$1' + name + '?v=' + hash + '"');
  console.log('cache-bust: ' + name + '?v=' + hash);
});

fs.writeFileSync(htmlFile, html);
