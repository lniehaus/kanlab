// Copies analytics.js to dist/, filling in GA_MEASUREMENT_ID from the
// environment or, if unset there, from .env. Without an ID analytics stays off.
var fs = require('fs');
var path = require('path');

var root = path.join(__dirname, '..');

function readDotEnv(file) {
  var vars = {};
  if (!fs.existsSync(file)) return vars;
  fs.readFileSync(file, 'utf8').split(/\r?\n/).forEach(function (line) {
    var m = line.match(/^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$/);
    if (m) vars[m[1]] = m[2].replace(/^(['"])(.*)\1$/, '$2');
  });
  return vars;
}

var id = process.env.GA_MEASUREMENT_ID || readDotEnv(path.join(root, '.env')).GA_MEASUREMENT_ID || '';
if (id && !/^G-[A-Z0-9]+$/.test(id)) {
  console.error('error: GA_MEASUREMENT_ID must look like G-XXXXXXXXXX, got "' + id + '"');
  process.exit(1);
}

var src = fs.readFileSync(path.join(root, 'analytics.js'), 'utf8');
var out = path.join(root, 'dist', 'analytics.js');
fs.mkdirSync(path.dirname(out), { recursive: true });
fs.writeFileSync(out, src.replace('__GA_MEASUREMENT_ID__', id));
console.log(id ? 'analytics: using measurement ID ' + id : 'analytics: GA_MEASUREMENT_ID not set, analytics disabled');
