// Inlines the engine and results into page.html.
// Writes index.html (open it directly) and artifact.html (same page, no doctype, for publishing).
const fs = require('node:fs');
const path = require('node:path');

const read = (name) => fs.readFileSync(path.join(__dirname, name), 'utf8');
let resultsJs;
try {
  resultsJs = read('results.js');
} catch {
  resultsJs = `window.RESULTS = ${read('results.quick.json')};\n`;
  console.warn('results.js not found, using results.quick.json');
}
const page = read('page.html')
  .replace('<script src="engine.js"></script>', () => `<script>\n${read('engine.js')}</script>`)
  .replace('<script src="results.js"></script>', () => `<script>\n${resultsJs}</script>`);

fs.writeFileSync(path.join(__dirname, 'artifact.html'), page);
fs.writeFileSync(path.join(__dirname, 'index.html'), `<!doctype html>\n<html lang="en">\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n<style>[hidden]{display:none!important}body{margin:0}</style>\n${page}\n</html>\n`);
console.log(`built index.html and artifact.html (${Math.round(page.length / 1024)} KB)`);
