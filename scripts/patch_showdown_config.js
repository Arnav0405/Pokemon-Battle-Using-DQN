// Force the local Showdown config to bind loopback only.
// The example config ships bindaddress = '0.0.0.0'; with --no-security that would
// expose an unauthenticated battle server to the LAN.
const fs = require('fs');

const file = process.argv[2];
const before = "exports.bindaddress = '0.0.0.0';";
const after = "exports.bindaddress = '127.0.0.1';";

const src = fs.readFileSync(file, 'utf8');
if (src.includes(before)) {
  fs.writeFileSync(file, src.replace(before, after));
  console.log(`${file}: bindaddress -> 127.0.0.1`);
} else {
  console.log(`${file}: no 0.0.0.0 bindaddress to patch (already loopback?)`);
}
