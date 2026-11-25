import {spawnSync} from 'node:child_process';
import process from 'node:process';

const dir = process.argv[2] ?? '/frames';
const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';
const res = spawnSync(pythonCmd, ['order.py', dir], {stdio: 'inherit'});
if (res.error) {
  // Fallback if python not found: simple fallback
  import('./order.py');
}
