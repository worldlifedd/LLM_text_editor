// esbuild 打包脚本：src/extension.ts → dist/extension.js（外部化 vscode 模块）
const esbuild = require("esbuild");
const fs = require("fs");
const path = require("path");

const args = process.argv.slice(2);
const watch = args.includes("--watch");

// 把 Python 无头服务端复制进扩展目录，随 VSIX 一起发布：
// 插件自包含，安装后无需外部 server.py 即可自动拉起服务。
// 仓库根目录为单一事实来源，改动根文件后重新 compile 即可同步。
const PY_FILES = ["server.py", "core.py", "backend.py", "skills.py", "requirements.txt"];
function copyPython() {
  const root = path.resolve(__dirname, "..");
  const dest = path.join(__dirname, "python");
  fs.mkdirSync(dest, { recursive: true });
  for (const f of PY_FILES) {
    fs.copyFileSync(path.join(root, f), path.join(dest, f));
  }
  const skillsSrc = path.join(root, "skills");
  if (fs.existsSync(skillsSrc)) {
    fs.cpSync(skillsSrc, path.join(dest, "skills"), { recursive: true });
  }
  console.log("[bundle] python/ 已同步");
}
copyPython();

const opts = {
  entryPoints: ["src/extension.ts"],
  bundle: true,
  outfile: "dist/extension.js",
  external: ["vscode"],
  format: "cjs",
  platform: "node",
  target: "node18",
  sourcemap: false,
  minify: false,
  logLevel: "info",
};

if (watch) {
  esbuild.context(opts).then((ctx) => {
    ctx.watch();
    console.log("[esbuild] watching for changes...");
  });
} else {
  esbuild.build(opts).catch(() => process.exit(1));
}
