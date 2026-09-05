// esbuild 打包脚本：src/extension.ts → dist/extension.js（外部化 vscode 模块）
const esbuild = require("esbuild");
const fs = require("fs");
const path = require("path");

const args = process.argv.slice(2);
const watch = args.includes("--watch");

// 把 Python 无头服务端复制进扩展目录，随 VSIX 一起发布：
// 插件自包含，安装后无需外部 server.py 即可自动拉起服务。
// 仓库根目录为单一事实来源；python/ 为构建产物（git 不跟踪，见根 .gitignore），
// compile / vsce package（vscode:prepublish）时清空重建，勿手改。
const PY_FILES = ["server.py", "core.py", "backend.py", "skills.py", "requirements.txt"];
function copyPython() {
  const root = path.resolve(__dirname, "..");
  const dest = path.join(__dirname, "python");
  try {
    fs.rmSync(dest, { recursive: true, force: true });
  } catch (e) {
    // python/ 被占用（如 server.py 正在运行）时跳过重建；
    // 开发态用 gte.serverScript 指向仓库根 server.py，不依赖此副本。
    console.warn(`[bundle] python/ 删除失败，跳过同步：${e.code}`);
    return;
  }
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
  // web/index.html 以文本内联（webEditor.ts 引用，webview 标记单一事实来源）
  loader: { ".html": "text" },
  sourcemap: false,
  minify: false,
  logLevel: "info",
};

// Webview 前端：web/webview-main.js（共用 web/ 代码 + 平台抽象层）→
// dist/webview.js + dist/webview.css（style.css 由 esbuild 自动抽出）。
// 浏览器版不经此构建（server.py 直接静态托管 web/ 的 ES Modules）。
const webviewOpts = {
  entryPoints: [path.resolve(__dirname, "..", "web", "webview-main.js")],
  bundle: true,
  outfile: "dist/webview.js",
  format: "iife",
  platform: "browser",
  target: "es2020",
  sourcemap: false,
  minify: false,
  logLevel: "info",
};

if (watch) {
  Promise.all([esbuild.context(opts), esbuild.context(webviewOpts)]).then(([ctx1, ctx2]) => {
    ctx1.watch();
    ctx2.watch();
    console.log("[esbuild] watching for changes...");
  });
} else {
  Promise.all([esbuild.build(opts), esbuild.build(webviewOpts)]).catch(() => process.exit(1));
}
