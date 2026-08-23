// esbuild 打包脚本：src/extension.ts → dist/extension.js（外部化 vscode 模块）
const esbuild = require("esbuild");

const args = process.argv.slice(2);
const watch = args.includes("--watch");

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
