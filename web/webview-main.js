// VSCode Webview 打包入口（esbuild）：web/ 共用代码 → dist/webview.js。
// 与浏览器入口（index.html + app.js）的唯一差异：样式在此以模块方式引入，
// 由 esbuild 抽出为 dist/webview.css，供 webEditor.ts 注入 CSP 允许的 URI。
import "./style.css";
import "./app.js";
