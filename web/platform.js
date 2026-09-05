// 平台抽象：浏览器与 VSCode Webview 共用同一份 app.js。
// 运行时探测（acquireVsCodeApi 仅存在于 webview）；两个实现均静态导入、
// 内部惰性获取平台 API（browser 用 localStorage、webview 用 postMessage），
// 故同一份代码可被浏览器直服与 esbuild 打包进 webview。
import { platform as browserPlatform } from "./platform/browser.js";
import { platform as webviewPlatform } from "./platform/webview.js";

const isWebview = typeof acquireVsCodeApi === "function";

export const platform = isWebview ? webviewPlatform : browserPlatform;
