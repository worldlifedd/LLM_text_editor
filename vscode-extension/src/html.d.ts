// esbuild 以 text loader 导入 web/index.html（webview 标记单一事实来源）。
declare module "*.html" {
  const content: string;
  export default content;
}
