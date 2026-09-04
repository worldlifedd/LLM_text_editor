// prefill_ppl 合并 + reconcile 覆盖性测试（web/model.js 与 vscode state.ts 双实现）。
//
// 回归背景：
// 1. reconcile 的前缀分支（覆盖是 base 的严格前缀，如用户在着色块末尾追加
//    文本）此前不补灰段 → 与后续段拼接出现字符空洞，生成时 join 不再是
//    content 的前缀 → 整块着色消失。
// 2. prefill_ppl（服务端对活动块手动编辑文本的逐 token 打分）合并逻辑：
//    公共前缀保留旧着色、prefill 覆盖的尾部替换为新鲜打分、空洞补灰。
//
// 用法：node tests/test_prefill_merge.mjs

import path from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.dirname(here);
const bundleDir = path.join(root, ".esbuild-test");

// vscode 侧为 TS，需 esbuild 打包（总是重打包：缓存会掩盖源码改动）
execFileSync(
  process.execPath,
  [
    path.join(root, "vscode-extension/node_modules/esbuild/bin/esbuild"),
    path.join(root, "vscode-extension/src/state.ts"),
    "--bundle", "--format=esm", "--log-level=error",
    `--outdir=${bundleDir}`, "--out-extension:.js=.mjs",
  ],
  { stdio: "inherit" }
);

const web = await import(pathToFileURL(path.join(root, "web/model.js")).href);
const ts = await import(pathToFileURL(path.join(bundleDir, "state.mjs")).href);

let fail = 0;
function check(name, got, want) {
  const ok = JSON.stringify(got) === JSON.stringify(want);
  if (!ok) fail++;
  console.log(`${ok ? "  OK  " : "  FAIL"} ${name}`);
  if (!ok) console.log(`         实际 ${JSON.stringify(got)}\n         期望 ${JSON.stringify(want)}`);
}

for (const [label, m] of [["web/model.js", web], ["vscode/state.ts", ts]]) {
  console.log(`\n== ${label} ==`);
  // 统一两套 API：reconcile → PplSeg 形态
  const rec = (t, p, base) => {
    if ("reconcilePpl" in m) {
      const [t2, p2] = m.reconcilePpl(t, p, base);
      return { token_texts: t2, token_ppls: p2 };
    }
    return m.reconcileActivePpl(t, p, base);
  };
  const merge = (b, a, d) => m.mergePrefillPpl(b, a, d);
  const cov = (seg) => seg.token_texts.join("");

  // 1. reconcile：完全覆盖原样返回
  let r = rec(["秋", "风"], [1.2, 3.4], "秋风");
  check("reconcile 全覆盖", [cov(r), r.token_ppls], ["秋风", [1.2, 3.4]]);

  // 2. reconcile：覆盖是严格前缀（末尾追加文本）→ 补灰段覆盖整个 base
  r = rec(["秋"], [1.2], "秋风起");
  check("reconcile 前缀追加补灰", [cov(r), r.token_ppls], ["秋风起", [1.2, null]]);

  // 3. reconcile：中点编辑 → 保留公共前缀完整 token，其后灰段
  r = rec(["秋风", "起了"], [1.2, 3.4], "秋风来了");
  check("reconcile 中点编辑", [cov(r), r.token_ppls], ["秋风来了", [1.2, null]]);

  // 4. merge：prefill 全覆盖（活动块全新文本、无旧着色）
  let b = merge({ token_texts: [], token_ppls: [] }, "秋风起",
    { token_texts: ["秋", "风", "起"], token_ppls: [5, 6, 7] });
  check("merge 全覆盖", [cov(b), b.token_ppls], ["秋风起", [5, 6, 7]]);

  // 5. merge：后缀覆盖 + 旧着色前缀保留（用户只在尾部追加）
  b = merge({ token_texts: ["秋风"], token_ppls: [1.2] }, "秋风起了",
    { token_texts: ["起", "了"], token_ppls: [8, 9] });
  check("merge 尾部追加", [cov(b), b.token_ppls], ["秋风起了", [1.2, 8, 9]]);

  // 6. merge：中点编辑 → 跨编辑点旧 token 丢弃，prefill 尾部替换
  b = merge({ token_texts: ["秋风", "起了"], token_ppls: [1.2, 3.4] }, "秋风来了",
    { token_texts: ["来", "了"], token_ppls: [8, 9] });
  check("merge 中点编辑", [cov(b), b.token_ppls], ["秋风来了", [1.2, 8, 9]]);

  // 7. merge：基线带灰段（此前编辑过）+ prefill 覆盖灰段 → 替换为真实打分
  b = merge({ token_texts: ["秋风", "起了"], token_ppls: [1.2, null] }, "秋风起了",
    { token_texts: ["起", "了"], token_ppls: [8, 9] });
  check("merge 灰段替换", [cov(b), b.token_ppls], ["秋风起了", [1.2, 8, 9]]);

  // 8. merge：数据不是 activeText 后缀 → null（丢弃）
  check("merge 失配丢弃",
    merge({ token_texts: [], token_ppls: [] }, "秋风",
      { token_texts: ["风", "起"], token_ppls: [1, 2] }) === null, true);

  // 9. merge：空数据 → null
  check("merge 空数据",
    merge({ token_texts: [], token_ppls: [] }, "秋风", { token_texts: [], token_ppls: [] }) === null, true);

  // 10. 不变量：merge 结果与后续生成 token 拼接后 join 必须等于完整文本
  b = merge({ token_texts: ["秋风"], token_ppls: [1.2] }, "秋风起了",
    { token_texts: ["起", "了"], token_ppls: [8, 9] });
  const full = {
    token_texts: [...b.token_texts, "。"],
    token_ppls: [...b.token_ppls, 2],
  };
  check("merge+生成 不变量", cov(full), "秋风起了。");
}

console.log(fail ? `\n${fail} FAILED` : "\nALL PREFILL MERGE TESTS PASSED");
process.exit(fail ? 1 : 0);
