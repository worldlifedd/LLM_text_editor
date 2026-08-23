# -*- coding: utf-8 -*-
"""Agent Skill 加载与解析。

支持两种主流格式，统一解析为 Skill(name, description, instructions)：
1. Anthropic/Claude 风格 SKILL.md：YAML frontmatter（name/description）+ markdown 指令体，
   目录结构 skills/<skill-name>/SKILL.md
2. 普通 markdown 提示词文件：无 frontmatter 时 name=文件名，全文为 instructions
"""
import os
import shutil
from dataclasses import dataclass

import yaml

SKILLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills")


@dataclass
class Skill:
    name: str
    description: str
    instructions: str
    path: str = ""


def _parse_frontmatter(text: str):
    """解析 `---` 包裹的 YAML frontmatter，返回 (meta, body)。"""
    text = text.lstrip("\ufeff")
    if not text.startswith("---"):
        return {}, text
    parts = text.split("\n---", 2)
    if len(parts) < 2:
        return {}, text
    meta_raw = parts[0][3:].strip()
    body = parts[1].lstrip("\n") if len(parts) >= 2 else ""
    # 去掉 body 开头可能的孤立 '---' 残留
    if body.startswith("---"):
        body = body[3:].lstrip("\n")
    try:
        meta = yaml.safe_load(meta_raw) or {}
    except yaml.YAMLError:
        meta = {}
    if not isinstance(meta, dict):
        meta = {}
    return meta, body


def parse_skill_file(path: str) -> Skill:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    meta, body = _parse_frontmatter(text)
    name = str(meta.get("name") or os.path.splitext(os.path.basename(path))[0])
    description = str(meta.get("description") or "")
    return Skill(name=name, description=description, instructions=body.strip(), path=path)


def scan_skills(skills_dir: str = SKILLS_DIR) -> list:
    """扫描技能目录。优先取 <dir>/<name>/SKILL.md，其次取顶层 *.md。"""
    skills = []
    if not os.path.isdir(skills_dir):
        return skills
    for entry in sorted(os.listdir(skills_dir)):
        full = os.path.join(skills_dir, entry)
        skill_md = os.path.join(full, "SKILL.md")
        if os.path.isdir(full) and os.path.isfile(skill_md):
            skills.append(parse_skill_file(skill_md))
        elif os.path.isfile(full) and entry.lower().endswith(".md"):
            skills.append(parse_skill_file(full))
    return skills


def skills_to_context(enabled_skills: list) -> str:
    """把启用的技能拼接为系统级前置指令文本。"""
    if not enabled_skills:
        return ""
    parts = []
    for skill in enabled_skills:
        header = f"# 技能指令: {skill.name}"
        if skill.description:
            header += f"\n# 说明: {skill.description}"
        parts.append(f"{header}\n{skill.instructions}")
    return "\n\n".join(parts)


def import_skill_file(src_path: str, skills_dir: str = SKILLS_DIR) -> str:
    """把上传的 .md 复制进技能库目录，返回目标路径。"""
    os.makedirs(skills_dir, exist_ok=True)
    basename = os.path.basename(src_path)
    if basename.lower() == "skill.md":
        # 以所在临时目录名命名，避免互相覆盖
        basename = (os.path.basename(os.path.dirname(src_path)) or "skill") + ".md"
    dst = os.path.join(skills_dir, basename)
    i = 1
    while os.path.exists(dst):
        stem, ext = os.path.splitext(basename)
        dst = os.path.join(skills_dir, f"{stem}_{i}{ext}")
        i += 1
    shutil.copy(src_path, dst)
    return dst
