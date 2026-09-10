"""把 workflow_code/*.py 注入 workflow.yml，生成可导入 Dify 的最终文件。"""
import json
from pathlib import Path

WF_DIR = Path(__file__).resolve().parent
ROOT = WF_DIR.parent  # 决策模块根目录
CODE_DIR = ROOT / "workflow_code"
TEMPLATE = WF_DIR / "workflow.template.yml"
OUTPUT = WF_DIR / "workflow.yml"

NODE_FILES = {
    "1784611350176": "preprocess.py",
    "1783921220295": "parse_llm.py",
    "1783922686244": "normalize_status.py",
    "1783923227015": "format_output.py",
}


def yaml_double_quote(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)


def find_code_span(text: str, node_id: str):
    marker = f"id: '{node_id}'"
    id_pos = text.find(marker)
    if id_pos == -1:
        raise RuntimeError(f"找不到节点 id: {node_id}")

    before = text[:id_pos]
    open_marker = 'code: "'
    open_pos = before.rfind(open_marker)
    if open_pos == -1:
        raise RuntimeError(f"找不到 code 块: {node_id}")

    start = open_pos + len(open_marker)
    i = start
    while i < len(text):
        ch = text[i]
        if ch == "\\":
            i += 2
            continue
        if ch == '"':
            return open_pos, i + 1
        i += 1
    raise RuntimeError(f"code 块未闭合: {node_id}")


def inject_codes(yaml_text: str) -> str:
    for node_id in reversed(list(NODE_FILES.keys())):
        filename = NODE_FILES[node_id]
        code = (CODE_DIR / filename).read_text(encoding="utf-8")
        inner = json.dumps(code, ensure_ascii=False)[1:-1]

        open_pos, end_pos = find_code_span(yaml_text, node_id)
        content_start = open_pos + len('code: "')
        yaml_text = yaml_text[:content_start] + inner + yaml_text[end_pos - 1 :]

    return yaml_text


def main():
    if not TEMPLATE.exists():
        TEMPLATE.write_text(OUTPUT.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"已创建模板: {TEMPLATE}")

    yaml_text = TEMPLATE.read_text(encoding="utf-8")
    yaml_text = inject_codes(yaml_text)

    OUTPUT.write_text(yaml_text, encoding="utf-8")
    print(f"[OK] 生成工作流: {OUTPUT}")


if __name__ == "__main__":
    main()
