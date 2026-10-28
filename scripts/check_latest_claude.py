import json
import os
import subprocess
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.read_claude_verdict import call_mcp

def main():
    init_res, sid = call_mcp('initialize', {'protocolVersion': '2024-11-05', 'capabilities': {}, 'clientInfo': {'name': 'antigravity', 'version': '1.0'}})

    js_code = """
    const page = browser.page(2);
    const snap = await page.snapshot({ mode: 'interactive' });
    const lines = snap.text.split('\\n');
    const articles = lines.filter(l => l.includes('article "Message'));
    const copyBtns = lines.filter(l => l.includes('button "Copy"'));
    return { articles, copyBtns };
    """

    res, _ = call_mcp('tools/call', {'name': 'run', 'arguments': {'code': js_code}}, session_id=sid)
    val = res.get('result', {}).get('structuredContent', {}).get('value', {})
    print("Articles:", val.get('articles'))
    print("Copy buttons:", val.get('copyBtns'))

    copy_btns = val.get('copyBtns', [])
    if len(copy_btns) > 0:
        last_btn = copy_btns[-1]
        import re
        m = re.search(r'ref=(e\d+)', last_btn)
        if m:
            ref = m.group(1)
            print(f"Clicking latest copy button: {ref}")
            js_click = f"""
            const page = browser.page(2);
            await page.click('{ref}');
            return true;
            """
            call_mcp('tools/call', {'name': 'run', 'arguments': {'code': js_click}}, session_id=sid)
            cb = subprocess.check_output(['powershell', '-Command', 'Get-Clipboard'], text=True)
            print("=" * 80)
            print("COPIED TEXT FROM CLAUDE:")
            print("=" * 80)
            print(cb[:3000])
            print("=" * 80)
            with open(r'C:\Users\shoun\.gemini\antigravity-ide\brain\90a3fca8-d685-484b-8083-e1b4aa8109fb\claude_latest_review.md', 'w', encoding='utf-8') as f:
                f.write(cb)
            print("Saved to claude_latest_review.md")

if __name__ == '__main__':
    main()
