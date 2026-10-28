import json
import base64
import os
import urllib.request

def call_mcp(method, params, session_id=None):
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json, text/event-stream'}
    if session_id:
        headers['mcp-session-id'] = session_id
    req = urllib.request.Request(
        'http://127.0.0.1:9210/mcp',
        data=json.dumps({'jsonrpc': '2.0', 'id': 1, 'method': method, 'params': params}).encode(),
        headers=headers,
    )
    res = urllib.request.urlopen(req)
    sid = res.headers.get('mcp-session-id')
    raw = res.read().decode('utf-8')
    for line in raw.split('\n'):
        if line.startswith('data: '):
            data_str = line[6:].strip()
            if data_str:
                return json.loads(data_str), sid
    return None, sid

def main():
    init_res, sid = call_mcp('initialize', {'protocolVersion': '2024-11-05', 'capabilities': {}, 'clientInfo': {'name': 'antigravity', 'version': '1.0'}})

    # Scroll up into the response and capture
    js_scroll = """
    const page = browser.page(2);
    await page.scroll('up', 1500);
    await page.waitForTime(500);
    return true;
    """
    call_mcp('tools/call', {'name': 'run', 'arguments': {'code': js_scroll}}, session_id=sid)

    res, _ = call_mcp('tools/call', {
        'name': 'screenshot',
        'arguments': {
            'page': 2,
            'format': 'png',
            'fullPage': False,
        }
    }, session_id=sid)

    contents = res.get('result', {}).get('content', [])
    for c in contents:
        if c.get('type') == 'image':
            b64_data = c.get('data')
            out_path = r'C:\Users\shoun\.gemini\antigravity-ide\brain\90a3fca8-d685-484b-8083-e1b4aa8109fb\claude_verification_screenshot_top.png'
            with open(out_path, 'wb') as f:
                f.write(base64.b64decode(b64_data))
            print(f"Top screenshot saved to: {out_path}")
            return

if __name__ == '__main__':
    main()
