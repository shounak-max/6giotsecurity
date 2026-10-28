import json
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

    js_code = """
    const page = browser.page(2);
    // Use CDP to evaluate expression in page context
    const cdpRes = await page.cdp('Runtime.evaluate', {
        expression: `
            (() => {
                const prose = document.querySelectorAll('.prose, [class*="prose"], div[data-is-streaming]');
                if (prose.length > 0) {
                    return prose[prose.length - 1].innerText;
                }
                const headings = Array.from(document.querySelectorAll('h3, h4'));
                const h = headings.find(el => el.innerText.includes('Complete vs. provisional') || el.innerText.includes('Verdict'));
                if (h) {
                    let parent = h;
                    while (parent && parent.innerText.length < 500) {
                        parent = parent.parentElement;
                    }
                    return parent ? parent.innerText : h.innerText;
                }
                return document.body.innerText;
            })()
        `,
        returnByValue: true
    });
    return cdpRes;
    """

    res, _ = call_mcp('tools/call', {'name': 'run', 'arguments': {'code': js_code}}, session_id=sid)
    val = res.get('result', {}).get('structuredContent', {}).get('value', {})
    print("=" * 80)
    print("CDP RUNTIME RESULT:")
    print("=" * 80)
    text = val.get('result', {}).get('value', '')
    print(text)
    print("=" * 80)

if __name__ == '__main__':
    main()
