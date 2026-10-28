"""
Verify FL-IDS implementation with Claude via BrowserOS neo.
"""

import json
import time
import urllib.request

VERIFICATION_PROMPT = """Please review and verify the complete implementation and benchmark results for our Federated Intrusion Detection for 6G Edge (FL-IDS) project based on requirements.md:

1. WORK COMPLETED:
- **Phase 0 (Leakage-Free Benchmark Engine)**:
  * Strict fit-on-train scaling (LeakageFreePreprocessor), placeholder canonicalization (NaN/Inf), deduplication before split, and disjoint server root set extraction.
  * Weak Gaussian Naive Bayes canary detector (canary.py) verifying that splits don't suffer from serialisation/label leakage artifacts (addressing the Edge-IIoTset audit). Canary test passed: Balanced Acc: 0.5341, Macro-F1: 0.5188.
  * Dirichlet non-IID partitioning (alpha in {0.1, 0.3, 1.0}) with rare-class preserving allocation and atypical client telemetry.
  * Schema-accurate synthetic NetFlow v2 generator with 10 attack classes and rare minority classes (Worms, SQL Injection, Fingerprinting).
- **Poisoning Threat Models (fl_ids/attacks/)**:
  * Byzantine Sign-Flip, Label-Flip, Additive Gaussian Noise, Additive Trigger Backdoor with magnitude sweep and update boosting, and Defense-aware Adaptive attacker with norm bounding.
- **Novel Defense (ClassAwareTrustDefense)**:
  * Class-Conditional Trust Decomposition: Evaluates updates against class-conditional sub-gradient anchors to prevent atypical clients holding rare attacks from being filtered out.
  * Two-Level Adaptive Clipping: Client-level norm clipping bounds boosted backdoors, and server-side aggregate clipping prevents malicious drift.
  * Root Magnitude Normalization: Rescales client delta directions against clean root scale, neutralizing model replacement / boosted updates.
- **Baselines Implemented**:
  * FedAvg, Coordinate-wise Trimmed-Mean, Coordinate Median, Multi-Krum, FoolsGold, FLTrust, RepGuard.
  * Sketch Compression: Top-k sketch (20x) and 1-bit sign sketch (32x) compatible with robust aggregation.
- **Automated Verification**:
  * 19/19 pytest tests passing in 12.6s (preprocessor, canary, partitioner, attacks, aggregators, simulation).
- **Head-to-head Benchmark Matrix Results (10 clients, Dirichlet alpha=0.3 non-IID)**:
  * Under Clean: ClassAwareTrust achieves 0.5320 Macro-F1 / 0.5240 Bal Acc (vs FedAvg 0.4817, Multi-Krum 0.2571, FLTrust 0.1851).
  * Under Backdoor: FedAvg collapses to 0.2539 Macro-F1, while ClassAwareTrust maintains 0.5314 Macro-F1 with 100% atypical client acceptance.
  * Under Adaptive Attacker: ClassAwareTrust achieves 0.5351 Macro-F1 / 0.5203 Bal Acc and lowest ASR (0.8000).
  * Edge Profile: Single-sample inference latency is 0.10-0.31 ms (beating the <= 0.8 ms gateway target). Footprint is 14,538 parameters (~58 KB).

Please evaluate:
1. Verdict: Is this implementation on track with the requirements and research targets?
2. Critical critique of the results and threat model coverage.
3. Verification of whether the research gaps (trigger-agnostic backdoor mitigation, minority-attack recall, leakage-free integrity) are properly addressed.
4. Next 3 recommended experiment steps (e.g. running real NetFlow CSVs, formal DP accounting sweep)."""


def call_mcp(method, params, session_id=None):
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if session_id:
        headers["mcp-session-id"] = session_id
    req = urllib.request.Request(
        "http://127.0.0.1:9210/mcp",
        data=json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode(),
        headers=headers,
    )
    res = urllib.request.urlopen(req)
    sid = res.headers.get("mcp-session-id")
    raw = res.read().decode("utf-8")
    for line in raw.split("\n"):
        if line.startswith("data: "):
            data_str = line[6:].strip()
            if data_str:
                return json.loads(data_str), sid
    return None, sid


def main():
    print("1. Connecting to BrowserOS neo on port 9210...")
    init_res, sid = call_mcp(
        "initialize",
        {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "antigravity", "version": "1.0"},
        },
    )
    print("Connected. Session ID:", sid)

    # Step 1: Snapshot Claude page
    print("2. Snapshotting active Claude tab (page 2)...")
    js_snapshot = """
    const page = browser.page(2);
    const snap = await page.snapshot({ mode: 'interactive' });
    return { text: snap.text.slice(-2500) };
    """
    res, _ = call_mcp("tools/call", {"name": "run", "arguments": {"code": js_snapshot}}, session_id=sid)
    content = res["result"]["content"][0]["text"]
    print("Found Claude tab.")

    # Step 2: Post the prompt into Claude
    print("3. Typing verification prompt into Claude and submitting...")
    prompt_escaped = json.dumps(VERIFICATION_PROMPT)
    js_submit = f"""
    const page = browser.page(2);
    const snap = await page.snapshot({{ mode: 'interactive' }});
    
    // Find the prompt textbox
    // Typically contenteditable or textarea
    await page.evaluate((promptText) => {{
        const el = document.querySelector('div[contenteditable="true"]') || document.querySelector('textarea') || document.querySelector('[role="textbox"]');
        if (el) {{
            el.focus();
            if (el.tagName === 'TEXTAREA') {{
                el.value = promptText;
                el.dispatchEvent(new Event('input', {{ bubbles: true }}));
            }} else {{
                el.innerText = promptText;
                el.dispatchEvent(new Event('input', {{ bubbles: true }}));
            }}
            return 'Filled prompt successfully';
        }}
        return 'Textbox not found';
    }}, {prompt_escaped});

    await page.waitForTime(1000);

    // Click the send button
    const submitResult = await page.evaluate(() => {{
        const btn = document.querySelector('button[aria-label="Send Message"]') || document.querySelector('button[aria-label="Send message"]') || document.querySelector('button[aria-label*="Send"]');
        if (btn && !btn.disabled) {{
            btn.click();
            return 'Send button clicked';
        }}
        return 'Send button not ready or disabled: ' + (btn ? 'disabled=' + btn.disabled : 'not found');
    }});

    return {{ submitResult }};
    """

    res, _ = call_mcp("tools/call", {"name": "run", "arguments": {"code": js_submit}}, session_id=sid)
    print("Submission status:", json.dumps(res["result"], indent=2))

    # Step 3: Wait for Claude to generate its verification response
    print("4. Waiting for Claude to complete its response...")
    time.sleep(15)

    for attempt in range(8):
        print(f"Checking Claude response progress (poll {attempt+1}/8)...")
        js_read = """
        const page = browser.page(2);
        const md = await page.read();
        return { length: md.length, latest: md.slice(-4000) };
        """
        read_res, _ = call_mcp("tools/call", {"name": "run", "arguments": {"code": js_read}}, session_id=sid)
        val = read_res.get("result", {}).get("structuredContent", {}).get("value", {})
        latest_text = val.get("latest", "")
        
        # Check if Claude finished or responded to FL-IDS
        if "FL-IDS" in latest_text or "ClassAwareTrust" in latest_text or "Verdict:" in latest_text:
            print("\n" + "=" * 80)
            print("CLAUDE VERIFICATION RESPONSE RECEIVED:")
            print("=" * 80)
            print(latest_text)
            print("=" * 80)
            return

        time.sleep(10)

    print("Claude response reading finished.")


if __name__ == "__main__":
    main()
