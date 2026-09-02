"""Playwright 布局自检：正文列与左右侧栏/TOC 的真溢出检测。

用法: python scripts/check_layout.py [宽度,宽度...] [页面路径片段...]
默认: 1024/1280/1536/1920 四档宽度，抽检各卷代表页。
"""
import glob
import sys

from playwright.sync_api import sync_playwright

BASE = "http://localhost:8123"

JS = """() => {
    const main = document.querySelector('main');
    if (!main) return {err: 1};
    const mb = main.getBoundingClientRect();
    let real = [];
    main.querySelectorAll('*').forEach(el => {
        const b = el.getBoundingClientRect();
        if (b.width > 0 && b.right > mb.right + 2 && getComputedStyle(el).position !== 'fixed') {
            let anc = el.parentElement, clipped = false;
            while (anc && anc !== main) {
                const o = getComputedStyle(anc).overflowX;
                if (o !== 'visible') { clipped = true; break; }
                anc = anc.parentElement;
            }
            if (!clipped) real.push(el.tagName + '+' + Math.round(b.right - mb.right) + 'px');
        }
    });
    const toc = document.querySelector('nav#TOC');
    let rightGap = null;
    if (toc) {
        let minL = Infinity;
        toc.querySelectorAll('li').forEach(li => {
            const b = li.getBoundingClientRect();
            if (b.width > 0) minL = Math.min(minL, b.left);
        });
        if (minL < Infinity) rightGap = Math.round(minL - mb.right);
    }
    return {n: real.length, ex: real.slice(0, 3), rightGap: rightGap, width: Math.round(mb.width)};
}"""


def main():
    args = sys.argv[1:]
    widths = [int(a) for a in args if a.isdigit()]
    keys = [a for a in args if not a.isdigit()]
    if not widths:
        widths = [1024, 1280, 1536, 1920]
    if keys:
        pages = [p.replace("\\", "/").replace("_book", "")
                 for p in glob.glob("_book/chapters/vol*/*.html")
                 if any(k in p for k in keys)]
    else:
        pages = [p.replace("\\", "/").replace("_book", "")
                 for p in glob.glob("_book/chapters/vol*/*.html")]

    bad = 0
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        for f in pages:
            for w in widths:
                page = browser.new_page(viewport={"width": w, "height": 900})
                ok = False
                for attempt in range(3):
                    try:
                        page.goto(BASE + f, timeout=15000)
                        page.wait_for_load_state("networkidle")
                        page.wait_for_timeout(300)
                        ok = True
                        break
                    except Exception:
                        page.wait_for_timeout(1500)
                if not ok:
                    print(f"X {f} @{w}: load fail (3 retries)")
                    bad += 1
                    page.close()
                    continue
                r = page.evaluate(JS)
                name = f.split("/")[-1]
                if r.get("err"):
                    print(f"X {name} @{w}: no <main>")
                    bad += 1
                elif r["n"]:
                    print(f"X {name} @{w}: {r['n']} real overflow {r['ex']}")
                    bad += 1
                elif r["rightGap"] is not None and r["rightGap"] < 0:
                    print(f"X {name} @{w}: TOC gap {r['rightGap']}px")
                    bad += 1
                page.close()
        browser.close()
    total = len(pages) * len(widths)
    print(f"\n{total} checked, {bad} bad" if bad else f"\n{total} checked, ALL PASS")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
