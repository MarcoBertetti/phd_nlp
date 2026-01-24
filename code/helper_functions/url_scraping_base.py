import os
import random
import logging
import asyncio
import aiohttp
import async_timeout
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from urllib.parse import urlparse
import undetected_chromedriver as uc
from selenium.common.exceptions import TimeoutException, WebDriverException

###############################################################################
# CONFIG
###############################################################################
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_3) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/109.0",
    "Mozilla/5.0 (Linux; Android 12; SM-G991B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110 Mobile Safari/537.36",
]

BROWSER_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}

MAX_DOMAIN_FAILURES = 100       # relaxed threshold
DOMAIN_CHECK_TIMEOUT = 3
BLACKLIST_PATH = "domain_blacklist.txt"

domain_failures = {}
domain_blacklist = set()

###############################################################################
# HELPERS
###############################################################################
def get_domain(url):
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return None


def load_blacklist():
    if os.path.exists(BLACKLIST_PATH):
        with open(BLACKLIST_PATH) as f:
            domain_blacklist.update(line.strip() for line in f if line.strip())
        print(f"Loaded {len(domain_blacklist)} blacklisted domains.")


def save_blacklist():
    with open(BLACKLIST_PATH, "w") as f:
        for d in sorted(domain_blacklist):
            f.write(d + "\n")

###############################################################################
# ASYNC FETCH
###############################################################################
async def async_fetch(session, url, timeout=10, max_retries=2, proxies=None):
    if proxies is None:
        proxies = []

    dom = get_domain(url)
    if not dom or dom in domain_blacklist:
        return (None, None, 499)

    for attempt in range(max_retries):
        headers = {**BROWSER_HEADERS, "User-Agent": random.choice(USER_AGENTS)}
        proxy_to_use = random.choice(proxies) if proxies else None

        try:
            with async_timeout.timeout(timeout):
                async with session.get(
                    url,
                    headers=headers,
                    ssl=False,
                    allow_redirects=True,
                    proxy=proxy_to_use
                ) as resp:
                    code = resp.status
                    if code == 200:
                        html = await resp.text(errors="ignore")
                        soup = BeautifulSoup(html, "html.parser")
                        header = soup.title.string.strip() if (soup.title and soup.title.string) else "No Title"
                        body = ' '.join(p.get_text(strip=True) for p in soup.find_all('p'))
                        return (header, body, 200)
                    elif code == 403:
                        return (None, None, 403)
                    else:
                        return (None, None, code)

        except Exception as e:
            logging.debug(f"[Async] {dom} failed ({e})")
            domain_failures[dom] = domain_failures.get(dom, 0) + 1
            if domain_failures[dom] >= MAX_DOMAIN_FAILURES:
                domain_blacklist.add(dom)
            if attempt == max_retries - 1:
                return (None, None, 403)
            await asyncio.sleep(random.uniform(0.5, 1.0))

    return (None, None, 403)


async def async_fetch_all(urls, concurrency=200, timeout=4, max_retries=3, proxies=None):
    sem = asyncio.Semaphore(concurrency)
    results = {}

    async with aiohttp.ClientSession() as session:
        async def fetch_with_sem(u):
            async with sem:
                return await async_fetch(session, u, timeout, max_retries, proxies)

        tasks = {u: asyncio.create_task(fetch_with_sem(u)) for u in urls}
        with tqdm(total=len(tasks), desc="Async fetch", dynamic_ncols=True) as pbar:
            for url, t in tasks.items():
                header, body, code = await t
                results[url] = (header, body, code)
                pbar.update(1)
    return results


def run_async_phase(urls, concurrency=200, timeout=4, max_retries=3, proxies=None):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(
        async_fetch_all(urls, concurrency, timeout, max_retries, proxies)
    )

###############################################################################
# SELENIUM FALLBACK
###############################################################################
selenium_semaphore = threading.Semaphore(2)
thread_local_data = threading.local()

def get_or_create_driver_headless(timeout=30):
    if not hasattr(thread_local_data, "driver"):
        acquired = selenium_semaphore.acquire(timeout=timeout)
        if not acquired:
            return None
        options = uc.ChromeOptions()
        options.headless = True
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument(f"--user-agent={random.choice(USER_AGENTS)}")
        try:
            driver = uc.Chrome(options=options, suppress_welcome=True)
            driver.set_page_load_timeout(5)
            driver.set_script_timeout(5)
            thread_local_data.driver = driver
        except Exception as e:
            logging.error(f"Failed to create Chrome driver: {e}")
            selenium_semaphore.release()
            return None
    return thread_local_data.driver


def release_driver():
    if hasattr(thread_local_data, "driver"):
        driver = thread_local_data.driver
        try:
            driver.quit()
        except Exception as e:
            logging.error(f"Error quitting driver: {e}")
        selenium_semaphore.release()
        del thread_local_data.driver


def process_url_selenium(url, timeout=30):
    header_text = "Error"
    body_text = "Error"
    dom = get_domain(url)
    if dom in domain_blacklist:
        return (header_text, body_text)
    driver = get_or_create_driver_headless(timeout=timeout)
    if driver is None:
        return (header_text, body_text)
    try:
        driver.get(url)
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, "html.parser")
        header_text = soup.title.string.strip() if (soup.title and soup.title.string) else "No Title"
        body_text = ' '.join(p.get_text(strip=True) for p in soup.find_all('p'))
    except Exception:
        domain_failures[dom] = domain_failures.get(dom, 0) + 1
        if domain_failures[dom] >= MAX_DOMAIN_FAILURES:
            domain_blacklist.add(dom)
    finally:
        return (header_text, body_text)


def parallel_selenium_fallback(urls, max_selenium_workers=2):
    if not urls:
        return {}
    results = {}
    progress_bar = tqdm(total=len(urls), desc="Selenium fallback", dynamic_ncols=True)
    lock = threading.Lock()

    def worker(sub_urls):
        local_dict = {}
        try:
            for u in sub_urls:
                hh, bb = process_url_selenium(u)
                with lock:
                    progress_bar.update(1)
                local_dict[u] = (hh, bb)
        finally:
            release_driver()
        return local_dict

    n = max_selenium_workers
    chunk_size = max(1, len(urls) // n)
    sublists = [urls[i:i + chunk_size] for i in range(0, len(urls), chunk_size)]
    with ThreadPoolExecutor(max_workers=n) as executor:
        for fut in as_completed([executor.submit(worker, sl) for sl in sublists]):
            results.update(fut.result())
    progress_bar.close()
    return results

###############################################################################
# WRITER
###############################################################################
def safe_text(x):
    if not isinstance(x, str):
        return "Error"
    # remove surrogate escapes or invalid chars
    return x.encode("utf-8", "ignore").decode("utf-8", "ignore")

def write_parquet_chunk(chunk_urls, final_results, output_path):
    df = pd.DataFrame({
        "url": chunk_urls,
        "header": [final_results.get(u, ("Error",))[0] for u in chunk_urls],
        "body": [final_results.get(u, ("Error", "Error"))[1] for u in chunk_urls]
    })
    df["header"] = df["header"].apply(safe_text)
    df["body"] = df["body"].apply(safe_text)
    df.to_parquet(output_path)

###############################################################################
# MAIN ORCHESTRATOR
###############################################################################
def process_urls_in_chunks(
    urls,
    chunk_size=1000,
    concurrency=200,
    timeout=4,
    max_retries=3,
    proxies=None,
    chunk_id=0,
    max_selenium_workers=2,
    output_dir="parquet_output",
    fallback_mode="async_and_selenium"
):
    os.makedirs(output_dir, exist_ok=True)
    load_blacklist()

    total_urls = len(urls)
    print(f"Total URLs to process: {total_urls}")

    start_idx = 0
    while start_idx < total_urls:
        end_idx = min(start_idx + chunk_size, total_urls)
        chunk_urls = [u for u in urls[start_idx:end_idx] if get_domain(u) not in domain_blacklist]
        print(f"\n=== Processing chunk {chunk_id} ({len(chunk_urls)} URLs after blacklist) ===")

        final_results = {}
        success_count = 0

        if fallback_mode == "selenium_only":
            selenium_res = parallel_selenium_fallback(chunk_urls, max_selenium_workers)
            final_results.update(selenium_res)
            success_count = sum(1 for v in selenium_res.values() if v[0] != "Error")

        else:
            async_results = run_async_phase(chunk_urls, concurrency, timeout, max_retries, proxies)
            fallback_urls = []
            for url, (hdr, bod, code) in async_results.items():
                if code == 200:
                    final_results[url] = (hdr, bod)
                    success_count += 1
                elif code == 403 and fallback_mode == "async_and_selenium":
                    fallback_urls.append(url)
                else:
                    final_results[url] = ("Error", "Error")

            print(f"  -> {success_count} succeeded, {len(fallback_urls)} need Selenium fallback")

            if fallback_mode == "async_and_selenium" and fallback_urls:
                selenium_res = parallel_selenium_fallback(fallback_urls, max_selenium_workers)
                for u, (hh, bb) in selenium_res.items():
                    final_results[u] = (hh, bb)
                selenium_successes = sum(1 for v in selenium_res.values() if v[0] != "Error")
                success_count += selenium_successes

        print(f"  -> Final: {success_count} succeeded, {len(final_results) - success_count} failed")
        output_path = os.path.join(output_dir, f"chunk_{chunk_id}.parquet")
        write_parquet_chunk(chunk_urls, final_results, output_path)
        print(f"Saved chunk {chunk_id} -> {output_path}")

        save_blacklist()
        chunk_id += 1
        start_idx = end_idx

    print("\n✅ All chunks processed!")
