# llm_retry.py
import time
import random
import requests

def post_with_retry_forever(
    url: str,
    *,
    headers: dict,
    payload: dict,
    timeout: int = 120,
    min_sleep: float = 1.0,
    max_sleep: float = 30.0,
    backoff: float = 1.8,
    jitter: float = 0.3,
    session: requests.Session | None = None,
    print_prefix: str = "[LLM-RETRY]",
):
    """
    永远重试直到成功返回 2xx。
    - 对网络错误/SSL EOF/超时：直接重试
    - 对 429/5xx：重试
    - 对 4xx(非429)：一般是参数/鉴权问题，默认也重试（按你的要求“直到正常继续”）
      但会把响应内容打印出来，便于你发现根因
    """
    s = session or requests.Session()
    attempt = 0
    sleep_s = min_sleep

    while True:
        attempt += 1
        try:
            resp = s.post(url, headers=headers, json=payload, timeout=timeout)
            code = resp.status_code

            # 成功
            if 200 <= code < 300:
                return resp

            # 失败：打印一点信息
            text_preview = (resp.text or "")[:400].replace("\n", "\\n")
            print(f"{print_prefix} HTTP {code} attempt={attempt} resp={text_preview}")

            # 任何非2xx都重试（按你的“必须一直尝试”）
        except KeyboardInterrupt:
            print(f"{print_prefix} interrupted by user (Ctrl+C).")
            raise
        except (requests.exceptions.SSLError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.RequestException) as e:
            print(f"{print_prefix} EXC {type(e).__name__} attempt={attempt} err={e}")

        # 退避 + 抖动
        # jitter: 让多进程/多请求时别一起撞
        j = 1.0 + (random.random() * 2 - 1) * jitter
        time.sleep(max(0.0, min(sleep_s * j, max_sleep)))
        sleep_s = min(max_sleep, sleep_s * backoff)
