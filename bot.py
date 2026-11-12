import calendar
import html
import json
import os
import re
import time
from pathlib import Path

import feedparser
import requests
from dotenv import load_dotenv

from agent import create_article_agent

load_dotenv()

SEEN_FILE = Path("seen.json")
MAX_AGE_HOURS = 48

FEEDS = {
    # Security news
    "The Hacker News": "https://thehackernews.com/feeds/posts/default",
    "BleepingComputer": "https://www.bleepingcomputer.com/feed/",
    "SecurityWeek": "https://www.securityweek.com/feed/",
    "Dark Reading": "https://www.darkreading.com/rss.xml",
    "The Register": "https://www.theregister.com/security/headlines.atom",
    "Wired": "https://www.wired.com/feed/category/security/latest/rss",
    "The Record": "https://therecord.media/feed",
    # AI-security specific
    "AI Incident Database": "https://incidentdatabase.ai/rss.xml",
    "Simon Willison": "https://simonwillison.net/atom/everything/",
    "Embrace The Red": "https://embracethered.com/blog/index.xml",
}


def clean(text):
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def post_to_discord(source, title, link, review, published, webhook_url):
    details = [
        ("Summary", review.summary),
        ("Affected", review.affected),
        ("Attack / Issue", review.attack),
        (
            "Impact",
            f"{review.impact.title()}: {review.impact_details}",
        ),
        ("Source", f"[{source}]({link}) · {published}"),
    ]
    payload = {
        "embeds": [
            {
                "title": title[:256],
                "url": link,
                "color": 0xE74C3C if review.impact == "high" else 0xF39C12,
                "fields": [
                    {
                        "name": name,
                        "value": (
                            value[:899] + "…" if len(value) > 900 else value
                        )
                        or "Not specified in source.",
                    }
                    for name, value in details
                ],
            }
        ],
    }
    with requests.post(webhook_url, json=payload, timeout=15) as response:
        print(f"[DEBUG] Discord replied: {response.status_code}")
        response.raise_for_status()


def main():
    print("[DEBUG] Starting news check", flush=True)
    first_run = not SEEN_FILE.exists()
    seen = (
        set()
        if first_run
        else set(json.loads(SEEN_FILE.read_text(encoding="utf-8")))
    )
    print(
        f"[DEBUG] Loaded {len(seen)} handled links from {SEEN_FILE.resolve()}"
    )
    print(f"[DEBUG] First run: {first_run}")
    print(f"[DEBUG] Webhook configured: {bool(os.environ.get('WEBHOOK_URL'))}")
    webhook_url = os.environ["WEBHOOK_URL"]
    agent = create_article_agent() if not first_run else None
    now = time.time()
    posted = 0

    for source, url in FEEDS.items():
        print(f"[DEBUG] Reading {source}: {url}", flush=True)
        try:
            feed = feedparser.parse(url, agent="Mozilla/5.0 (SecurityNewsBot)")
        except OSError as error:
            print(f"[!] Could not read {source}: {error}")
            continue
        if feed.get("status", 200) >= 400 or (feed.bozo and not feed.entries):
            print(
                f"[!] Could not read {source}: "
                f"HTTP {feed.get('status', 'unknown')}, "
                f"parse error: {feed.bozo}"
            )
            continue

        print(f"[DEBUG] {source}: found {len(feed.entries)} articles")
        for item in reversed(feed.entries):  # oldest first
            link = item.get("link")
            if not link:
                print("[DEBUG] Skipped article without a link")
                continue
            if link in seen:
                print(f"[DEBUG] Skipped already handled: {link}")
                continue
            seen.add(link)

            published_time = item.get("published_parsed") or item.get(
                "updated_parsed"
            )
            if published_time and (
                now - calendar.timegm(published_time) > MAX_AGE_HOURS * 3600
            ):
                print(
                    f"[DEBUG] Skipped older than {MAX_AGE_HOURS} hours: {link}"
                )
                continue

            title = clean(item.get("title", ""))
            content = clean(
                " ".join(
                    part.get("value", "") for part in item.get("content", [])
                )
            ) or clean(item.get("summary", ""))
            if first_run:
                print(
                    f"[DEBUG] First run, remembered without posting: {title}"
                )
                continue

            print(
                f"[DEBUG] Checking AI-security relevance: {title}", flush=True
            )
            try:
                response = agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": (
                                    f"Title: {title}\nContent: {content}"
                                ),
                            }
                        ],
                    }
                )
                review = response["structured_response"]
            except Exception as error:
                message = getattr(error, "body", None) or str(error)
                retry_after = getattr(error, "headers", {}).get("retry-after")
                if retry_after:
                    message += f"\nRetry-After: {retry_after}"
                for name in ("OPENROUTER_API_KEY", "WEBHOOK_URL"):
                    secret = os.environ.get(name)
                    if secret:
                        message = message.replace(secret, "[redacted]")
                print(
                    f"[!] AI check failed for {link}: "
                    f"{type(error).__name__}: {message}"
                )
                seen.discard(link)
                print("[DEBUG] Removed from seen; will retry on the next run")
                if getattr(error, "status_code", None) in (404, 429):
                    SEEN_FILE.write_text(
                        json.dumps(list(seen)), encoding="utf-8"
                    )
                    print(
                        "[!] Stopping; pending articles will retry next run."
                    )
                    return
                continue
            print(f"[DEBUG] AI-security related: {review.is_ai_security}")
            if not review.is_ai_security:
                print(f"[DEBUG] Skipped unrelated article: {title}")
                continue
            print(f"[DEBUG] Assessed impact: {review.impact}")
            if review.impact not in ("high", "medium"):
                print(f"[DEBUG] Skipped low or unclear impact: {title}")
                continue

            published = (
                time.strftime("%b %d %Y", published_time)
                if published_time
                else ""
            )
            try:
                print(
                    f"[DEBUG] Posting to Discord: {source}: {title}",
                    flush=True,
                )
                post_to_discord(
                    source, title, link, review, published, webhook_url
                )
                posted += 1
                print(f"[DEBUG] Posted: {link}")
            except requests.RequestException as error:
                # Exception messages can contain the secret webhook URL.
                print(f"[!] Failed to post {link}: {type(error).__name__}")
                seen.discard(link)
                print("[DEBUG] Removed from seen; will retry on the next run")
            time.sleep(1)

    SEEN_FILE.write_text(json.dumps(list(seen)), encoding="utf-8")
    print(f"[DEBUG] Saved {len(seen)} handled links to {SEEN_FILE}")
    print(f"[DEBUG] Done. Posted {posted} articles")


if __name__ == "__main__":
    main()
