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
from openrouter.errors import OpenRouterError

from agent import create_article_agent

load_dotenv()

webhook_url = os.environ["WEBHOOK_URL"]

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
    "TechRadar Security": "https://www.techradar.com/feeds/tag/security",

    # AI-security specific
    "AI Incident Database": "https://incidentdatabase.ai/rss.xml",
    "Simon Willison": "https://simonwillison.net/atom/everything/",
    "Embrace The Red": "https://embracethered.com/blog/index.xml",

    # Vendor research and statements
    "Kaspersky Securelist": "https://securelist.com/feed/",
    "Snyk": "https://snyk.io/blog/feed/",
    "Trend Micro": "https://newsroom.trendmicro.com/cyberthreat?pagetemplate=rss",
    "Google Security Blog": "https://security.googleblog.com/feeds/posts/default",
    "OpenAI News": "https://openai.com/news/rss.xml",
    
    # General tech (noisy - rely on the AI filter)
    "Fast Company Tech": "https://www.fastcompany.com/technology/rss",
    "Jalopnik Tech": "https://jalopnik.com/category/tech/feed",
}


def clean(text):
    """Remove HTML tags, decode entities, and normalize whitespace."""
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def post_to_discord(source, title, link, review, published, webhook_url):
    """Send a structured article card and raise if Discord rejects it."""
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
                "color": 0xE74C3C,
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
        response.raise_for_status()


def read_feed(source, url):
    """Download a feed and return its entries, or warn and return none."""
    try:
        feed = feedparser.parse(url, agent="Mozilla/5.0 (SecurityNewsBot)")
    except OSError as error:
        print(f"[!] Could not read {source}: {error}")
        return []
    if feed.get("status", 200) >= 400 or (feed.bozo and not feed.entries):
        print(
            f"[!] Could not read {source}: "
            f"HTTP {feed.get('status', 'unknown')}, parse error: {feed.bozo}"
        )
        return []
    return feed.entries


def process_article(item, source, seen, first_run, agent, now, webhook_url):
    """Filter and post one article, leaving failed links eligible for retry.

    Failed articles remain unseen. Provider 404/429 errors propagate so
    the run can save progress and exit with a failure status.
    """
    link = item.get("link")
    if not link or link in seen:
        return
    seen.add(link)

    published_time = item.get("published_parsed") or item.get("updated_parsed")
    if published_time and (
        now - calendar.timegm(published_time) > MAX_AGE_HOURS * 3600
    ):
        return

    title = clean(item.get("title", ""))
    if first_run:
        return
    content = clean(
        " ".join(part.get("value", "") for part in item.get("content", []))
    ) or clean(item.get("summary", ""))

    try:
        response = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Title: {title}\nContent: {content}",
                    }
                ],
            }
        )
        review = response["structured_response"]
    except Exception as error:
        seen.discard(link)
        if not isinstance(error, OpenRouterError):
            raise
        print(f"[!] AI check failed for {link}: {type(error).__name__}")
        if error.status_code in (404, 429):
            raise
        return

    if not review.is_ai_security:
        return
    if review.impact != "high":
        return

    published = (
        time.strftime("%b %d %Y", published_time) if published_time else ""
    )
    try:
        post_to_discord(source, title, link, review, published, webhook_url)
    except requests.RequestException as error:
        # Exception messages can contain the secret webhook URL.
        print(f"[!] Failed to post {link}: {type(error).__name__}")
        seen.discard(link)
        return
    finally:
        time.sleep(1)


def main():
    """Run one polling cycle, save handled links, and return an exit status."""
    first_run = not SEEN_FILE.exists()
    seen = (
        set()
        if first_run
        else set(json.loads(SEEN_FILE.read_text(encoding="utf-8")))
    )
    
    agent = create_article_agent() if not first_run else None
    now = time.time()

    try:
        for source, url in FEEDS.items():
            for item in reversed(read_feed(source, url)):  # oldest first
                process_article(
                    item, source, seen, first_run, agent, now, webhook_url
                )
    except OpenRouterError:
        print("[!] Stopping; pending articles will retry next run.")
        return 1
    finally:
        SEEN_FILE.write_text(json.dumps(list(seen)), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
