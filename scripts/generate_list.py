#!/usr/bin/env python3
"""
Dynamic Awesome AI Agent Harnesses Generator
Fetches open source AI agent harnesses from GitHub, filters by license/quality,
groups them dynamically by GitHub topics, and generates a formatted README.md.
"""

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import jinja2
import requests
import yaml

# Base paths
ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / "config"
TEMPLATES_DIR = ROOT_DIR / "templates"
DATA_DIR = ROOT_DIR / "data"

GITHUB_API_BASE = "https://api.github.com"


class GitHubClient:
    """Handles rate-limited and authenticated interaction with GitHub API."""

    def __init__(self, token: Optional[str] = None):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "Awesome-Agent-Harness-Bot/1.0",
            }
        )
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"
            print("[INFO] Authenticated with GitHub Token.")
        else:
            print("[WARN] No GITHUB_TOKEN detected. Running in unauthenticated mode (strict rate limits).")

    def _handle_rate_limit(self, response: requests.Response) -> bool:
        if response.status_code in (403, 429):
            remaining = response.headers.get("X-RateLimit-Remaining")
            reset = response.headers.get("X-RateLimit-Reset")
            if reset:
                reset_time = datetime.datetime.fromtimestamp(int(reset), tz=datetime.timezone.utc)
                print(f"[WARN] Rate limit reached! Remaining: {remaining}. Resets at: {reset_time.isoformat()}")
            return False
        return True

    def get_repo(self, full_name: str) -> Optional[Dict[str, Any]]:
        url = f"{GITHUB_API_BASE}/repos/{full_name}"
        try:
            resp = self.session.get(url, timeout=15)
            if not self._handle_rate_limit(resp):
                return None
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 404:
                print(f"[WARN] Repo {full_name} not found (404).")
            else:
                print(f"[WARN] Error fetching {full_name}: {resp.status_code}")
        except Exception as e:
            print(f"[ERROR] Exception fetching {full_name}: {e}")
        return None

    def search_repositories(self, query: str, per_page: int = 30, page: int = 1) -> List[Dict[str, Any]]:
        url = f"{GITHUB_API_BASE}/search/repositories"
        params = {
            "q": query,
            "sort": "stars",
            "order": "desc",
            "per_page": min(per_page, 100),
            "page": page,
        }
        try:
            resp = self.session.get(url, params=params, timeout=15)
            if not self._handle_rate_limit(resp):
                return []
            if resp.status_code == 200:
                data = resp.json()
                return data.get("items", [])
            else:
                print(f"[WARN] Search query '{query}' failed with status {resp.status_code}: {resp.text[:120]}")
        except Exception as e:
            print(f"[ERROR] Search query '{query}' exception: {e}")
        return []


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def format_star_count(stars: int) -> str:
    if stars >= 1_000_000:
        return f"{stars / 1_000_000:.1f}M"
    elif stars >= 1_000:
        return f"{stars / 1_000:.1f}k"
    return str(stars)


def escape_markdown(text: Optional[str]) -> str:
    if not text:
        return "No description provided."
    cleaned = text.replace("\r", " ").replace("\n", " ").replace("|", "\\|").strip()
    if len(cleaned) > 160:
        return cleaned[:157] + "..."
    return cleaned


def is_open_source(repo: Dict[str, Any], allowed_licenses: List[str]) -> bool:
    license_info = repo.get("license")
    if not license_info or not isinstance(license_info, dict):
        return False
    spdx_id = license_info.get("spdx_id")
    key = (license_info.get("key") or "").lower()

    if not spdx_id or spdx_id in ("NOASSERTION", "NONE"):
        # Fallback to key check for common open-source licenses
        if key in ("mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "gpl-3.0", "agpl-3.0"):
            return True
        return False

    if not allowed_licenses:
        return True

    allowed_lower = {l.lower() for l in allowed_licenses}
    if spdx_id.lower() in allowed_lower or key in allowed_lower:
        return True

    # Handle variants like GPL-3.0-only or Apache-2.0 matching gpl-3.0, etc.
    for allowed in allowed_lower:
        if spdx_id.lower().startswith(allowed):
            return True

    return False


def should_include_repo(repo: Dict[str, Any], filters: Dict[str, Any], seed_set: Set[str]) -> bool:
    full_name = repo.get("full_name", "")
    is_seed = full_name in seed_set

    # Check blacklist
    exclude_list = set(filters.get("exclude_repos", []))
    if full_name in exclude_list:
        return False

    # Check archived
    if not filters.get("include_archived", False) and repo.get("archived", False):
        return False

    # Check forks
    if not filters.get("include_forks", False) and repo.get("fork", False):
        return False

    # Check stars
    min_stars = filters.get("min_stars", 10)
    stars = repo.get("stars", repo.get("stargazers_count", 0))
    if not is_seed and stars < min_stars:
        return False

    # Check license
    if filters.get("require_license", True):
        allowed_licenses = filters.get("allowed_licenses", [])
        if not is_open_source(repo, allowed_licenses) and not is_seed:
            return False

    return True


def normalize_repo_data(repo: Dict[str, Any]) -> Dict[str, Any]:
    topics = repo.get("topics", [])
    if not isinstance(topics, list):
        topics = []
    topics = [t.lower().strip() for t in topics if t]

    license_info = repo.get("license") or {}
    license_name = license_info.get("spdx_id") or license_info.get("name")
    if not license_name or license_name in ("NOASSERTION", "NONE"):
        license_name = (license_info.get("key") or "Open Source").upper()

    stars = repo.get("stargazers_count", repo.get("stars", 0))
    raw_desc = repo.get("description", "")

    # Filter generic topics for the "Key Topics" column
    generic_topics = {
        "python",
        "machine-learning",
        "ai",
        "artificial-intelligence",
        "deep-learning",
        "llm",
        "large-language-models",
        "agent",
        "agents",
    }
    key_topics = [t for t in topics if t not in generic_topics][:4]
    if not key_topics and topics:
        key_topics = topics[:3]

    return {
        "id": repo.get("id"),
        "full_name": repo.get("full_name"),
        "name": repo.get("name"),
        "owner": repo.get("owner", {}).get("login") if isinstance(repo.get("owner"), dict) else repo.get("owner"),
        "html_url": repo.get("html_url"),
        "description": raw_desc or "No description provided.",
        "escaped_description": escape_markdown(raw_desc),
        "stars": stars,
        "formatted_stars": format_star_count(stars),
        "forks": repo.get("forks_count", repo.get("forks", 0)),
        "open_issues": repo.get("open_issues_count", repo.get("open_issues", 0)),
        "license": license_info,
        "license_name": license_name,
        "language": repo.get("language") or "Multi",
        "topics": topics,
        "top_topics": key_topics,
        "fork": repo.get("fork", False),
        "archived": repo.get("archived", False),
        "updated_at": repo.get("pushed_at") or repo.get("updated_at"),
    }


def categorize_repositories(
    repos: List[Dict[str, Any]], categories_cfg: List[Dict[str, Any]], fallback_cfg: Dict[str, Any]
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Groups repositories by category based on topic matching.
    Each repo is assigned to the category with highest match score.
    Returns (categorized_list, topic_index_list).
    """
    cat_map = {
        cat["id"]: {
            "id": cat["id"],
            "name": cat["name"],
            "emoji": cat.get("emoji", "📌"),
            "description": cat["description"],
            "topics": set(t.lower() for t in cat.get("topics", [])),
            "repos": [],
        }
        for cat in categories_cfg
    }

    fallback_id = fallback_cfg["id"]
    fallback_category = {
        "id": fallback_id,
        "name": fallback_cfg["name"],
        "emoji": fallback_cfg.get("emoji", "🚀"),
        "description": fallback_cfg["description"],
        "topics": set(),
        "repos": [],
    }

    topic_counts: Dict[str, int] = {}

    for repo in repos:
        repo_topics = set(repo["topics"])
        for t in repo_topics:
            topic_counts[t] = topic_counts.get(t, 0) + 1

        best_cat_id = None
        best_score = 0

        for cat_id, cat_data in cat_map.items():
            match_count = len(repo_topics.intersection(cat_data["topics"]))
            if match_count > best_score:
                best_score = match_count
                best_cat_id = cat_id

        if best_cat_id and best_score > 0:
            cat_map[best_cat_id]["repos"].append(repo)
        else:
            fallback_category["repos"].append(repo)

    # Sort repos within categories by star count
    for cat_data in cat_map.values():
        cat_data["repos"].sort(key=lambda x: x["stars"], reverse=True)
    fallback_category["repos"].sort(key=lambda x: x["stars"], reverse=True)

    # Filter out empty categories
    active_categories = [c for c in cat_map.values() if len(c["repos"]) > 0]
    if len(fallback_category["repos"]) > 0:
        active_categories.append(fallback_category)

    # Build topic index list sorted by frequency descending, then alphabetically
    topic_index = [
        {"topic": t, "count": count}
        for t, count in sorted(topic_counts.items(), key=lambda item: (-item[1], item[0]))
    ]

    return active_categories, topic_index


def main():
    parser = argparse.ArgumentParser(description="Generate dynamic Awesome AI Agent Harnesses list.")
    parser.add_argument("--dry-run", action="store_true", help="Run without writing files")
    parser.add_argument("--offline", action="store_true", help="Use local cached data/harnesses.json without calling GitHub API")
    parser.add_argument("--token", type=str, default=None, help="GitHub Personal Access Token")
    args = parser.parse_args()

    token = args.token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")

    settings_path = CONFIG_DIR / "settings.yaml"
    topics_path = CONFIG_DIR / "topics.yaml"
    template_path = TEMPLATES_DIR / "README.md.j2"
    output_readme = ROOT_DIR / "README.md"
    cache_file = DATA_DIR / "harnesses.json"

    print("[INFO] Loading configurations...")
    settings = load_yaml(settings_path)
    topics_cfg = load_yaml(topics_path)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    collected_repos: Dict[str, Dict[str, Any]] = {}

    # Load cache if available
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
                for r in cached_data:
                    norm = normalize_repo_data(r)
                    collected_repos[norm["full_name"]] = norm
            print(f"[INFO] Loaded {len(collected_repos)} repositories from local cache {cache_file}.")
        except Exception as e:
            print(f"[WARN] Failed to read existing cache: {e}")

    if not args.offline:
        client = GitHubClient(token)
        search_queries = settings.get("github", {}).get("search_queries", [])
        max_pages = settings.get("github", {}).get("max_pages_per_query", 2)
        per_page = settings.get("github", {}).get("per_page", 30)
        seed_repos = settings.get("seed_repos", [])

        # 1. Fetch seed repos directly
        print(f"[INFO] Fetching {len(seed_repos)} seed repositories...")
        for full_name in seed_repos:
            repo_raw = client.get_repo(full_name)
            if repo_raw:
                norm = normalize_repo_data(repo_raw)
                collected_repos[norm["full_name"]] = norm
            time.sleep(0.5)

        # 2. Search queries
        print(f"[INFO] Running {len(search_queries)} search queries...")
        for query in search_queries:
            print(f"  -> Query: '{query}'")
            for page in range(1, max_pages + 1):
                items = client.search_repositories(query, per_page=per_page, page=page)
                if not items:
                    break
                for item in items:
                    full_name = item.get("full_name")
                    if full_name:
                        collected_repos[full_name] = normalize_repo_data(item)
                # Polite delay between search requests
                time.sleep(2.0)

    filters = settings.get("filters", {})
    seed_set = set(settings.get("seed_repos", []))

    # Apply quality & license filters
    filtered_repos: List[Dict[str, Any]] = []
    for full_name, repo_data in collected_repos.items():
        if should_include_repo(repo_data, filters, seed_set):
            filtered_repos.append(repo_data)

    print(f"[INFO] Total candidate repos: {len(collected_repos)}, After filters: {len(filtered_repos)}")

    # Group by topics
    categories_cfg = topics_cfg.get("categories", [])
    fallback_cfg = topics_cfg.get("fallback_category", {
        "id": "emerging-harnesses",
        "name": "Emerging Harnesses",
        "emoji": "🚀",
        "description": "Discovered open source agent harnesses."
    })

    categories, topic_index = categorize_repositories(filtered_repos, categories_cfg, fallback_cfg)

    # Render template
    stats = {
        "total_repos": len(filtered_repos),
        "total_categories": len(categories),
        "total_topics": len(topic_index),
        "last_updated": datetime.datetime.now(datetime.timezone.utc).strftime("%Y--%m--%d"),
    }

    print("[INFO] Rendering README template...")
    with open(template_path, "r", encoding="utf-8") as f:
        template = jinja2.Template(f.read())

    rendered_markdown = template.render(
        title=settings.get("title", "Awesome Open Source AI Agent Harnesses"),
        description=settings.get("description", ""),
        categories=categories,
        topic_index=topic_index,
        stats=stats,
    )

    if args.dry_run:
        print("[INFO] Dry run complete! Categories found:")
        for c in categories:
            print(f"  - {c['emoji']} {c['name']}: {len(c['repos'])} repos")
        print(f"[INFO] Top 10 topics: {[t['topic'] for t in topic_index[:10]]}")
    else:
        # Write cache
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(filtered_repos, f, indent=2, ensure_ascii=False)
        print(f"[SUCCESS] Wrote structured data to {cache_file}")

        # Write README
        with open(output_readme, "w", encoding="utf-8") as f:
            f.write(rendered_markdown)
        print(f"[SUCCESS] Wrote generated Awesome List to {output_readme}")


if __name__ == "__main__":
    main()
