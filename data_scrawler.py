#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
通过 Wikipedia API 递归抓取体育相关条目（包含子分类），
输出 sports_wiki.jsonl
每行格式: {"text": "..."}
"""

import requests
import json
import time
from pathlib import Path
import re
from collections import deque

API_URL = "https://en.wikipedia.org/w/api.php"

HEADERS = {
    # 建议把邮箱改成你自己的，方便 Wikipedia 那边真出事了能联系到你
    "User-Agent": "SportsWikiDataCollector/0.2 (2284605489@qq.com)"
}

# 起始体育相关分类（可以继续加）
ROOT_CATEGORIES = [
    # --- General sports / organization / competitions ---
    "Category:Sports",
    "Category:Sports by type",
    "Category:Sports by country",
    "Category:Sports competitions",
    "Category:Sports leagues",
    "Category:Sports clubs and teams",
    "Category:Sports venues",
    "Category:Sports governing bodies",
    "Category:Sports people",
    "Category:Sport of athletics",

    # --- Major multi-sport events ---
    "Category:Olympic Games",
    "Category:Paralympic Games",
    "Category:World Games",
    "Category:Commonwealth Games",
    "Category:Asian Games",
    "Category:Pan American Games",
    "Category:African Games",

    # --- Football (soccer) ---
    "Category:Association football",
    "Category:Association football competitions",
    "Category:Association football clubs",
    "Category:Association football players",
    "Category:FIFA",
    "Category:FIFA World Cup",
    "Category:UEFA",
    "Category:UEFA Champions League",
    "Category:Premier League",
    "Category:La Liga",
    "Category:Serie A",
    "Category:Bundesliga",
    "Category:Ligue 1",

    # --- American football ---
    "Category:American football",
    "Category:National Football League",
    "Category:College football",

    # --- Basketball ---
    "Category:Basketball",
    "Category:National Basketball Association",
    "Category:Women's National Basketball Association",
    "Category:NCAA basketball",

    # --- Baseball / softball ---
    "Category:Baseball",
    "Category:Major League Baseball",
    "Category:Softball",

    # --- Ice hockey ---
    "Category:Ice hockey",
    "Category:National Hockey League",

    # --- Cricket / rugby ---
    "Category:Cricket",
    "Category:International Cricket Council",
    "Category:Rugby union",
    "Category:Rugby league",

    # --- Tennis / golf ---
    "Category:Tennis",
    "Category:Grand Slam (tennis)",
    "Category:Golf",
    "Category:PGA Tour",
    "Category:LPGA Tour",

    # --- Motorsport ---
    "Category:Motorsport",
    "Category:Formula One",
    "Category:NASCAR",
    "Category:IndyCar Series",
    "Category:Motorcycle racing",
    "Category:MotoGP",

    # --- Combat sports ---
    "Category:Boxing",
    "Category:Mixed martial arts",
    "Category:Ultimate Fighting Championship",
    "Category:Wrestling",
    "Category:Judo",
    "Category:Taekwondo",

    # --- Other popular sports ---
    "Category:Swimming",
    "Category:Cycling",
    "Category:Track cycling",
    "Category:Road cycling",
    "Category:Athletics (sport)",
    "Category:Marathon running",
    "Category:Gymnastics",
    "Category:Volleyball",
    "Category:Handball",
    "Category:Badminton",
    "Category:Table tennis",
    "Category:Field hockey",
    "Category:Lacrosse",
    "Category:Skateboarding",
    "Category:Surfing",
    "Category:Skiing",
    "Category:Snowboarding",
]

# ROOT_CATEGORIES = [
#     # 核心娱乐
#     "Category:Entertainment",
#     "Category:Popular_culture",

#     # 电影 / 电视
#     "Category:Film",
#     "Category:Cinema_of_the_United_States",
#     "Category:Television",
#     "Category:Television_programs",
#     "Category:Animated_films",
#     "Category:Animation",
#     "Category:Anime",
#     "Category:Manga",

#     # 音乐
#     "Category:Music",
#     "Category:Musical_groups",
#     "Category:Singers",
#     "Category:Albums",
#     "Category:Songs",
#     "Category:Music_awards",
#     "Category:Music_genres",

#     # 名人、演员、娱乐人物
#     "Category:Entertainers",
#     "Category:Actors",
#     "Category:Actresses",
#     "Category:Film_directors",
#     "Category:Celebrities",

#     # 电子游戏
#     "Category:Video_games",
#     "Category:Video_game_industry",
#     "Category:Video_game_development",
#     "Category:Esports",

#     # 娱乐产业 & 奖项
#     "Category:Entertainment_industry",
#     "Category:Film_awards",
#     "Category:Television_awards",
#     "Category:Music_awards",

#     # 喜剧、综艺、社交行为
#     "Category:Comedy",
#     "Category:Stand-up_comedy",
#     "Category:Humor",

#     # 文学（娱乐性质）
#     "Category:Fiction",
#     "Category:Novels",
#     "Category:Fantasy",
#     "Category:Science_fiction",

#     # 文化活动
#     "Category:Festivals",
#     "Category:Entertainment_events",
# ]


# 最大下钻深度（0 表示只看 root；1 表示 root 的子分类；2 表示子子分类）
MAX_DEPTH = 2

# 要收集的最大条目数，够多的话训练小模型已经很爽了
TARGET_ARTICLE_COUNT = 10000

MIN_WORDS = 50  # 太短就丢掉


def clean_text(text: str) -> str:
    """简单清洗 Wikipedia extract：去空行、压缩空白。"""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    text = " ".join(lines)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_category_members(cat_title: str, namespaces="0|14"):
    """
    拿一个分类的成员：
    - ns=0: 词条页面
    - ns=14: 子分类
    文档: https://www.mediawiki.org/wiki/API:Categorymembers
    """
    session = requests.Session()
    members = []
    cmcontinue = None

    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": cat_title,
            "cmlimit": "500",
            "cmnamespace": namespaces,
            "format": "json",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue

        resp = session.get(API_URL, headers=HEADERS, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        part = data.get("query", {}).get("categorymembers", [])
        members.extend(part)

        cont = data.get("continue", {})
        cmcontinue = cont.get("cmcontinue")
        if not cmcontinue:
            break

        time.sleep(0.1)

    return members


def bfs_collect_page_ids(root_categories, max_depth, target_count):
    """
    从若干根分类开始做 BFS：
    - 收集 ns=0 的 pageid 作为条目
    - 发现 ns=14 的子分类时，如果还没超过 max_depth，则加入队列
    """
    visited_cats = set()
    page_ids = set()

    queue = deque()
    for cat in root_categories:
        # 确保传的是 Category:XXX 格式
        if not cat.startswith("Category:"):
            cat = "Category:" + cat
        queue.append((cat, 0))

    while queue and len(page_ids) < target_count:
        cat_title, depth = queue.popleft()
        if cat_title in visited_cats:
            continue
        visited_cats.add(cat_title)

        print(f"[BFS] Category: {cat_title}, depth={depth}")
        try:
            members = fetch_category_members(cat_title, namespaces="0|14")
        except Exception as e:
            print(f"  !! error fetching category {cat_title}: {e}")
            continue

        new_pages = 0
        new_cats = 0

        for m in members:
            ns = m.get("ns")
            title = m.get("title", "")
            if ns == 0:  # 普通条目
                title = m.get("title")

                if title not in page_ids:
                    page_ids.add(title)
                    new_pages += 1
            elif ns == 14 and depth < max_depth:  # 子分类
                # title 本身就像 "Category:Something"
                if title not in visited_cats:
                    queue.append((title, depth + 1))
                    new_cats += 1

        print(f"  + pages: {new_pages}, + subcats queued: {new_cats}, total pages so far: {len(page_ids)}")

    return list(page_ids)

def fetch_page_texts_by_title(page_titles, min_words=30):
    session = requests.Session()
    texts = []

    BATCH = 20
    titles = list(page_titles)

    for i in range(0, len(titles), BATCH):
        batch = titles[i:i + BATCH]
        titles_str = "|".join(batch)

        params = {
            "action": "query",
            "format": "json",

            # ✅ 必须
            "redirects": "1",

            # ✅ 关键：用 titles 而不是 pageids
            "titles": titles_str,

            # ✅ extracts 设置
            "prop": "extracts|pageprops",
            "explaintext": "1",
            "exintro": "1",     # 🔥 关键修复点
            "exlimit": "max",

            # 用来识别 disambiguation
            "ppprop": "disambiguation",
        }

        try:
            resp = session.get(API_URL, headers=HEADERS, params=params, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            print(f"HTTP error: {e}")
            continue

        pages = resp.json().get("query", {}).get("pages", {})

        batch_valid = 0
        for _, p in pages.items():
            title = p.get("title", "")
            text = p.get("extract", "") or ""

            if not text.strip():
                continue

            if "pageprops" in p and "disambiguation" in p["pageprops"]:
                continue

            if title.lower().startswith("list of "):
                continue

            cleaned = clean_text(text)
            if len(cleaned.split()) < min_words:
                continue

            texts.append(cleaned)
            batch_valid += 1

        print(
            f"fetched batch {i//BATCH + 1}, "
            f"pages: {len(batch)}, valid texts: {batch_valid}, "
            f"total texts: {len(texts)}"
        )

        time.sleep(0.1)

    return texts



def main(output_file="sports_wiki.jsonl"):
    print("=== BFS collecting page ids from entertainment categories ===")
    page_ids = bfs_collect_page_ids(ROOT_CATEGORIES, MAX_DEPTH, TARGET_ARTICLE_COUNT)
    print(f"Collected {len(page_ids)} unique page ids.")

    print("=== Fetching page texts ===")
    texts = fetch_page_texts_by_title(page_ids)
    print(f"Got {len(texts)} valid articles with enough words.")

    out_path = Path(output_file)
    with out_path.open("w", encoding="utf-8") as f:
        for t in texts:
            rec = {"text": t}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Done. Saved {len(texts)} articles to {out_path}")


if __name__ == "__main__":
    main()
