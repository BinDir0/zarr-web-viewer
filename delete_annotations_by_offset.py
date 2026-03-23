#!/usr/bin/env python3
import argparse
import sqlite3
from pathlib import Path
from typing import Dict, List

from app import (
    DB_PATH,
    _fetch_annotation_rows,
    build_rework_queue_rows,
    get_factory_episode_lookup,
    load_rework_source_rows,
    scan_factory_episodes,
)


def load_sequential_episode_ids(start: int, end: int) -> List[str]:
    episodes = scan_factory_episodes()
    return [ep["episode_id"] for ep in episodes[start:end]]


def load_rework_episode_ids(start: int, end: int) -> List[str]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        factory_lookup = get_factory_episode_lookup()
        local_rows = _fetch_annotation_rows(
            conn,
            "(content LIKE 'BAD_FRAMES:%' OR content LIKE 'BAD_FRAME:%' OR content LIKE 'REWORK_V1:%')",
        )
        source_rows = load_rework_source_rows()
        rows = build_rework_queue_rows(local_rows, source_rows, factory_lookup)
        return [row["episode_id"] for row in rows[start:end]]
    finally:
        conn.close()


def chunked(seq: List[str], n: int = 900):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def count_rows(conn: sqlite3.Connection, table: str, episode_ids: List[str]) -> int:
    total = 0
    for batch in chunked(episode_ids):
        placeholders = ",".join(["?"] * len(batch))
        total += conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE episode_id IN ({placeholders})",
            batch,
        ).fetchone()[0]
    return total


def delete_rows(conn: sqlite3.Connection, table: str, episode_ids: List[str]) -> int:
    total = 0
    for batch in chunked(episode_ids):
        placeholders = ",".join(["?"] * len(batch))
        cur = conn.execute(
            f"DELETE FROM {table} WHERE episode_id IN ({placeholders})",
            batch,
        )
        total += cur.rowcount if cur.rowcount is not None else 0
    return total


def main():
    parser = argparse.ArgumentParser(
        description="按网站队列 offset 删除本地 annotations/review 状态，而不是按 SQLite 行号删除。"
    )
    parser.add_argument(
        "--queue",
        choices=["sequential", "rework"],
        default="sequential",
        help="按哪个网站队列顺序解释 offset。默认 sequential。",
    )
    parser.add_argument("--start", type=int, required=True, help="起始 offset，包含。")
    parser.add_argument("--end", type=int, required=True, help="结束 offset，不包含。")
    parser.add_argument(
        "--delete-view-log",
        action="store_true",
        help="额外删除 view_log 中对应 episode 的浏览记录。",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="真正执行删除；不带该参数时只做 dry-run。",
    )
    args = parser.parse_args()

    if args.start < 0 or args.end < 0 or args.end <= args.start:
        raise SystemExit("start/end 不合法：需要满足 0 <= start < end")

    db_path = Path(DB_PATH)
    if not db_path.exists():
        raise SystemExit(f"数据库不存在: {db_path}")

    if args.queue == "sequential":
        episode_ids = load_sequential_episode_ids(args.start, args.end)
    else:
        episode_ids = load_rework_episode_ids(args.start, args.end)

    print(f"队列类型: {args.queue}")
    print(f"offset 范围: [{args.start}, {args.end})")
    print(f"命中 episode 数: {len(episode_ids)}")
    if episode_ids:
        print(f"首个 episode_id: {episode_ids[0]}")
        print(f"末个 episode_id: {episode_ids[-1]}")

    if not episode_ids:
        return

    conn = sqlite3.connect(DB_PATH)
    try:
        tables = ["annotations", "reviewed_episodes", "reviewing_episodes"]
        if args.delete_view_log:
            tables.append("view_log")

        print("\n将影响的本地表记录数:")
        for table in tables:
            print(f"  {table}: {count_rows(conn, table, episode_ids)}")

        if not args.execute:
            print("\ndry-run 结束；确认无误后加 --execute 真正删除。")
            return

        deleted: Dict[str, int] = {}
        for table in tables:
            deleted[table] = delete_rows(conn, table, episode_ids)
        conn.commit()

        print("\n已删除:")
        for table in tables:
            print(f"  {table}: {deleted[table]}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
