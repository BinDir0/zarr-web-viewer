#!/usr/bin/env python3
import argparse
import shutil
import sqlite3
from pathlib import Path


ROOT = Path(__file__).parent
DEFAULT_TARGET = ROOT / "annotations.db"


def table_exists(conn: sqlite3.Connection, schema: str, table: str) -> bool:
    row = conn.execute(
        f"SELECT 1 FROM {schema}.sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def count_rows(conn: sqlite3.Connection, schema: str, table: str) -> int:
    return int(conn.execute(f"SELECT COUNT(*) FROM {schema}.{table}").fetchone()[0])


def ensure_target_table(conn: sqlite3.Connection, table: str) -> None:
    if table == "annotation_reasons":
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS annotation_reasons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id TEXT NOT NULL UNIQUE,
                content TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_annotation_reasons_episode ON annotation_reasons(episode_id)"
        )


def merge_annotations(conn: sqlite3.Connection) -> int:
    if not table_exists(conn, "src", "annotations"):
        return 0
    before = count_rows(conn, "main", "annotations")
    conn.execute(
        """
        INSERT INTO main.annotations (
            episode_id, episode_name, dataset_name, episode_index, content, updated_at
        )
        SELECT
            episode_id, episode_name, dataset_name, episode_index, content, updated_at
        FROM src.annotations
        WHERE 1
        ON CONFLICT(episode_id) DO UPDATE SET
            episode_name = excluded.episode_name,
            dataset_name = excluded.dataset_name,
            episode_index = excluded.episode_index,
            content = excluded.content,
            updated_at = excluded.updated_at
        WHERE excluded.updated_at >= main.annotations.updated_at
        """
    )
    after = count_rows(conn, "main", "annotations")
    return after - before


def merge_annotation_reasons(conn: sqlite3.Connection) -> int:
    if not table_exists(conn, "src", "annotation_reasons"):
        return 0
    ensure_target_table(conn, "annotation_reasons")
    before = count_rows(conn, "main", "annotation_reasons")
    conn.execute(
        """
        INSERT INTO main.annotation_reasons (episode_id, content, updated_at)
        SELECT episode_id, content, updated_at
        FROM src.annotation_reasons
        WHERE 1
        ON CONFLICT(episode_id) DO UPDATE SET
            content = excluded.content,
            updated_at = excluded.updated_at
        WHERE excluded.updated_at >= main.annotation_reasons.updated_at
        """
    )
    after = count_rows(conn, "main", "annotation_reasons")
    return after - before


def merge_reviewed_episodes(conn: sqlite3.Connection) -> int:
    if not table_exists(conn, "src", "reviewed_episodes"):
        return 0
    before = count_rows(conn, "main", "reviewed_episodes")
    conn.execute(
        """
        INSERT INTO main.reviewed_episodes (episode_id, user_id, has_annotation, reviewed_at)
        SELECT episode_id, user_id, has_annotation, reviewed_at
        FROM src.reviewed_episodes
        WHERE 1
        ON CONFLICT(episode_id) DO UPDATE SET
            user_id = excluded.user_id,
            has_annotation = excluded.has_annotation,
            reviewed_at = excluded.reviewed_at
        WHERE excluded.reviewed_at >= main.reviewed_episodes.reviewed_at
        """
    )
    after = count_rows(conn, "main", "reviewed_episodes")
    return after - before


def merge_view_log(conn: sqlite3.Connection) -> int:
    if not table_exists(conn, "src", "view_log"):
        return 0
    before = count_rows(conn, "main", "view_log")
    conn.execute(
        """
        INSERT OR IGNORE INTO main.view_log (episode_id, user_id, viewed_at)
        SELECT episode_id, user_id, viewed_at
        FROM src.view_log
        """
    )
    after = count_rows(conn, "main", "view_log")
    return after - before


def main() -> None:
    parser = argparse.ArgumentParser(
        description="合并另一个工作目录里的 annotations.db 到当前数据库。"
    )
    parser.add_argument("source_db", help="源 annotations.db 路径")
    parser.add_argument(
        "--target-db",
        default=str(DEFAULT_TARGET),
        help=f"目标 annotations.db 路径，默认 {DEFAULT_TARGET}",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="跳过目标库备份（默认会先备份）。",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="真正执行合并；不带该参数时只打印信息。",
    )
    args = parser.parse_args()

    source_db = Path(args.source_db).expanduser().resolve()
    target_db = Path(args.target_db).expanduser().resolve()

    if not source_db.exists():
        raise SystemExit(f"源数据库不存在: {source_db}")
    if not target_db.exists():
        raise SystemExit(f"目标数据库不存在: {target_db}")
    if source_db == target_db:
        raise SystemExit("源数据库和目标数据库不能相同")

    conn = sqlite3.connect(target_db)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("ATTACH DATABASE ? AS src", (str(source_db),))

        summary = {
            "src.annotations": count_rows(conn, "src", "annotations") if table_exists(conn, "src", "annotations") else 0,
            "src.reviewed_episodes": count_rows(conn, "src", "reviewed_episodes") if table_exists(conn, "src", "reviewed_episodes") else 0,
            "src.view_log": count_rows(conn, "src", "view_log") if table_exists(conn, "src", "view_log") else 0,
            "src.annotation_reasons": count_rows(conn, "src", "annotation_reasons") if table_exists(conn, "src", "annotation_reasons") else 0,
        }
        print(f"source_db: {source_db}")
        print(f"target_db: {target_db}")
        for key, value in summary.items():
            print(f"{key}: {value}")

        if not args.execute:
            print("\ndry-run 结束；确认无误后加 --execute 真正合并。")
            return

        if not args.no_backup:
            backup_path = target_db.with_name(target_db.name + ".backup_before_merge")
            shutil.copy2(target_db, backup_path)
            print(f"已备份目标库到: {backup_path}")

        conn.execute("PRAGMA busy_timeout = 15000")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("BEGIN IMMEDIATE")

        inserted_annotations = merge_annotations(conn)
        inserted_reviewed = merge_reviewed_episodes(conn)
        inserted_view_log = merge_view_log(conn)
        inserted_reasons = merge_annotation_reasons(conn)

        conn.commit()

        print("\n合并完成:")
        print(f"annotations 新增净行数: {inserted_annotations}")
        print(f"reviewed_episodes 新增净行数: {inserted_reviewed}")
        print(f"view_log 新增净行数: {inserted_view_log}")
        print(f"annotation_reasons 新增净行数: {inserted_reasons}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
