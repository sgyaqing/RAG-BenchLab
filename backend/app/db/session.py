import logging

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    from app.db import models  # noqa: F401  (registers ORM tables)

    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    _migrate(engine)
    logger.info("Database initialized at %s", settings.database_url)


def _migrate(engine) -> None:
    """Lightweight column additions for existing SQLite databases."""
    with engine.connect() as conn:
        cols = {r[1] for r in conn.exec_driver_sql("PRAGMA table_info(testsets)")}
        if cols and "prompt_language" not in cols:
            conn.exec_driver_sql(
                "ALTER TABLE testsets ADD COLUMN prompt_language VARCHAR(8) DEFAULT 'auto'"
            )
            conn.commit()
            logger.info("Migrated testsets: added prompt_language column")
        if cols and "amplify" not in cols:
            conn.exec_driver_sql(
                "ALTER TABLE testsets ADD COLUMN amplify FLOAT DEFAULT 1.1"
            )
            conn.commit()
            logger.info("Migrated testsets: added amplify column")
        if cols and "gen_amplify" not in cols:
            conn.exec_driver_sql(
                "ALTER TABLE testsets ADD COLUMN gen_amplify FLOAT DEFAULT 1.1"
            )
            conn.commit()
            logger.info("Migrated testsets: added gen_amplify column")
        if cols and "run_seed" not in cols:
            conn.exec_driver_sql(
                "ALTER TABLE testsets ADD COLUMN run_seed INTEGER"
            )
            conn.commit()
            logger.info("Migrated testsets: added run_seed column")

        # One generating testset per corpus, enforced by the database. A
        # customer's database written before this index existed could in
        # principle hold two for one corpus; resume_interrupted would resolve
        # that a moment later anyway, so the same resolution runs here first —
        # the index cannot be created over the rows it would reject.
        conn.exec_driver_sql(
            "UPDATE testsets SET status = 'failed', error = ? "
            "WHERE status = 'generating' AND id NOT IN ("
            "  SELECT MAX(id) FROM testsets WHERE status = 'generating' GROUP BY corpus_id"
            ")",
            ("Another testset on this corpus was still generating when the server "
             "restarted. Resume this one once that run has finished.",),
        )
        conn.exec_driver_sql(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_testsets_generating_per_corpus "
            "ON testsets(corpus_id) WHERE status = 'generating'"
        )
        conn.commit()

        # corpora carried only counters; the conversion now also keeps the
        # timestamped events the other three entities log.
        cols = {r[1] for r in conn.exec_driver_sql("PRAGMA table_info(corpora)")}
        if cols and "log_entries" not in cols:
            conn.exec_driver_sql(
                "ALTER TABLE corpora ADD COLUMN log_entries TEXT DEFAULT '[]'"
            )
            conn.commit()
            logger.info("Migrated corpora: added log_entries column")

        # model_configs was the one store with no uniqueness rule at all, and a
        # name is what a testset records of the model it generated with. Dups
        # in an older database are renamed rather than dropped, oldest first, so
        # the config the earlier records resolve to keeps its name — the index
        # cannot be built over rows that still collide.
        cols = {r[1] for r in conn.exec_driver_sql("PRAGMA table_info(model_configs)")}
        if cols:
            conn.exec_driver_sql(
                "UPDATE model_configs SET name = name || ' (' || id || ')' WHERE id NOT IN ("
                "  SELECT MIN(id) FROM model_configs GROUP BY LOWER(TRIM(name))"
                ")"
            )
            conn.exec_driver_sql(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_model_configs_name "
                "ON model_configs(name)"
            )
            conn.commit()
        # The thinking probe's findings ride on the config: which state the
        # endpoint was found in, and the body fragment that turns it off.
        if cols and "thinking_state" not in cols:
            conn.exec_driver_sql(
                "ALTER TABLE model_configs ADD COLUMN thinking_state VARCHAR(16)"
            )
            conn.exec_driver_sql(
                "ALTER TABLE model_configs ADD COLUMN thinking_param TEXT"
            )
            conn.commit()
            logger.info("Migrated model_configs: added thinking_state/thinking_param")

        # rag_system_configs gained a lifecycle (agent/smart-fill) later
        cols = {r[1] for r in conn.exec_driver_sql("PRAGMA table_info(rag_system_configs)")}
        if cols:
            rag_new_cols = {
                "llm_config_id": "INTEGER",
                "llm_name": "VARCHAR(128)",
                "platform_hint": "VARCHAR(512)",
                "status": "VARCHAR(16) DEFAULT 'completed'",
                "progress": "FLOAT DEFAULT 100",
                "stage": "VARCHAR(32) DEFAULT 'done'",
                "error": "TEXT",
                "log_entries": "TEXT DEFAULT '[]'",
                "lang": "VARCHAR(8) DEFAULT ''",
                "completed_at": "DATETIME",
            }
            for col, ddl in rag_new_cols.items():
                if col not in cols:
                    conn.exec_driver_sql(
                        f"ALTER TABLE rag_system_configs ADD COLUMN {col} {ddl}"
                    )
                    logger.info("Migrated rag_system_configs: added %s column", col)
            # Existing rows predate the lifecycle: they were saved directly, mark done
            conn.exec_driver_sql(
                "UPDATE rag_system_configs SET status='completed', progress=100, "
                "stage='done', completed_at=updated_at WHERE status IS NULL OR status=''"
            )
            # Adapters persist only templates/paths: endpoint+key are transient
            # (used while configuring, cleared at terminal state). Timeout moved
            # to eval_runs; reset it to the fixed probe default on adapters.
            conn.exec_driver_sql(
                "UPDATE rag_system_configs SET base_url='', api_key=NULL, timeout=120 "
                "WHERE status IN ('completed','failed')"
            )
            conn.commit()

        cols = {r[1] for r in conn.exec_driver_sql("PRAGMA table_info(eval_runs)")}
        if cols:
            for col, ddl in {
                "run_base_url": "VARCHAR(512) DEFAULT ''",
                "run_api_key": "VARCHAR(512)",
                "timeout": "INTEGER DEFAULT 120",
                "judge_concurrency": "INTEGER DEFAULT 16",
                "use_judge_cache": "INTEGER DEFAULT 1",
            }.items():
                if col not in cols:
                    conn.exec_driver_sql(f"ALTER TABLE eval_runs ADD COLUMN {col} {ddl}")
                    logger.info("Migrated eval_runs: added %s column", col)
            conn.commit()

        cols = {r[1] for r in conn.exec_driver_sql("PRAGMA table_info(eval_run_items)")}
        if cols and "seconds" not in cols:
            conn.exec_driver_sql("ALTER TABLE eval_run_items ADD COLUMN seconds FLOAT")
            conn.commit()
            logger.info("Migrated eval_run_items: added seconds column")
        if cols and "context_verdicts" not in cols:
            # Runs recorded before this column exists keep the empty default and
            # simply report no Hit@k: the verdicts were never stored, and they
            # cannot be recovered from the score.
            conn.exec_driver_sql(
                "ALTER TABLE eval_run_items ADD COLUMN context_verdicts TEXT DEFAULT '[]'")
            conn.commit()
            logger.info("Migrated eval_run_items: added context_verdicts column")

        # Metric key renamed upstream (ragas): answer_relevancy -> response_relevancy.
        # Rewrite the key inside stored JSON blobs of existing runs.
        if cols and conn.exec_driver_sql(
            "SELECT 1 FROM eval_runs WHERE metrics LIKE '%answer_relevancy%' "
            "OR summary LIKE '%answer_relevancy%' LIMIT 1"
        ).first():
            conn.exec_driver_sql(
                "UPDATE eval_runs SET "
                "metrics = REPLACE(metrics, 'answer_relevancy', 'response_relevancy'), "
                "summary = REPLACE(summary, 'answer_relevancy', 'response_relevancy')"
            )
            conn.exec_driver_sql(
                "UPDATE eval_run_items SET "
                "scores = REPLACE(scores, 'answer_relevancy', 'response_relevancy') "
                "WHERE scores LIKE '%answer_relevancy%'"
            )
            conn.commit()
            logger.info("Migrated metric key: answer_relevancy -> response_relevancy")


def get_session_factory() -> sessionmaker:
    settings = get_settings()
    engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)
