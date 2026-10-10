import psycopg2
from qualume import config


def get_connection():
    """Return a new psycopg2 connection using config loaded from .env.

    Caller owns the connection lifecycle (close it when done). Kept as a thin
    wrapper, not a pooled/managed connection, since Phase 0/1 usage is
    low-frequency scripts, not a long-running service.
    """
    return psycopg2.connect(
        host=config.POSTGRES_HOST,
        port=config.POSTGRES_PORT,
        dbname=config.POSTGRES_DB,
        user=config.POSTGRES_USER,
        password=config.POSTGRES_PASSWORD,
    )
