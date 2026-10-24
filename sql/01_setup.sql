-- Schemas (the database "zomato" is created by docker-compose via POSTGRES_DB)
CREATE EXTENSION IF NOT EXISTS vector;

CREATE SCHEMA IF NOT EXISTS raw;        -- Bronze: CSV loaded as-is from MinIO
CREATE SCHEMA IF NOT EXISTS staging;    -- Silver: cleaned / conformed (dbt)
CREATE SCHEMA IF NOT EXISTS marts;      -- Gold: facts, dims, analytical marts (dbt)
CREATE SCHEMA IF NOT EXISTS snapshots;  -- SCD2 history (dbt)
CREATE SCHEMA IF NOT EXISTS ai;         -- LLM-enriched tables + embeddings

-- Roles (passwords are set by scripts/setup_db.py from .env)
DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'loader') THEN
    CREATE ROLE loader LOGIN;
  END IF;
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'dbt_user') THEN
    CREATE ROLE dbt_user LOGIN;
  END IF;
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'app_readonly') THEN
    CREATE ROLE app_readonly LOGIN;
  END IF;
END $$;

-- Database access (dbt needs CREATE so it can make schemas if missing)
GRANT CONNECT ON DATABASE zomato TO loader, app_readonly;
GRANT CONNECT, CREATE ON DATABASE zomato TO dbt_user;

-- Schema ownership: loader owns raw, dbt owns everything it builds
ALTER SCHEMA raw       OWNER TO loader;
ALTER SCHEMA staging   OWNER TO dbt_user;
ALTER SCHEMA marts     OWNER TO dbt_user;
ALTER SCHEMA snapshots OWNER TO dbt_user;
ALTER SCHEMA ai        OWNER TO dbt_user;

-- dbt can read raw, including tables the loader creates later
GRANT USAGE ON SCHEMA raw TO dbt_user;
GRANT SELECT ON ALL TABLES IN SCHEMA raw TO dbt_user;
ALTER DEFAULT PRIVILEGES FOR ROLE loader IN SCHEMA raw
  GRANT SELECT ON TABLES TO dbt_user;

-- Read-only app role (Streamlit + text-to-SQL): marts and ai only, never raw
GRANT USAGE ON SCHEMA marts, ai TO app_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA marts, ai TO app_readonly;
ALTER DEFAULT PRIVILEGES FOR ROLE dbt_user IN SCHEMA marts, ai
  GRANT SELECT ON TABLES TO app_readonly;
ALTER ROLE app_readonly SET statement_timeout = '30s';
ALTER ROLE app_readonly SET default_transaction_read_only = on;