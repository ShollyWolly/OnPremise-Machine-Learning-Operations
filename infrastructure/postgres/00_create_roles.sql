-- =============================================================================
-- Runs BEFORE init.sql (alphabetical order). Creates application user.
-- Executed by postgres superuser on first container start.
-- =============================================================================

DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'mlops_user') THEN
        CREATE ROLE mlops_user WITH LOGIN PASSWORD 'mlops_password_2024';
    END IF;
END
$$;

GRANT ALL PRIVILEGES ON DATABASE mlops TO mlops_user;
