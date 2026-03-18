FROM postgres:16

# 01: Restore schema + filtered data dump (usable records only)
COPY dump.sql.gz /docker-entrypoint-initdb.d/01_dump.sql.gz

# 02: Clear output tables and nullify llm_reparsed_at (--force semantics)
COPY reset.sql /docker-entrypoint-initdb.d/02_reset.sql

ENV POSTGRES_DB=inference_fixture
ENV POSTGRES_USER=fixture
ENV POSTGRES_PASSWORD=fixture
ENV POSTGRES_HOST_AUTH_METHOD=trust

EXPOSE 5432
