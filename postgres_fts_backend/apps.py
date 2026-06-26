import warnings

from django.apps import AppConfig


class PostgresFTSConfig(AppConfig):
    name = "postgres_fts_backend"
    default_auto_field = "django.db.models.AutoField"

    def ready(self):
        from postgres_fts_backend.models import (  # noqa: PLC0415
            generate_index_models,
            validate_all_schemas,
        )

        # Build the dynamically-generated index models at startup so they live
        # in Django's app registry. Without this they exist in the migration
        # state (created by the generated migrations) but not in the registry,
        # so the autodetector behind a plain `makemigrations` concludes they
        # were deleted and silently emits destructive DeleteModel operations for
        # every index table. build_postgres_schema remains the command that
        # manages their schema; this just keeps `makemigrations` from arming a
        # data-loss migration. Guarded so a startup without discoverable indexes
        # is no worse off than before.
        try:
            generate_index_models()
        except Exception:
            warnings.warn(
                "Could not build search index models at startup; a plain "
                "'makemigrations' may mis-detect them as deletions."
            )

        validate_all_schemas()
