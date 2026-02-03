from django.apps import AppConfig


class PostgresFTSConfig(AppConfig):
    name = "postgres_fts_backend"
    default_auto_field = "django.db.models.AutoField"

    def ready(self):
        from postgres_fts_backend import validate_all_schemas  # noqa: PLC0415

        validate_all_schemas()
