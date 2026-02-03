import os

# Haystack settings for running tests.
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": os.environ.get("PGDATABASE", "haystack_tests"),
        "USER": os.environ.get("PGUSER", ""),
        "PASSWORD": os.environ.get("PGPASSWORD", ""),
        "HOST": os.environ.get("PGHOST", ""),
        "PORT": os.environ.get("PGPORT", ""),
    }
}

# Use BigAutoField as the default auto field for all models
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

INSTALLED_APPS = [
    "django.contrib.postgres",
    "haystack",
    "tests.core",
    "postgres_fts_backend",
]

MIGRATION_MODULES = {
    "postgres_fts_backend": "tests.search_migrations",
}

USE_TZ = False

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "APP_DIRS": True,
    },
]

HAYSTACK_CONNECTIONS = {
    "default": {"ENGINE": "postgres_fts_backend.PostgresFTSEngine"},
}
