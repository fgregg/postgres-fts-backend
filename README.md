# postgres-fts-backend

A [Django Haystack](https://django-haystack.readthedocs.io/) backend that uses
PostgreSQL's built-in full-text search. No external search service required.

## Requirements

- Python >= 3.8
- Django >= 5.0
- django-haystack >= 2.8.0
- PostgreSQL

## Installation

```bash
pip install postgres-fts-backend
```

Add to `INSTALLED_APPS`:

```python
INSTALLED_APPS = [
    "django.contrib.postgres",
    "haystack",
    "postgres_fts_backend",
    # ...
]
```

Set a migration module so the generated search index migrations live in
your project rather than inside the installed package:

```python
MIGRATION_MODULES = {
    "postgres_fts_backend": "myapp.search_migrations",
}
```

Configure Haystack:

```python
HAYSTACK_CONNECTIONS = {
    "default": {
        "ENGINE": "postgres_fts_backend.PostgresFTSEngine",
    },
}
```

To use a search configuration other than `"english"`:

```python
HAYSTACK_CONNECTIONS = {
    "default": {
        "ENGINE": "postgres_fts_backend.PostgresFTSEngine",
        "SEARCH_CONFIG": "spanish",
    },
}
```

## Other Peculiarities of this backend

### Build indexes through models and migrations

```bash
python manage.py build_postgres_schema
python manage.py migrate postgres_fts_backend
```

Run these two commands again whenever you change a `SearchIndex` definition.

### Fuzzy search

Fuzzy queries use PostgreSQL's trigram similarity matching (`pg_trgm`):

```python
results = SearchQuerySet().filter(author__fuzzy="Janee")
```

The similarity threshold is controlled by PostgreSQL's
`pg_trgm.similarity_threshold` setting (default 0.3). To adjust it:

```sql
ALTER DATABASE mydb SET pg_trgm.similarity_threshold = 0.5;
```

### `more_like_this` not implemented
PostgreSQL FTS doesn't provide any facilities for this. It could be done, but I just need to think more about it.

### `spelling_suggestions` are not supported
