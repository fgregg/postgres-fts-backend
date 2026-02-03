# postgres-fts-backend

A [Django Haystack](https://django-haystack.readthedocs.io/) backend that uses
PostgreSQL's built-in full-text search. No external search service required.

## Requirements

- Python >= 3.8
- Django >= 5.0
- django-haystack >= 2.8.0
- PostgreSQL (the `pg_trgm` extension is installed automatically via migration)

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

## Setup

### 1. Define your model

```python
# myapp/models.py
from django.db import models

class Article(models.Model):
    title = models.CharField(max_length=255)
    body = models.TextField()
    author = models.CharField(max_length=100)
    pub_date = models.DateTimeField()
```

### 2. Define a search index

```python
# myapp/search_indexes.py
from haystack import indexes
from myapp.models import Article

class ArticleIndex(indexes.SearchIndex, indexes.Indexable):
    text = indexes.CharField(document=True, use_template=True)
    author = indexes.CharField(model_attr="author")
    pub_date = indexes.DateTimeField(model_attr="pub_date")

    def get_model(self):
        return Article
```

### 3. Create the document template

```
{# myapp/templates/search/indexes/myapp/article_text.txt #}
{{ object.title }}
{{ object.body }}
```

### 4. Generate and apply migrations

```bash
python manage.py build_postgres_schema
python manage.py migrate postgres_fts_backend
```

Run these two commands again whenever you change a `SearchIndex` definition.

### 5. Index your data

```bash
python manage.py update_index
```

## Searching

Standard Haystack `SearchQuerySet` API works as expected:

```python
from haystack.query import SearchQuerySet

# Full-text search (uses PostgreSQL websearch syntax)
results = SearchQuerySet().auto_query("climate change")

# Filter to a specific model
results = SearchQuerySet().models(Article).auto_query("climate change")

# Field filters
results = SearchQuerySet().filter(author__exact="Jane Doe")

# Ordering
results = SearchQuerySet().auto_query("climate").order_by("-pub_date")

# Highlighting
results = SearchQuerySet().auto_query("climate").highlight()

# Boosting
results = SearchQuerySet().auto_query("climate").boost("urgent", 1.5)
```

## Fuzzy search

Fuzzy queries use PostgreSQL's trigram similarity matching (`pg_trgm`):

```python
results = SearchQuerySet().filter(author__fuzzy="Janee")
```

The similarity threshold is controlled by PostgreSQL's
`pg_trgm.similarity_threshold` setting (default 0.3). To adjust it:

```sql
ALTER DATABASE mydb SET pg_trgm.similarity_threshold = 0.5;
```

## Autocomplete with trigram indexes

For autocomplete on a field, declare it as an `EdgeNgramField`. The
backend creates a GIN trigram index and uses trigram similarity for
matching.

```python
class ArticleIndex(indexes.SearchIndex, indexes.Indexable):
    text = indexes.CharField(document=True, use_template=True)
    title = indexes.EdgeNgramField(model_attr="title")

    def get_model(self):
        return Article
```

```python
results = SearchQuerySet().autocomplete(title="clim")
```

After adding or changing `EdgeNgramField`s, regenerate migrations:

```bash
python manage.py build_postgres_schema
python manage.py migrate postgres_fts_backend
```

## Development

Start a local PostgreSQL with Docker:

```bash
docker compose up -d
```

Run the tests:

```bash
PGHOST=localhost PGUSER=postgres PGPASSWORD=postgres python -m pytest
```

## License

MIT
