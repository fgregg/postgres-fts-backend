from django.contrib.postgres.indexes import GinIndex, OpClass
from django.contrib.postgres.search import SearchQuery, SearchRank, SearchVectorField
from django.db import models
from haystack import connections as haystack_connections
from haystack import indexes as haystack_indexes


class IndexQuerySet(models.QuerySet):
    def search(self, search_text, config="english"):
        """Full-text filter + rank annotation in one call."""
        sq = SearchQuery(search_text, search_type="websearch", config=config)
        return self.filter(search_vector=sq).annotate(
            rank=SearchRank("search_vector", sq, cover_density=True, normalization=32)
        )

    def ranked(self, search_text, config="english"):
        """Add rank annotation only (when filter is applied separately)."""
        sq = SearchQuery(search_text, search_type="websearch", config=config)
        return self.annotate(
            rank=SearchRank("search_vector", sq, cover_density=True, normalization=32)
        )


FIELD_MAP = {
    haystack_indexes.CharField: lambda: models.TextField(null=True),
    haystack_indexes.EdgeNgramField: lambda: models.TextField(null=True),
    haystack_indexes.NgramField: lambda: models.TextField(null=True),
    haystack_indexes.DateTimeField: lambda: models.DateTimeField(null=True),
    haystack_indexes.DateField: lambda: models.DateField(null=True),
    haystack_indexes.IntegerField: lambda: models.IntegerField(null=True),
    haystack_indexes.FloatField: lambda: models.FloatField(null=True),
    haystack_indexes.BooleanField: lambda: models.BooleanField(null=True),
}


def _django_field_for(haystack_field):
    for haystack_cls, factory in FIELD_MAP.items():
        if isinstance(haystack_field, haystack_cls):
            return factory()
    return models.TextField(null=True)


# Django models for search index tables are created dynamically from Haystack
# SearchIndex definitions. Unlike normal Django models (defined statically in
# models.py), these must be built at runtime because the set of fields depends
# on the user's SearchIndex classes, which can change independently of
# migrations. The database schema is managed separately via the
# build_postgres_schema management command, so the runtime model and the
# database can be out of sync until the user regenerates and applies migrations.
# validate_all_schemas() (called at startup) checks for this.

_index_models_cache = {}


def _build_index_model(source_model, search_index):
    app_label = source_model._meta.app_label
    model_name = source_model._meta.model_name

    class_name = f"HaystackIndex_{app_label.capitalize()}_{model_name.capitalize()}"

    if class_name in _index_models_cache:
        return _index_models_cache[class_name]

    attrs = {
        "__module__": "postgres_fts_backend.models",
        "django_id": models.CharField(max_length=255),
        "django_ct": models.CharField(max_length=255),
        "search_vector": SearchVectorField(null=True),
        "objects": IndexQuerySet.as_manager(),
    }

    for field_name, field_obj in search_index.fields.items():
        if field_name in ("django_ct", "django_id"):
            continue
        attrs[field_name] = _django_field_for(field_obj)

    table_name = f"haystack_index_{app_label}_{model_name}"

    db_indexes = [
        GinIndex(
            fields=["search_vector"],
            name=f"{table_name}_sv_gin",
        )
    ]

    for field_name, field_obj in search_index.fields.items():
        if isinstance(
            field_obj, (haystack_indexes.EdgeNgramField, haystack_indexes.NgramField)
        ):
            db_indexes.append(
                GinIndex(
                    OpClass(models.F(field_name), name="gin_trgm_ops"),
                    name=f"{table_name}_{field_name}_trgm",
                )
            )

    meta = type(
        "Meta",
        (),
        {
            "app_label": "postgres_fts_backend",
            "db_table": table_name,
            "unique_together": [("django_ct", "django_id")],
            "indexes": db_indexes,
        },
    )
    attrs["Meta"] = meta

    model_cls = type(class_name, (models.Model,), attrs)
    _index_models_cache[class_name] = model_cls
    return model_cls


def get_index_model(source_model):
    ui = haystack_connections["default"].get_unified_index()
    search_index = ui.get_index(source_model)
    return _build_index_model(source_model, search_index)


def generate_index_models():
    ui = haystack_connections["default"].get_unified_index()
    return {
        source_model: _build_index_model(source_model, search_index)
        for source_model, search_index in ui.get_indexes().items()
    }
