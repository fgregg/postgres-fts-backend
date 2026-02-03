from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from django.contrib.postgres.indexes import GinIndex, OpClass
from django.contrib.postgres.search import SearchQuery, SearchRank, SearchVectorField
from django.db import models
from django.db.models import Value
from haystack import connections as haystack_connections
from haystack import indexes as haystack_indexes


class IndexQuerySet(models.QuerySet):
    def search(self, search_text: str, config: str = "english") -> IndexQuerySet:
        """Full-text filter + rank annotation in one call."""
        sq = SearchQuery(search_text, search_type="websearch", config=config)
        return self.filter(search_vector=sq).annotate(
            rank=SearchRank("search_vector", sq, cover_density=True, normalization=32)
        )

    def ranked(self, search_text: str, config: str = "english") -> IndexQuerySet:
        """Add rank annotation only (when filter is applied separately)."""
        sq = SearchQuery(search_text, search_type="websearch", config=config)
        return self.annotate(
            rank=SearchRank("search_vector", sq, cover_density=True, normalization=32)
        )

    def aligned_union(self, other: IndexQuerySet) -> AlignedUnionQuerySet:
        """Start a chainable aligned union with another queryset.

        Returns an AlignedUnionQuerySet that can be further chained:
            qs1.aligned_union(qs2).aligned_union(qs3)
        """
        return AlignedUnionQuerySet([self, other])


class AlignedUnionQuerySet:
    """Accumulates querysets and builds an aligned union lazily.

    Introspects model fields and annotations to build a superset of columns.
    Missing columns are filled with Value(None, output_field=...).
    """

    SKIP_FIELDS: set[str] = {"id", "search_vector"}

    def __init__(self, querysets: list[IndexQuerySet]) -> None:
        self._querysets = list(querysets)
        self._built: models.QuerySet | None = None

    def aligned_union(self, other: IndexQuerySet) -> AlignedUnionQuerySet:
        return AlignedUnionQuerySet(self._querysets + [other])

    def _build(self) -> models.QuerySet:
        if self._built is not None:
            return self._built

        all_columns = {}
        per_qs_columns = []

        for qs in self._querysets:
            qs_cols = set()
            for f in qs.model._meta.get_fields():
                if f.name in self.SKIP_FIELDS:
                    continue
                if hasattr(f, "column"):
                    qs_cols.add(f.name)
                    if f.name not in all_columns:
                        all_columns[f.name] = f
            if hasattr(qs, "query") and hasattr(qs.query, "annotations"):
                for name, annotation in qs.query.annotations.items():
                    qs_cols.add(name)
                    if name not in all_columns:
                        all_columns[name] = annotation.output_field
            per_qs_columns.append(qs_cols)

        sorted_cols = sorted(all_columns.keys())

        aligned = []
        for qs, qs_cols in zip(self._querysets, per_qs_columns):
            missing = set(sorted_cols) - qs_cols
            if missing:
                annotations = {}
                for col_name in missing:
                    field_meta = all_columns[col_name]
                    if hasattr(field_meta, "column"):
                        output_field = field_meta.__class__(null=True)
                    else:
                        output_field = field_meta.__class__()
                    annotations[col_name] = Value(None, output_field=output_field)
                qs = qs.annotate(**annotations)
            aligned.append(qs.values(*sorted_cols))

        self._built = aligned[0].union(*aligned[1:], all=True)
        return self._built

    def order_by(self, *args: str) -> models.QuerySet:
        return self._build().order_by(*args)

    def count(self) -> int:
        return self._build().count()

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self._build())

    def __getitem__(self, key: int | slice) -> Any:
        return self._build()[key]


FIELD_MAP: dict[type, Callable[[], models.Field]] = {
    haystack_indexes.CharField: lambda: models.TextField(null=True),
    haystack_indexes.EdgeNgramField: lambda: models.TextField(null=True),
    haystack_indexes.NgramField: lambda: models.TextField(null=True),
    haystack_indexes.DateTimeField: lambda: models.DateTimeField(null=True),
    haystack_indexes.DateField: lambda: models.DateField(null=True),
    haystack_indexes.IntegerField: lambda: models.IntegerField(null=True),
    haystack_indexes.FloatField: lambda: models.FloatField(null=True),
    haystack_indexes.BooleanField: lambda: models.BooleanField(null=True),
}


def _django_field_for(haystack_field: haystack_indexes.SearchField) -> models.Field:
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

_index_models_cache: dict[str, type[models.Model]] = {}


def _build_index_model(
    source_model: type[models.Model], search_index: haystack_indexes.SearchIndex
) -> type[models.Model]:
    app_label: str = source_model._meta.app_label
    model_name: str = source_model._meta.model_name  # type: ignore[assignment]

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


def get_index_model(source_model: type[models.Model]) -> type[models.Model]:
    ui = haystack_connections["default"].get_unified_index()
    search_index = ui.get_index(source_model)
    return _build_index_model(source_model, search_index)


def generate_index_models() -> dict[type[models.Model], type[models.Model]]:
    ui = haystack_connections["default"].get_unified_index()
    return {
        source_model: _build_index_model(source_model, search_index)
        for source_model, search_index in ui.get_indexes().items()
    }
