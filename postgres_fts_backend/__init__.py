from __future__ import annotations

import logging
import re
import warnings
from typing import Any, NotRequired, TypedDict

from django.apps import apps as django_apps
from django.contrib.postgres.search import (
    SearchHeadline,
    SearchQuery,
    SearchRank,
    SearchVector,
)
from django.core.exceptions import FieldDoesNotExist
from django.db import DatabaseError, connection, models
from django.db.models import Count, F, FloatField, Q, Value
from django.db.models.functions import Trunc
from django.utils.encoding import force_str
from haystack import connections
from haystack.backends import (
    BaseEngine,
    BaseSearchBackend,
    BaseSearchQuery,
    SearchNode,
    log_query,
)
from haystack.constants import DJANGO_CT, DJANGO_ID
from haystack.indexes import SearchIndex
from haystack.models import SearchResult
from haystack.utils import get_model_ct

from postgres_fts_backend.models import (
    AlignedUnionQuerySet,
    IndexQuerySet,
    generate_index_models,
    get_index_model,
)


class FacetResults(TypedDict, total=False):
    fields: dict[str, list[tuple[Any, int]]]
    dates: dict[str, list[tuple[Any, int]]]
    queries: dict[str, int]


class SearchResponse(TypedDict):
    results: list[SearchResult]
    hits: int
    spelling_suggestion: str | None
    facets: NotRequired[FacetResults]


class _CtInfo(TypedDict):
    model: type[models.Model]
    field_names: list[str]


default_app_config = "postgres_fts_backend.apps.PostgresFTSConfig"

log = logging.getLogger("haystack")


def _table_name(model: type[models.Model]) -> str:
    return f"haystack_index_{model._meta.app_label}_{model._meta.model_name}"


def validate_all_schemas() -> None:
    """Validate all index tables at startup. Called from AppConfig.ready()."""
    try:
        ui = connections["default"].get_unified_index()
        existing_tables = connection.introspection.table_names()
    except Exception:
        warnings.warn(
            "Could not connect to database to validate index schemas. "
            "Run 'manage.py build_postgres_schema' then "
            "'manage.py migrate postgres_fts_backend' once the database is available."
        )
        return

    for model, index in ui.get_indexes().items():
        table = _table_name(model)

        if table not in existing_tables:
            warnings.warn(
                f"Table '{table}' does not exist. Run 'manage.py build_postgres_schema' "
                "then 'manage.py migrate postgres_fts_backend'."
            )
            continue

        expected_columns = {"id", "django_id", "django_ct", "search_vector"}
        for field_name in index.fields:
            if field_name not in ("django_ct", "django_id"):
                expected_columns.add(field_name)

        with connection.cursor() as cursor:
            db_columns = {
                info.name
                for info in connection.introspection.get_table_description(
                    cursor, table
                )
            }

        missing = expected_columns - db_columns
        if missing:
            warnings.warn(
                "Index table '{}' schema is out of date (missing columns: {}). "
                "Run 'manage.py build_postgres_schema' then "
                "'manage.py migrate postgres_fts_backend'.".format(
                    table, ", ".join(sorted(missing))
                )
            )


def _field_names(index: SearchIndex) -> list[str]:
    return [name for name in index.fields if name not in ("django_ct", "django_id")]


def _resolve_field_name(field_name: str) -> str:
    if field_name.endswith("_exact"):
        return field_name[:-6]
    return field_name


def _parse_narrow_query(query_string: str) -> tuple[str, str]:
    match = re.match(r'^(\w+):"(.+)"$', query_string)
    if not match:
        raise ValueError(f"Cannot parse narrow query: '{query_string}'")
    return match.group(1), match.group(2)


class IndexSearch:
    def __init__(
        self,
        qs: IndexQuerySet,
        index: SearchIndex,
        search_config: str,
        has_rank: bool = False,
        search_text: str | None = None,
    ) -> None:
        self.qs = qs
        self.index = index
        self.search_config: str = search_config
        self.has_rank: bool = has_rank
        self.search_text: str | None = search_text
        self.score_field: str = "rank"
        self.highlight_field: str | None = None

    @classmethod
    def from_query_string(
        cls,
        index_model: type[models.Model],
        index: SearchIndex,
        search_config: str,
        query_string: str,
    ) -> IndexSearch:
        if query_string == "*":
            qs = index_model.objects.all().annotate(  # type: ignore[attr-defined]
                rank=Value(0, output_field=FloatField())
            )
            return cls(qs, index, search_config)

        if ":" in query_string and not query_string.startswith('"'):
            field, _, value = query_string.partition(":")
            content_field = index.get_content_field()
            if field == content_field:
                return cls(
                    index_model.objects.search(value, config=search_config),  # type: ignore[attr-defined]
                    index,
                    search_config,
                    has_rank=True,
                    search_text=value,
                )
            qs = index_model.objects.filter(**{field: value}).annotate(  # type: ignore[attr-defined]
                rank=Value(0, output_field=FloatField())
            )
            return cls(qs, index, search_config)

        return cls(
            index_model.objects.search(query_string, config=search_config),  # type: ignore[attr-defined]
            index,
            search_config,
            has_rank=True,
            search_text=query_string,
        )

    @classmethod
    def from_orm_query(
        cls,
        index_model: type[models.Model],
        index: SearchIndex,
        search_config: str,
        orm_query: Q,
    ) -> IndexSearch:
        content_search_text = orm_query.content_search_text  # type: ignore[attr-defined]
        qs = index_model.objects.filter(orm_query)  # type: ignore[attr-defined]
        if content_search_text:
            qs = qs.ranked(content_search_text, config=search_config)
        else:
            qs = qs.annotate(rank=Value(0, output_field=FloatField()))
        return cls(
            qs,
            index,
            search_config,
            has_rank=bool(content_search_text),
            search_text=content_search_text,
        )

    def narrow(self, narrow_queries: list[str]) -> None:
        for nq in narrow_queries:
            field, value = _parse_narrow_query(nq)
            col = _resolve_field_name(field)
            try:
                self.qs.model._meta.get_field(col)
            except FieldDoesNotExist:
                self.qs = self.qs.none()
                return
            self.qs = self.qs.filter(**{col: value})

    def highlight(self) -> None:
        if self.search_text is None:
            return
        content_field = self.index.get_content_field()
        sq = SearchQuery(
            self.search_text, search_type="websearch", config=self.search_config
        )
        self.qs = self.qs.annotate(
            headline=SearchHeadline(content_field, sq, config=self.search_config)
        )
        self.highlight_field = content_field

    def boost(self, boost_dict: dict[str, float]) -> None:
        if not self.has_rank or not boost_dict:
            return
        annotations: dict[str, Any] = {}
        combined: Any = F("rank")
        for i, (term, weight) in enumerate(boost_dict.items()):
            alias = f"_boost_{i}"
            bq = SearchQuery(term, search_type="websearch", config=self.search_config)
            annotations[alias] = SearchRank(
                "search_vector", bq, cover_density=True, normalization=32
            )
            combined = combined * (1.0 + F(alias) * weight)
        annotations["_boosted_rank"] = combined
        self.qs = self.qs.annotate(**annotations)
        self.score_field = "_boosted_rank"

    def count(self) -> int:
        return self.qs.count()

    def facets(
        self,
        facets: list[str] | None = None,
        date_facets: dict[str, Any] | None = None,
        query_facets: list[tuple[str, str]] | None = None,
    ) -> FacetResults:
        result: FacetResults = {}

        if facets:
            result["fields"] = {}
            for field_name in facets:
                col = _resolve_field_name(field_name)
                facet_qs = (
                    self.qs.values(col)
                    .annotate(count=Count("id"))
                    .order_by("-count", col)
                )
                result["fields"][field_name] = [
                    (row[col], row["count"]) for row in facet_qs
                ]

        if date_facets:
            result["dates"] = {}
            for field_name, facet_opts in date_facets.items():
                col = _resolve_field_name(field_name)
                gap_by = facet_opts["gap_by"]
                start_date = facet_opts["start_date"]
                end_date = facet_opts["end_date"]
                facet_qs = (
                    self.qs.filter(
                        **{
                            f"{col}__gte": start_date,
                            f"{col}__lt": end_date,
                        }
                    )
                    .annotate(bucket=Trunc(col, gap_by))
                    .values("bucket")
                    .annotate(count=Count("id"))
                    .order_by("bucket")
                )
                result["dates"][field_name] = [
                    (row["bucket"], row["count"]) for row in facet_qs
                ]

        if query_facets:
            result["queries"] = {}
            for field_name, value in query_facets:
                col = _resolve_field_name(field_name)
                count = self.qs.filter(**{col: value}).count()
                result["queries"][f"{field_name}_{value}"] = count

        return result

    def results(
        self,
        sort_by: list[str] | None = None,
        start_offset: int = 0,
        end_offset: int | None = None,
        result_class: type = SearchResult,
    ) -> list[SearchResult]:
        qs = self.qs

        # Ordering
        if sort_by:
            qs = qs.order_by(*sort_by)
        elif self.has_rank:
            qs = qs.order_by(f"-{self.score_field}")

        # Pagination
        if end_offset is not None:
            qs = qs[start_offset:end_offset]
        elif start_offset:
            qs = qs[start_offset:]

        # Materialize
        model = self.index.get_model()
        field_names = _field_names(self.index)
        app_label = model._meta.app_label
        model_name = model._meta.model_name

        results = []
        for obj in qs:
            stored_fields = {fn: getattr(obj, fn) for fn in field_names}
            if self.highlight_field:
                headline = getattr(obj, "headline", None)
                if headline:
                    stored_fields["highlighted"] = {self.highlight_field: [headline]}
            rank = getattr(obj, self.score_field, None)
            score = float(rank) if self.has_rank and rank is not None else 0
            results.append(
                result_class(
                    app_label, model_name, obj.django_id, score, **stored_fields
                )
            )
        return results


class MultiIndexSearch:
    """Wraps multiple IndexSearch instances for cross-model search."""

    def __init__(self, searches: list[tuple[IndexSearch, type[models.Model]]]) -> None:
        self.searches = searches

    def count(self) -> int:
        return sum(s.count() for s, _model in self.searches)

    def facets(
        self,
        facets: list[str] | None = None,
        date_facets: dict[str, Any] | None = None,
        query_facets: list[tuple[str, str]] | None = None,
    ) -> FacetResults:
        merged: FacetResults = {}

        if facets:
            merged["fields"] = {}
            for field_name in facets:
                col = _resolve_field_name(field_name)
                combined_counts: dict[Any, int] = {}
                for s, model in self.searches:
                    index_model = get_index_model(model)
                    try:
                        index_model._meta.get_field(col)
                    except FieldDoesNotExist:
                        continue
                    sub = s.facets(facets=[field_name])
                    for value, count in sub.get("fields", {}).get(field_name, []):
                        combined_counts[value] = combined_counts.get(value, 0) + count
                merged["fields"][field_name] = sorted(
                    combined_counts.items(), key=lambda x: (-x[1], x[0])
                )

        if date_facets:
            merged["dates"] = {}
            for field_name, facet_opts in date_facets.items():
                col = _resolve_field_name(field_name)
                combined_buckets: dict[Any, int] = {}
                for s, model in self.searches:
                    index_model = get_index_model(model)
                    try:
                        index_model._meta.get_field(col)
                    except FieldDoesNotExist:
                        continue
                    sub = s.facets(date_facets={field_name: facet_opts})
                    for bucket, count in sub.get("dates", {}).get(field_name, []):
                        combined_buckets[bucket] = (
                            combined_buckets.get(bucket, 0) + count
                        )
                merged["dates"][field_name] = sorted(combined_buckets.items())

        if query_facets:
            merged["queries"] = {}
            for field_name, value in query_facets:
                col = _resolve_field_name(field_name)
                total = 0
                for s, model in self.searches:
                    index_model = get_index_model(model)
                    try:
                        index_model._meta.get_field(col)
                    except FieldDoesNotExist:
                        continue
                    sub = s.facets(query_facets=[(field_name, value)])
                    key = f"{field_name}_{value}"
                    total += sub.get("queries", {}).get(key, 0)
                merged["queries"][f"{field_name}_{value}"] = total

        return merged

    def results(
        self,
        sort_by: list[str] | None = None,
        start_offset: int = 0,
        end_offset: int | None = None,
        result_class: type = SearchResult,
    ) -> list[SearchResult]:
        # These are uniform across all searches (same kwargs applied to each)
        first_search = self.searches[0][0]
        score_field = first_search.score_field
        has_rank = first_search.has_rank
        highlight_field = first_search.highlight_field

        # Build the aligned union
        union_qs: IndexQuerySet | AlignedUnionQuerySet = first_search.qs
        for s, model in self.searches[1:]:
            union_qs = union_qs.aligned_union(s.qs)

        # Per-model lookup: field_names and model identity vary across indexes
        ct_map: dict[str, _CtInfo] = {
            get_model_ct(model): {
                "model": model,
                "field_names": _field_names(s.index),
            }
            for s, model in self.searches
        }

        # Ordering — always include tiebreakers for stable pagination
        if sort_by:
            ordered_qs = union_qs.order_by(*sort_by, "django_ct", "django_id")
        else:
            ordered_qs = union_qs.order_by("-rank", "django_ct", "django_id")

        # Pagination
        if end_offset is not None:
            ordered_qs = ordered_qs[start_offset:end_offset]
        elif start_offset:
            ordered_qs = ordered_qs[start_offset:]

        # Materialize
        results = []
        for row in ordered_qs:
            info = ct_map[row["django_ct"]]
            model = info["model"]

            stored_fields = {fn: row.get(fn) for fn in info["field_names"]}

            if highlight_field and "headline" in row and row["headline"]:
                stored_fields["highlighted"] = {highlight_field: [row["headline"]]}

            rank = row.get(score_field)
            score = float(rank) if has_rank and rank is not None else 0

            results.append(
                result_class(
                    model._meta.app_label,
                    model._meta.model_name,
                    row["django_id"],
                    score,
                    **stored_fields,
                )
            )
        return results


class PostgresFTSSearchBackend(BaseSearchBackend):
    def __init__(self, connection_alias, **connection_options):
        super().__init__(connection_alias, **connection_options)
        self.search_config = connection_options.get("SEARCH_CONFIG", "english")

    def build_schema(self, fields):
        return generate_index_models()

    def update(self, index, iterable, commit=True):
        try:
            model = index.get_model()
            index_model = get_index_model(model)
            field_names = _field_names(index)
            content_field = index.get_content_field()

            rows = []
            for obj in iterable:
                prepared = index.full_prepare(obj)
                defaults = {fn: prepared.get(fn) for fn in field_names}
                rows.append(
                    index_model(
                        django_ct=prepared[DJANGO_CT],
                        django_id=prepared[DJANGO_ID],
                        **defaults,
                    )
                )

            if rows:
                index_model.objects.bulk_create(
                    rows,
                    update_conflicts=True,
                    unique_fields=["django_ct", "django_id"],
                    update_fields=field_names,
                )
                index_model.objects.filter(
                    django_ct=get_model_ct(model),
                    django_id__in=[r.django_id for r in rows],
                ).update(
                    search_vector=SearchVector(content_field, config=self.search_config)
                )
        except DatabaseError:
            if not self.silently_fail:
                raise
            log.exception("Failed to update index for %s", index)

    def remove(self, obj_or_string, commit=True):
        try:
            if isinstance(obj_or_string, str):
                # String format: "app_label.model_name.pk"
                parts = obj_or_string.split(".", 2)
                if len(parts) != 3:
                    raise ValueError(
                        "String identifier must be 'app_label.model_name.pk', "
                        f"got '{obj_or_string}'"
                    )
                model = django_apps.get_model(parts[0], parts[1])
                django_ct = get_model_ct(model)
                django_id = parts[2]
            else:
                model = type(obj_or_string)
                django_ct = get_model_ct(model)
                django_id = force_str(obj_or_string.pk)

            index_model = get_index_model(model)
            index_model.objects.filter(
                django_ct=django_ct, django_id=django_id
            ).delete()
        except DatabaseError:
            if not self.silently_fail:
                raise
            log.exception("Failed to remove document '%s'", obj_or_string)

    def clear(self, models=None, commit=True):
        try:
            if models is None:
                ui = connections["default"].get_unified_index()
                models = ui.get_indexes().keys()

            for model in models:
                index_model = get_index_model(model)
                index_model.objects.all().delete()
        except DatabaseError:
            if not self.silently_fail:
                raise
            log.exception("Failed to clear index")

    @log_query
    def search(self, query_string: str, **kwargs: Any) -> SearchResponse:
        if not query_string or not query_string.strip():
            return {"results": [], "hits": 0, "spelling_suggestion": None}

        try:
            ui = connections["default"].get_unified_index()

            requested_models = kwargs.get("models")
            if requested_models:
                model_index_pairs = [(m, ui.get_index(m)) for m in requested_models]
            else:
                model_index_pairs = list(ui.get_indexes().items())

            orm_query = kwargs.pop("orm_query", None)

            result_class = kwargs.get("result_class", SearchResult)
            sort_by = kwargs.get("sort_by")
            start_offset = int(kwargs.get("start_offset", 0))
            end_offset = (
                int(kwargs["end_offset"])
                if kwargs.get("end_offset") is not None
                else None
            )

            # Build IndexSearch per model
            searches = []
            for model, index in model_index_pairs:
                index_model = get_index_model(model)
                if orm_query is not None:
                    s = IndexSearch.from_orm_query(
                        index_model, index, self.search_config, orm_query
                    )
                else:
                    s = IndexSearch.from_query_string(
                        index_model, index, self.search_config, query_string
                    )

                if narrow_queries := kwargs.get("narrow_queries"):
                    s.narrow(narrow_queries)
                if boost := kwargs.get("boost"):
                    s.boost(boost)
                if kwargs.get("highlight"):
                    s.highlight()

                searches.append((s, model))

            if len(searches) == 1:
                # Single-model path (no behavior change)
                s, model = searches[0]
                total_count = s.count()
                facets = s.facets(
                    facets=kwargs.get("facets"),
                    date_facets=kwargs.get("date_facets"),
                    query_facets=kwargs.get("query_facets"),
                )
                results = s.results(sort_by, start_offset, end_offset, result_class)
            else:
                # Multi-model path
                multi = MultiIndexSearch(searches)
                total_count = multi.count()
                facets = multi.facets(
                    facets=kwargs.get("facets"),
                    date_facets=kwargs.get("date_facets"),
                    query_facets=kwargs.get("query_facets"),
                )
                results = multi.results(sort_by, start_offset, end_offset, result_class)

            response: SearchResponse = {
                "results": results,
                "hits": total_count,
                "spelling_suggestion": None,
            }
            if facets:
                response["facets"] = facets
            return response
        except DatabaseError:
            if not self.silently_fail:
                raise
            log.exception("Failed to search with query '%s'", query_string)
            return {"results": [], "hits": 0, "spelling_suggestion": None}

    def prep_value(self, value: Any) -> Any:
        return value

    def more_like_this(
        self,
        model_instance: models.Model,
        additional_query_string: str | None = None,
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError(
            "postgres_fts_backend does not support more_like_this. "
            "PostgreSQL has no native document similarity feature."
        )


class ORMSearchNode(SearchNode):
    def as_orm_query(self, query_fragment_callback):
        result = []
        for child in self.children:
            if hasattr(child, "as_orm_query"):
                result.append(child.as_orm_query(query_fragment_callback))
            else:
                expression, value = child
                field, filter_type = self.split_expression(expression)
                result.append(query_fragment_callback(field, filter_type, value))

        query = Q()
        if self.connector == self.AND:
            for subquery in result:
                query &= subquery
        elif self.connector == self.OR:
            for subquery in result:
                query |= subquery

        if query and self.negated:
            query = ~query

        return query


class ORMSearchQuery(BaseSearchQuery):

    def __init__(self, using="default"):
        super().__init__(using=using)
        self.query_filter = ORMSearchNode()
        self.content_search_text = None

    def clean(self, query_fragment):
        return query_fragment

    def build_query_fragment(self, field, filter_type, value):
        if hasattr(value, "prepare"):
            value = value.prepare(self)

        if filter_type == "content":
            if field == "content":
                self.content_search_text = value
                return Q(
                    search_vector=SearchQuery(
                        value,
                        search_type="websearch",
                        config=self.backend.search_config,
                    )
                )
            return Q(**{f"{field}__trigram_similar": value})

        if filter_type == "fuzzy":
            return Q(**{f"{field}__trigram_similar": value})

        if filter_type == "in":
            value = list(value)
            if not value:
                return Q(pk__in=[])
        elif filter_type == "range":
            value = (value[0], value[1])

        # contains/startswith/endswith → case-insensitive Django lookups
        lookup = {
            "contains": "icontains",
            "startswith": "istartswith",
            "endswith": "iendswith",
        }.get(filter_type, filter_type)

        return Q(**{f"{field}__{lookup}": value})

    def build_query(self):
        final_query = self.query_filter.as_orm_query(self.build_query_fragment)
        if not final_query:
            return Q()
        return final_query

    def matching_all_fragment(self):
        return Q()

    def run(self, spelling_query=None, **kwargs):
        final_query = self.build_query()
        search_kwargs = self.build_params(spelling_query=spelling_query)

        if kwargs:
            search_kwargs.update(kwargs)

        final_query.content_search_text = self.content_search_text
        search_kwargs["orm_query"] = final_query

        results = self.backend.search("*", **search_kwargs)
        self._results = results.get("results", [])
        self._hit_count = results.get("hits", 0)
        self._facet_counts = self.post_process_facets(results)
        self._spelling_suggestion = results.get("spelling_suggestion", None)

    def get_count(self):
        if self._hit_count is None:
            self.run()
        return self._hit_count


class PostgresFTSEngine(BaseEngine):
    backend = PostgresFTSSearchBackend
    query = ORMSearchQuery
