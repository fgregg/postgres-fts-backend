# Multi-index support

The backend supports registering multiple search indexes — each model gets
its own table (`haystack_index_<app>_<model>`), and `update()`, `remove()`,
and `clear()` all work per-index.  Searching currently requires narrowing to
a single model with `.models(MyModel)`.  The backend raises
`NotImplementedError` if a search spans multiple models.

This document sketches the options for lifting that restriction so that
`SearchQuerySet().auto_query("climate")` returns results from all registered
indexes, merged by relevance — as Haystack's API expects.

## Background

Elasticsearch and Solr handle cross-model search natively — all document
types live in one index with a `django_ct` discriminator field.

Our backend creates a separate PostgreSQL table per model, each with
different columns because each search index defines different fields.
Cross-model search therefore requires combining results from multiple tables.

## Requirements

A multi-index implementation must support:

1. **Correct pagination** — `LIMIT`/`OFFSET` across the merged result set
2. **Relevance ordering** — interleave by `ts_rank` score
3. **Field ordering** — `order_by("-pub_date")` across models
4. **Filtering** — `filter(pub_date__gte=date)` scoped to models that have
   the field
5. **Facets** — merged counts across all matching indexes
6. **Trigram / fuzzy** — `filter(author__fuzzy="danial")` using `pg_trgm`

---

## Option A: Python-side merge

Search each `IndexSearch` independently, merge results in Python.

```
for model, index in indexes.items():
    s = IndexSearch(...)
    results.extend(s.results(...))
results.sort(key=lambda r: -r.score)
results = results[start_offset:end_offset]
```

### Pros

- Simple.  Already fits the `IndexSearch` class design.
- Each per-index query is a normal ORM query with full feature support
  (trigram, filtering, facets).
- No schema changes.

### Cons

- **Pagination is wrong.**  To return results 100–110 you must fetch the top
  110 from *every* index, merge, and slice.  Cost grows linearly with page
  depth × number of indexes.
- Memory pressure for deep pages.

### Verdict

Workable for small result sets and shallow pages.  Not correct in general.

---

## Option B: UNION ALL with per-model tables (current schema)

Build a SQL `UNION ALL` over the identifier + sort columns, with correct
`ORDER BY` / `LIMIT` / `OFFSET` at the database level.

```sql
SELECT django_ct, django_id, rank, pub_date
  FROM haystack_index_core_article
 WHERE search_vector @@ websearch_to_tsquery('english', 'climate')
UNION ALL
SELECT django_ct, django_id, rank, NULL AS pub_date
  FROM haystack_index_core_blogpost
 WHERE search_vector @@ websearch_to_tsquery('english', 'climate')
ORDER BY rank DESC
LIMIT 10 OFFSET 20
```

Stored fields for display come from a second pass (`load_all()` or per-index
`WHERE django_id IN (...)`) after pagination.

### Pros

- Correct pagination at the database level.
- No schema changes.

### Cons

- **Column alignment.**  Each branch of the UNION must SELECT the same
  columns.  Missing fields (e.g. one model lacks `pub_date`) require
  `NULL AS pub_date` padding.  This means inspecting every index's schema at
  query time to determine which columns need padding.
- **Django ORM support is poor.**  `QuerySet.union()` requires compatible
  querysets.  Injecting `Value(None, output_field=...)` annotations for
  missing columns fights the ORM.  Likely requires raw SQL.
- **Two-pass for stored fields.**  The UNION returns only identifiers + sort
  keys; a second query per index hydrates the display fields for the page of
  results.

### Verdict

Correct but awkward.  The raw SQL requirement and column-alignment logic add
significant complexity.

---

## Option C: Single table, real columns (wide sparse table)

Merge all indexes into one table.  Every field from every index becomes a
column, NULLed for models that don't use it.

```
haystack_index
  django_ct   TEXT
  django_id   TEXT
  search_vector  TSVECTOR
  text         TEXT          -- used by all
  author       TEXT          -- only article
  pub_date     TIMESTAMPTZ   -- used by all
  ...
```

### Pros

- No UNION needed — just `WHERE django_ct IN (...)`.
- All ORM features work directly (filtering, ordering, trigram).
- Correct pagination with a single query.

### Cons

- **Column name collisions.**  If two indexes define `status` with different
  types (CharField vs IntegerField), they clash.  Requires namespacing
  (`article__status`, `blogpost__status`) which complicates field resolution.
- **Table width grows** with every registered index.  Acceptable for 2–5
  indexes, ugly at scale.
- PostgreSQL stores NULLs efficiently, so storage isn't a major concern.

### Verdict

Simple and correct when field names don't collide.  Column collisions require
namespacing, which adds mapping complexity.

---

## Option D: Single table, JSONB stored fields

One table with a JSONB column for all non-core fields.

```
haystack_index
  django_ct      TEXT
  django_id      TEXT
  search_vector  TSVECTOR
  stored_fields  JSONB
```

Filtering uses JSON key extraction:

```sql
WHERE stored_fields->>'author' % 'danial1'
```

Trigram indexes work on JSONB expressions:

```sql
CREATE INDEX idx_author_trgm ON haystack_index
  USING GIN ((stored_fields->>'author') gin_trgm_ops);
```

### Pros

- No UNION, no column collisions — field names are JSON keys.
- Single table, single query, correct pagination.
- Schema is fixed regardless of how many indexes are registered.
- PostgreSQL supports GIN trigram indexes on `jsonb ->>` expressions.

### Cons

- **ORM integration is harder.**  `filter(author__trigram_similar="danial1")`
  doesn't map to a JSON key extraction natively.  Requires custom lookups
  built on `KeyTextTransform`.
- **Expression indexes must be created per field.**  Each filterable/sortable
  JSON key needs its own index — these can't be auto-discovered from the
  model definition as easily.
- **Type handling.**  JSON stores everything as text/number/boolean.
  DateTimeField values need casting (`(stored_fields->>'pub_date')::timestamptz`)
  in queries and indexes.

### Verdict

Cleanest schema, but pushes complexity into the ORM layer.  Custom lookups
and expression indexes are well-supported by Django, but it's more work to
build and maintain.

---

## Summary

| Concern             | A (Python merge) | B (UNION ALL) | C (Wide table) | D (JSONB) |
|---------------------|:-:|:-:|:-:|:-:|
| Correct pagination  | No | Yes | Yes | Yes |
| ORM-friendly        | Yes | No (raw SQL) | Yes | Partial |
| Trigram/fuzzy        | Yes | Yes | Yes | Yes (expression index) |
| Column collisions   | N/A | N/A | Problem | N/A |
| Schema complexity   | None (current) | None (current) | Moderate | Low |
| Query-building complexity | Low | High | Low | Moderate |
