import os
import pickle
import shutil
import sys
import tempfile
import unittest
from datetime import datetime
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest
from django.contrib.postgres.indexes import GinIndex
from django.contrib.postgres.search import SearchQuery, SearchVectorField
from django.core.exceptions import FieldError
from django.core.management import call_command
from django.db import DatabaseError, connection, models
from django.db.models import FloatField, Value
from django.test import TestCase, override_settings
from haystack import connections
from haystack import indexes as haystack_indexes
from haystack.inputs import AutoQuery
from haystack.query import SQ, RelatedSearchQuerySet, SearchQuerySet
from haystack.utils.loading import UnifiedIndex

import postgres_fts_backend.models as models_module
from postgres_fts_backend import validate_all_schemas
from postgres_fts_backend.models import generate_index_models, get_index_model
from tests.core.models import AnotherMockModel, MockModel, ScoreMockModel
from tests.mocks import MockSearchResult
from tests.search_indexes import (
    AnotherMockSearchIndex,
    MockSearchIndex,
    ScoreMockSearchIndex,
)


def backend_setup(test_case, indexes_to_register):
    """Save the existing unified index and install a fresh one with the given indexes."""
    test_case.old_ui = connections["default"].get_unified_index()
    ui = UnifiedIndex()
    ui.build(indexes=indexes_to_register)
    connections["default"]._index = ui
    test_case.backend = connections["default"].get_backend()


def backend_teardown(test_case):
    """Restore the original unified index."""
    connections["default"]._index = test_case.old_ui


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------


class TestIndexManagement(TestCase):
    fixtures = ["bulk_data.json"]

    def setUp(self):
        self.index = MockSearchIndex()
        backend_setup(self, [self.index])

    def tearDown(self):
        backend_teardown(self)

    def test_update(self):
        """Indexing objects should populate the index table."""
        self.backend.update(self.index, MockModel.objects.all())
        results = self.backend.search("indexing")
        assert results["hits"] > 0

    def test_update_existing_object(self):
        """Re-indexing an already-indexed object should update, not duplicate."""
        self.backend.update(self.index, MockModel.objects.all())
        count_before = self.backend.search("*")["hits"]

        # Update the same objects again
        self.backend.update(self.index, MockModel.objects.all())
        count_after = self.backend.search("*")["hits"]

        assert count_before == count_after

    def test_update_partial(self):
        """Re-indexing a subset should update only those documents."""
        self.backend.update(self.index, MockModel.objects.all())
        obj = MockModel.objects.first()
        original_author = obj.author

        obj.author = "modified_author"
        obj.save()
        self.backend.update(self.index, MockModel.objects.filter(pk=obj.pk))

        results = self.backend.search("author:modified_author")
        assert results["hits"] == 1

        # Other documents unchanged
        total = self.backend.search("*")["hits"]
        assert total == MockModel.objects.count()

        # Restore
        obj.author = original_author
        obj.save()

    def test_remove(self):
        """Removing an object should delete it from the index."""
        self.backend.update(self.index, MockModel.objects.all())
        count_before = self.backend.search("*")["hits"]

        obj = MockModel.objects.first()
        self.backend.remove(obj)

        count_after = self.backend.search("*")["hits"]
        assert count_after == count_before - 1

    def test_remove_by_string(self):
        """Removing by string identifier should delete the document."""
        self.backend.update(self.index, MockModel.objects.all())
        count_before = self.backend.search("*")["hits"]

        obj = MockModel.objects.first()
        self.backend.remove(f"core.mockmodel.{obj.pk}")

        count_after = self.backend.search("*")["hits"]
        assert count_after == count_before - 1

    def test_clear_all(self):
        """Clearing without specifying models should empty the entire index."""
        self.backend.update(self.index, MockModel.objects.all())
        assert self.backend.search("*")["hits"] > 0

        self.backend.clear()
        assert self.backend.search("*")["hits"] == 0

    def test_clear_by_model(self):
        """Clearing with a model list should only remove that model's documents."""
        another_index = AnotherMockSearchIndex()
        backend_setup(self, [self.index, another_index])
        self.backend.update(self.index, MockModel.objects.all())
        self.backend.update(another_index, AnotherMockModel.objects.all())

        self.backend.clear(models=[MockModel])

        # MockModel documents gone, AnotherMockModel documents remain
        results = self.backend.search("*", models=[AnotherMockModel])
        assert results["hits"] > 0
        for result in results["results"]:
            assert result.model != MockModel

    def test_update_empty_iterable(self):
        """Updating with no objects should not error."""
        self.backend.update(self.index, MockModel.objects.none())


# ---------------------------------------------------------------------------
# Basic full-text search (direct backend calls)
# ---------------------------------------------------------------------------


class TestBasicSearch(TestCase):
    fixtures = ["bulk_data.json"]

    def setUp(self):
        self.index = MockSearchIndex()
        backend_setup(self, [self.index])
        self.backend.update(self.index, MockModel.objects.all())

    def tearDown(self):
        backend_teardown(self)

    def test_search_single_word(self):
        """Searching for a single word should return matching documents."""
        results = self.backend.search("indexing")
        assert results["hits"] > 0

    def test_search_no_results(self):
        """Searching for a term not in any document should return zero hits."""
        results = self.backend.search("xyznonexistent")
        assert results["hits"] == 0
        assert len(results["results"]) == 0

    def test_search_phrase(self):
        """A quoted phrase should match only documents containing that exact phrase."""
        results = self.backend.search('"search backend"')
        assert results["hits"] > 0

    def test_search_and(self):
        """Multiple terms should be ANDed by default."""
        results_both = self.backend.search("search backend")
        results_single = self.backend.search("search")
        assert results_both["hits"] <= results_single["hits"]

    def test_search_or(self):
        """OR queries should return the union of results."""
        results = self.backend.search("indexing OR registering")
        results_a = self.backend.search("indexing")
        results_b = self.backend.search("registering")
        assert results["hits"] >= max(results_a["hits"], results_b["hits"])

    def test_search_negation(self):
        """Negation should exclude documents containing the negated term."""
        results_all = self.backend.search("search")
        results_neg = self.backend.search("search -backend")
        assert results_neg["hits"] < results_all["hits"]

    def test_search_empty_string(self):
        """An empty search string should return no results."""
        results = self.backend.search("")
        assert results["hits"] == 0

    def test_search_unicode(self):
        """Searching with unicode characters should not error."""
        results = self.backend.search("café résumé")
        assert isinstance(results["hits"], int)

    def test_search_special_characters(self):
        """Searching with special characters should not error or inject SQL."""
        results = self.backend.search('test\'s & "value"')
        assert isinstance(results["hits"], int)


# ---------------------------------------------------------------------------
# Field-scoped search (direct backend calls)
# ---------------------------------------------------------------------------


class TestFieldSearch(TestCase):
    fixtures = ["bulk_data.json"]

    def setUp(self):
        self.index = MockSearchIndex()
        backend_setup(self, [self.index])
        self.backend.update(self.index, MockModel.objects.all())

    def tearDown(self):
        backend_teardown(self)

    def test_search_specific_field(self):
        """Searching a specific indexed field should only match that field's content."""
        results = self.backend.search("author:daniel1")
        assert results["hits"] > 0
        for result in results["results"]:
            obj = MockModel.objects.get(pk=result.pk)
            assert obj.author == "daniel1"

    def test_search_document_field(self):
        """Searching the document field should do full-text search, not exact match."""
        results = self.backend.search("text:indexing")
        assert results["hits"] > 0
        for result in results["results"]:
            assert result.score > 0


# ---------------------------------------------------------------------------
# SQS filtering
# ---------------------------------------------------------------------------


class TestFiltering(TestCase):
    fixtures = ["bulk_data.json"]

    def setUp(self):
        self.index = MockSearchIndex()
        backend_setup(self, [self.index])
        self.backend.update(self.index, MockModel.objects.all())

    def tearDown(self):
        backend_teardown(self)

    def test_filter_exact(self):
        """Filtering with exact match should return only matching documents."""
        sqs = SearchQuerySet().filter(author__exact="daniel1")
        assert len(sqs) > 0
        for result in sqs:
            obj = MockModel.objects.get(pk=result.pk)
            assert obj.author == "daniel1"

    def test_filter_in(self):
        """Filtering with __in should return documents matching any of the values."""
        sqs = SearchQuerySet().filter(author__in=["daniel1", "daniel2"])
        assert len(sqs) > 0
        for result in sqs:
            obj = MockModel.objects.get(pk=result.pk)
            assert obj.author in ["daniel1", "daniel2"]

    def test_filter_in_empty_list(self):
        """Filtering with __in and an empty list should return no results."""
        sqs = SearchQuerySet().filter(author__in=[])
        assert len(sqs) == 0

    def test_filter_date_gt(self):
        """Filtering pub_date with __gt should exclude earlier documents."""
        cutoff = datetime(2009, 7, 17, 10, 0, 0)
        sqs = SearchQuerySet().filter(pub_date__gt=cutoff)
        assert len(sqs) > 0
        for result in sqs:
            obj = MockModel.objects.get(pk=result.pk)
            assert obj.pub_date > cutoff

    def test_filter_date_lt(self):
        """Filtering pub_date with __lt should exclude later documents."""
        cutoff = datetime(2009, 7, 17, 5, 0, 0)
        sqs = SearchQuerySet().filter(pub_date__lt=cutoff)
        assert len(sqs) > 0
        for result in sqs:
            obj = MockModel.objects.get(pk=result.pk)
            assert obj.pub_date < cutoff

    def test_filter_date_gte(self):
        """Filtering pub_date with __gte should include the boundary value."""
        cutoff = datetime(2009, 7, 17, 10, 30, 0)
        sqs = SearchQuerySet().filter(pub_date__gte=cutoff)
        assert len(sqs) > 0
        for result in sqs:
            obj = MockModel.objects.get(pk=result.pk)
            assert obj.pub_date >= cutoff

    def test_filter_date_lte(self):
        """Filtering pub_date with __lte should include the boundary value."""
        cutoff = datetime(2009, 7, 17, 5, 30, 0)
        sqs = SearchQuerySet().filter(pub_date__lte=cutoff)
        assert len(sqs) > 0
        for result in sqs:
            obj = MockModel.objects.get(pk=result.pk)
            assert obj.pub_date <= cutoff

    def test_filter_date_range(self):
        """Filtering with pub_date__range should return documents within the range."""
        start = datetime(2009, 7, 17, 5, 0, 0)
        end = datetime(2009, 7, 17, 10, 0, 0)
        sqs = SearchQuerySet().filter(pub_date__range=[start, end])
        assert len(sqs) > 0
        for result in sqs:
            obj = MockModel.objects.get(pk=result.pk)
            assert obj.pub_date >= start
            assert obj.pub_date <= end

    def test_filter_contains(self):
        """Filtering with __contains should match partial strings."""
        sqs = SearchQuerySet().filter(author__contains="iel1")
        assert len(sqs) > 0
        for result in sqs:
            obj = MockModel.objects.get(pk=result.pk)
            assert "iel1" in obj.author

    def test_filter_startswith(self):
        """Filtering with __startswith should match the beginning of strings."""
        sqs = SearchQuerySet().filter(author__startswith="daniel")
        assert len(sqs) == MockModel.objects.count()

    def test_filter_endswith(self):
        """Filtering with __endswith should match the end of strings."""
        sqs = SearchQuerySet().filter(author__endswith="1")
        assert len(sqs) > 0
        for result in sqs:
            obj = MockModel.objects.get(pk=result.pk)
            assert obj.author.endswith("1")

    def test_filter_combined_with_search(self):
        """Filtering and full-text search should work together."""
        sqs = SearchQuerySet().filter(author__exact="daniel1").auto_query("indexing")
        assert len(sqs) > 0
        for result in sqs:
            obj = MockModel.objects.get(pk=result.pk)
            assert obj.author == "daniel1"

    def test_filter_multiple_and(self):
        """Chaining multiple filters should AND them together."""
        cutoff = datetime(2009, 7, 17, 5, 0, 0)
        sqs = (
            SearchQuerySet().filter(author__exact="daniel1").filter(pub_date__gt=cutoff)
        )
        assert len(sqs) > 0
        for result in sqs:
            obj = MockModel.objects.get(pk=result.pk)
            assert obj.author == "daniel1"
            assert obj.pub_date > cutoff

    def test_exclude(self):
        """Excluding should remove matching documents from results."""
        sqs_all = SearchQuerySet().all()
        sqs_excluded = SearchQuerySet().exclude(author="daniel1")
        assert len(sqs_excluded) < len(sqs_all)
        for result in sqs_excluded:
            obj = MockModel.objects.get(pk=result.pk)
            assert obj.author != "daniel1"

    def test_exclude_with_filter(self):
        """Exclude combined with a filter should apply both conditions."""
        sqs = (
            SearchQuerySet()
            .filter(author__in=["daniel1", "daniel2"])
            .exclude(author__exact="daniel1")
        )
        assert len(sqs) > 0
        for result in sqs:
            obj = MockModel.objects.get(pk=result.pk)
            assert obj.author == "daniel2"


# ---------------------------------------------------------------------------
# Field validation
# ---------------------------------------------------------------------------
# Field validation (errors come from the ORM, not the backend)
# ---------------------------------------------------------------------------


class TestFieldValidation(TestCase):
    fixtures = ["bulk_data.json"]

    def setUp(self):
        self.index = MockSearchIndex()
        backend_setup(self, [self.index])
        self.backend.update(self.index, MockModel.objects.all())

    def tearDown(self):
        backend_teardown(self)

    def test_search_unknown_field_falls_through_to_fts(self):
        """Searching field:value with a non-existent field falls through to full-text search."""
        result = self.backend.search("nonexistent_field:foo")
        assert isinstance(result["hits"], int)

    def test_filter_unknown_field_returns_empty(self):
        """Filtering on a non-existent field returns empty results."""
        sqs = SearchQuerySet().filter(nonexistent_field__exact="foo")
        assert len(sqs) == 0

    def test_order_by_rejects_unknown_field(self):
        """Ordering by a non-existent field should raise FieldError."""
        with pytest.raises(FieldError):
            list(SearchQuerySet().all().order_by("nonexistent_field"))


# ---------------------------------------------------------------------------
# Scoring and ordering
# ---------------------------------------------------------------------------


class TestScoringAndOrdering(TestCase):
    fixtures = ["bulk_data.json"]

    def setUp(self):
        self.index = MockSearchIndex()
        backend_setup(self, [self.index])
        self.backend.update(self.index, MockModel.objects.all())

    def tearDown(self):
        backend_teardown(self)

    def test_scores_are_nonzero(self):
        """Search results should have meaningful relevance scores via ts_rank."""
        results = self.backend.search("indexing")
        assert results["hits"] > 0
        for result in results["results"]:
            assert result.score > 0

    def test_order_by_relevance(self):
        """Results should be ordered by descending relevance score by default."""
        results = self.backend.search("indexing")
        scores = [r.score for r in results["results"]]
        assert scores == sorted(scores, reverse=True)

    def test_order_by_field_ascending(self):
        """Results should be orderable by a field in ascending order."""
        sqs = SearchQuerySet().all().order_by("pub_date")
        dates = [r.pub_date for r in sqs]
        assert dates == sorted(dates)

    def test_order_by_field_descending(self):
        """Results should be orderable by a field in descending order."""
        sqs = SearchQuerySet().all().order_by("-pub_date")
        dates = [r.pub_date for r in sqs]
        assert dates == sorted(dates, reverse=True)

    def test_order_by_multiple_fields(self):
        """Ordering by multiple fields should sort by the first, then the second."""
        sqs = SearchQuerySet().all().order_by("author", "pub_date")
        results = list(sqs)
        pairs = [(r.author, r.pub_date) for r in results]
        assert pairs == sorted(pairs)


# ---------------------------------------------------------------------------
# Highlighting
# ---------------------------------------------------------------------------


class TestHighlighting(TestCase):
    fixtures = ["bulk_data.json"]

    def setUp(self):
        self.index = MockSearchIndex()
        backend_setup(self, [self.index])
        self.backend.update(self.index, MockModel.objects.all())

    def tearDown(self):
        backend_teardown(self)

    def test_highlight(self):
        """Search with highlight=True should return highlighted snippets."""
        results = self.backend.search("indexing", highlight=True)
        assert results["hits"] > 0
        result = results["results"][0]
        assert "highlighted" in dir(result)


# ---------------------------------------------------------------------------
# More Like This
# ---------------------------------------------------------------------------


class TestMoreLikeThis(TestCase):
    fixtures = ["bulk_data.json"]

    def setUp(self):
        self.index = MockSearchIndex()
        backend_setup(self, [self.index])
        self.backend.update(self.index, MockModel.objects.all())

    def tearDown(self):
        backend_teardown(self)

    @unittest.skip("more_like_this not implemented")
    def test_more_like_this(self):
        """more_like_this should return similar documents."""
        obj = MockModel.objects.get(pk=1)
        results = self.backend.more_like_this(obj)
        assert results["hits"] > 0
        pks = {r.pk for r in results["results"]}
        assert str(obj.pk) not in pks

    @unittest.skip("more_like_this not implemented")
    def test_more_like_this_with_model_filter(self):
        """more_like_this with models kwarg should restrict to that model."""
        another_index = AnotherMockSearchIndex()
        backend_setup(self, [self.index, another_index])
        self.backend.update(self.index, MockModel.objects.all())
        self.backend.update(another_index, AnotherMockModel.objects.all())

        obj = MockModel.objects.get(pk=1)
        results = self.backend.more_like_this(obj, models=[MockModel])
        for result in results["results"]:
            assert result.model == MockModel


# ---------------------------------------------------------------------------
# Faceting
# ---------------------------------------------------------------------------


class TestFaceting(TestCase):
    fixtures = ["bulk_data.json"]

    def setUp(self):
        self.index = MockSearchIndex()
        backend_setup(self, [self.index])
        self.backend.update(self.index, MockModel.objects.all())

    def tearDown(self):
        backend_teardown(self)

    def test_field_facet(self):
        """Faceting on a field should return value counts."""
        sqs = SearchQuerySet().all().facet("author")
        counts = sqs.facet_counts()
        assert "fields" in counts
        assert "author" in counts["fields"]
        assert len(counts["fields"]["author"]) > 0

    def test_date_facet(self):
        """Date faceting should return counts grouped by time interval."""
        sqs = (
            SearchQuerySet()
            .all()
            .date_facet(
                "pub_date",
                start_date=datetime(2009, 1, 1),
                end_date=datetime(2010, 1, 1),
                gap_by="month",
            )
        )
        counts = sqs.facet_counts()
        assert "dates" in counts
        assert "pub_date" in counts["dates"]

    def test_query_facet(self):
        """Query faceting should return a count for a query expression."""
        sqs = SearchQuerySet().all().query_facet("author", "daniel1")
        counts = sqs.facet_counts()
        assert "queries" in counts
        assert len(counts["queries"]) > 0

    def test_narrow(self):
        """Narrowing should restrict results based on a facet query."""
        sqs = SearchQuerySet().all().narrow('author:"daniel1"')
        for result in sqs:
            obj = MockModel.objects.get(pk=result.pk)
            assert obj.author == "daniel1"


# ---------------------------------------------------------------------------
# Autocomplete
# ---------------------------------------------------------------------------


class TestAutocomplete(TestCase):
    fixtures = ["bulk_data.json"]

    def setUp(self):
        self.index = MockSearchIndex()
        backend_setup(self, [self.index])
        self.backend.update(self.index, MockModel.objects.all())

    def tearDown(self):
        backend_teardown(self)

    def test_autocomplete(self):
        """Autocomplete should return results matching a prefix."""
        sqs = SearchQuerySet().autocomplete(author="dan")
        assert len(sqs) > 0

    def test_autocomplete_multiple_words(self):
        """Autocomplete with multiple words should match the prefix of each."""
        sqs = SearchQuerySet().autocomplete(author="daniel")
        assert len(sqs) > 0


# ---------------------------------------------------------------------------
# Fuzzy search
# ---------------------------------------------------------------------------


class TestFuzzySearch(TestCase):
    fixtures = ["bulk_data.json"]

    def setUp(self):
        self.index = MockSearchIndex()
        backend_setup(self, [self.index])
        self.backend.update(self.index, MockModel.objects.all())

    def tearDown(self):
        backend_teardown(self)

    def test_fuzzy_matches_close_spelling(self):
        """Fuzzy filter should match values that are close but not exact."""
        # "danial1" is a misspelling — not a substring of any author value,
        # so icontains would return nothing, but trigram similarity finds matches
        sqs_fuzzy = SearchQuerySet().filter(author__fuzzy="danial1")
        sqs_icontains = SearchQuerySet().filter(author__contains="danial1")
        assert len(sqs_fuzzy) > 0
        assert len(sqs_icontains) == 0

    def test_fuzzy_no_match_for_unrelated(self):
        """Fuzzy filter should not match completely unrelated values."""
        sqs = SearchQuerySet().filter(author__fuzzy="zzzzzzz")
        assert len(sqs) == 0


# ---------------------------------------------------------------------------
# Boost
# ---------------------------------------------------------------------------


class TestBoost(TestCase):
    fixtures = ["bulk_data.json"]

    def setUp(self):
        self.index = MockSearchIndex()
        backend_setup(self, [self.index])
        self.backend.update(self.index, MockModel.objects.all())

    def tearDown(self):
        backend_teardown(self)

    def test_boost_term(self):
        """Boosting a term should increase the score of matching documents."""
        sqs_normal = SearchQuerySet().auto_query("indexing")
        sqs_boosted = SearchQuerySet().auto_query("indexing").boost("registering", 2.0)
        # Both should return results; boosted may reorder them
        assert len(sqs_normal) > 0
        assert len(sqs_boosted) > 0


# ---------------------------------------------------------------------------
# SearchQuerySet
# ---------------------------------------------------------------------------


class TestSearchQuerySet(TestCase):
    fixtures = ["bulk_data.json"]

    def setUp(self):
        self.index = MockSearchIndex()
        backend_setup(self, [self.index])
        self.backend.update(self.index, MockModel.objects.all())

    def tearDown(self):
        backend_teardown(self)

    def test_all(self):
        """SearchQuerySet.all() should return all indexed documents."""
        sqs = SearchQuerySet().all()
        assert len(sqs) == MockModel.objects.count()

    def test_auto_query(self):
        """auto_query should parse a user search string and return results."""
        sqs = SearchQuerySet().auto_query("indexing")
        assert len(sqs) > 0

    def test_auto_query_unicode(self):
        """auto_query with unicode input should not error."""
        sqs = SearchQuerySet().auto_query("café résumé")
        assert isinstance(len(sqs), int)

    def test_auto_query_special_characters(self):
        """auto_query with quotes and special chars should not error."""
        sqs = SearchQuerySet().auto_query('test\'s & "value"')
        assert isinstance(len(sqs), int)

    def test_count(self):
        """count() should return the total number of matching results."""
        sqs = SearchQuerySet().all()
        assert sqs.count() == MockModel.objects.count()

    def test_slicing(self):
        """Slicing a SearchQuerySet should return the correct subset."""
        sqs = SearchQuerySet().all()
        sliced = sqs[0:5]
        assert len(sliced) == 5

    def test_slicing_with_offset(self):
        """Slicing with a non-zero start should skip earlier results."""
        sqs = SearchQuerySet().all().order_by("pub_date")
        full = list(sqs)
        sliced = list(sqs[5:10])
        assert len(sliced) == 5
        assert [r.pk for r in sliced] == [r.pk for r in full[5:10]]

    def test_slicing_beyond_results(self):
        """Slicing beyond the result count should return only available results."""
        sqs = SearchQuerySet().all()
        total = len(sqs)
        sliced = sqs[total - 2 : total + 10]
        assert len(sliced) == 2

    def test_single_item_access(self):
        """Accessing a single item by index should return one SearchResult."""
        sqs = SearchQuerySet().all().order_by("pub_date")
        result = sqs[0]
        assert hasattr(result, "pk")

    def test_iteration(self):
        """Iterating over a SearchQuerySet should yield SearchResult objects."""
        sqs = SearchQuerySet().auto_query("indexing")
        results = list(sqs)
        assert len(results) > 0
        for result in results:
            assert hasattr(result, "pk")
            assert hasattr(result, "score")

    def test_custom_result_class(self):
        """Passing a custom result_class should return instances of that class."""
        sqs = SearchQuerySet().result_class(MockSearchResult).auto_query("indexing")
        results = list(sqs)
        assert len(results) > 0
        for result in results:
            assert isinstance(result, MockSearchResult)

    def test_models_filter(self):
        """SearchQuerySet.models() should restrict results to specific models."""
        another_index = AnotherMockSearchIndex()
        backend_setup(self, [self.index, another_index])
        self.backend.update(self.index, MockModel.objects.all())
        self.backend.update(another_index, AnotherMockModel.objects.all())

        sqs = SearchQuerySet().models(MockModel)
        for result in sqs:
            assert result.model == MockModel

    def test_values(self):
        """values() should return dictionaries with the requested fields."""
        sqs = SearchQuerySet().all().values("author", "pub_date")
        results = list(sqs)
        assert len(results) > 0
        for result in results:
            assert "author" in result
            assert "pub_date" in result

    def test_values_list(self):
        """values_list() should return tuples of the requested fields."""
        sqs = SearchQuerySet().all().values_list("author", flat=True)
        results = list(sqs)
        assert len(results) > 0
        for result in results:
            assert isinstance(result, str)

    def test_load_all(self):
        """load_all() should prefetch Django objects for all results."""
        sqs = SearchQuerySet().all().load_all()
        results = list(sqs)
        assert len(results) > 0
        for result in results:
            obj = result.object
            assert obj is not None
            assert isinstance(obj, MockModel)

    def test_queryset_and_combine(self):
        """Combining two SQS with & should AND their filters."""
        sqs1 = SearchQuerySet().filter(author__exact="daniel1")
        sqs2 = SearchQuerySet().filter(pub_date__gt=datetime(2009, 7, 17, 5, 0, 0))
        combined = sqs1 & sqs2
        assert len(combined) > 0
        for result in combined:
            obj = MockModel.objects.get(pk=result.pk)
            assert obj.author == "daniel1"
            assert obj.pub_date > datetime(2009, 7, 17, 5, 0, 0)

    def test_queryset_or_combine(self):
        """Combining two SQS with | should OR their filters."""
        sqs1 = SearchQuerySet().filter(author__exact="daniel1")
        sqs2 = SearchQuerySet().filter(author__exact="daniel2")
        combined = sqs1 | sqs2
        count_1 = len(sqs1)
        count_2 = len(sqs2)
        assert len(combined) >= max(count_1, count_2)
        for result in combined:
            obj = MockModel.objects.get(pk=result.pk)
            assert obj.author in ["daniel1", "daniel2"]


# ---------------------------------------------------------------------------
# AutoQuery behavior
# ---------------------------------------------------------------------------


class TestAutoQueryPrepare(TestCase):
    """Tests that AutoQuery.prepare() produces correct websearch_to_tsquery input.

    The ORMSearchQuery overrides clean() to strip reserved words (OR) and
    build_not_query() to emit '-' negation instead of 'NOT'.
    """

    fixtures = ["bulk_data.json"]

    def setUp(self):
        self.index = MockSearchIndex()
        backend_setup(self, [self.index])
        self.backend.update(self.index, MockModel.objects.all())
        self.query_obj = connections["default"].get_query()

    def tearDown(self):
        backend_teardown(self)

    def _prepare(self, query_string):
        return AutoQuery(query_string).prepare(self.query_obj)

    # --- clean: OR stripping ---

    def test_or_uppercase_stripped(self):
        """OR (uppercase) should be removed from the query."""
        prepared = self._prepare("cats OR dogs")
        assert "OR" not in prepared.split()
        assert "cats" in prepared.split()
        assert "dogs" in prepared.split()

    def test_or_lowercase_stripped(self):
        """or (lowercase) should be removed from the query."""
        prepared = self._prepare("cats or dogs")
        assert "or" not in prepared.split()
        assert "cats" in prepared.split()
        assert "dogs" in prepared.split()

    def test_or_mixed_case_stripped(self):
        """Or (mixed case) should be removed from the query."""
        prepared = self._prepare("cats Or dogs")
        assert "Or" not in prepared.split()
        assert "cats" in prepared.split()
        assert "dogs" in prepared.split()

    def test_or_not_stripped_within_word(self):
        """Words containing 'or' should not be affected."""
        assert self._prepare("organic oracle") == "organic oracle"

    # --- build_not_query: dash negation ---

    def test_negation_uses_dash(self):
        """Negated terms should use '-' prefix, not 'NOT'."""
        prepared = self._prepare("-dogs")
        assert prepared == "-dogs"

    def test_negation_with_positive_term(self):
        """Negated term mixed with positive term."""
        prepared = self._prepare("cats -dogs")
        assert prepared == "cats -dogs"

    def test_multiple_negations(self):
        """Multiple negated terms should each get '-' prefix."""
        prepared = self._prepare("-cats -dogs")
        assert prepared == "-cats -dogs"

    # --- exact phrases ---

    def test_quoted_phrase_preserved(self):
        """Quoted phrases should pass through with quotes."""
        prepared = self._prepare('"search backend"')
        assert prepared == '"search backend"'

    def test_quoted_phrase_with_other_terms(self):
        """Quoted phrase alongside plain terms."""
        prepared = self._prepare('cats "search backend" dogs')
        assert prepared == 'cats "search backend" dogs'

    # --- combined ---

    def test_negation_and_phrase(self):
        """Negation and quoted phrase together."""
        prepared = self._prepare('cats -dogs "exact phrase"')
        assert prepared == 'cats -dogs "exact phrase"'

    def test_or_stripped_with_negation_and_phrase(self):
        """OR stripped while negation and quotes are preserved."""
        prepared = self._prepare('cats OR -dogs "exact phrase"')
        assert "OR" not in prepared.split()
        assert "-dogs" in prepared
        assert '"exact phrase"' in prepared

    # --- integration: SQS auto_query produces correct results ---

    def test_auto_query_negation_excludes_results(self):
        """auto_query negation should exclude documents with the negated term."""
        sqs_positive = SearchQuerySet().auto_query("search")
        sqs_negated = SearchQuerySet().auto_query("search -backend")
        assert len(sqs_negated) < len(sqs_positive)

    def test_auto_query_phrase_search(self):
        """auto_query phrase should match only documents with that exact phrase."""
        sqs_phrase = SearchQuerySet().auto_query('"search backend"')
        sqs_words = SearchQuerySet().auto_query("search backend")
        # Phrase is at least as restrictive as individual words
        assert len(sqs_phrase) <= len(sqs_words)
        assert len(sqs_phrase) > 0

    def test_auto_query_or_not_treated_as_disjunction(self):
        """auto_query should strip OR, so 'a OR b' behaves like 'a b' (AND)."""
        sqs_or = SearchQuerySet().auto_query("indexing OR template")
        sqs_and = SearchQuerySet().auto_query("indexing template")
        assert len(sqs_or) == len(sqs_and)


# ---------------------------------------------------------------------------
# Multi-model search
# ---------------------------------------------------------------------------


class TestMultiModelSearch(TestCase):
    fixtures = ["bulk_data.json"]

    def setUp(self):
        self.mock_index = MockSearchIndex()
        self.another_index = AnotherMockSearchIndex()
        backend_setup(self, [self.mock_index, self.another_index])
        self.backend.update(self.mock_index, MockModel.objects.all())
        self.backend.update(self.another_index, AnotherMockModel.objects.all())

    def tearDown(self):
        backend_teardown(self)

    def test_index_multiple_models(self):
        """Indexing multiple models should store documents for all of them."""
        mock_results = self.backend.search("*", models=[MockModel])
        another_results = self.backend.search("*", models=[AnotherMockModel])
        total = mock_results["hits"] + another_results["hits"]
        expected = MockModel.objects.count() + AnotherMockModel.objects.count()
        assert total == expected

    def test_search_across_all_models(self):
        """Searching '*' across all models returns combined hit count."""
        results = self.backend.search("*")
        expected = MockModel.objects.count() + AnotherMockModel.objects.count()
        assert results["hits"] == expected

    def test_search_returns_both_models(self):
        """Searching 'daniel3' returns results from both model types."""
        results = self.backend.search("daniel3")
        model_types = {r.model for r in results["results"]}
        assert MockModel in model_types
        assert AnotherMockModel in model_types

    def test_relevance_scores(self):
        """Multi-model results have non-zero scores."""
        results = self.backend.search("daniel3")
        assert results["hits"] > 0
        for r in results["results"]:
            assert r.score > 0

    def test_ordered_by_relevance(self):
        """Multi-model results are ordered by descending score."""
        results = self.backend.search("daniel3")
        scores = [r.score for r in results["results"]]
        assert scores == sorted(scores, reverse=True)

    def test_pagination(self):
        """start_offset/end_offset work; hits reflects the total."""
        total = self.backend.search("*")["hits"]
        page = self.backend.search("*", start_offset=0, end_offset=5)
        assert len(page["results"]) == 5
        assert page["hits"] == total

    def test_no_duplicate_across_pages(self):
        """Paging through all results yields no duplicates."""
        page_size = 5
        all_ids = []
        total = self.backend.search("*")["hits"]
        offset = 0
        while offset < total:
            page = self.backend.search(
                "*", start_offset=offset, end_offset=offset + page_size
            )
            for r in page["results"]:
                all_ids.append((r.app_label, r.model_name, r.pk))
            offset += page_size
        assert len(all_ids) == total
        assert len(all_ids) == len(set(all_ids))

    def test_explicit_multi_model_list(self):
        """models=[MockModel, AnotherMockModel] works."""
        results = self.backend.search("*", models=[MockModel, AnotherMockModel])
        expected = MockModel.objects.count() + AnotherMockModel.objects.count()
        assert results["hits"] == expected

    def test_narrow_field_on_one_model(self):
        """Narrowing by 'author' (only on MockModel) excludes AnotherMockModel gracefully."""
        results = self.backend.search("*", narrow_queries=['author:"daniel1"'])
        assert results["hits"] > 0
        for r in results["results"]:
            assert r.model == MockModel

    def test_facets_shared_field(self):
        """Date facets on pub_date (both models) merge counts."""
        results = self.backend.search(
            "*",
            date_facets={
                "pub_date": {
                    "start_date": datetime(2009, 1, 1),
                    "end_date": datetime(2010, 1, 1),
                    "gap_by": "month",
                }
            },
        )
        buckets = results["facets"]["dates"]["pub_date"]
        total_count = sum(count for _, count in buckets)
        expected = MockModel.objects.count() + AnotherMockModel.objects.count()
        assert total_count == expected

    def test_facets_exclusive_field(self):
        """Field facets on 'author' (one model only) work without error."""
        results = self.backend.search("*", facets=["author"])
        assert "author" in results["facets"]["fields"]
        total_count = sum(c for _, c in results["facets"]["fields"]["author"])
        # Only MockModel has the author field in its index
        assert total_count == MockModel.objects.count()

    def test_highlight(self):
        """Highlighting works across models."""
        results = self.backend.search("daniel3", highlight=True)
        assert results["hits"] > 0
        highlighted_count = sum(
            1 for r in results["results"] if hasattr(r, "highlighted")
        )
        assert highlighted_count > 0

    def test_sort_by_field_on_one_model(self):
        """Sorting by 'author' works; AnotherMockModel rows get None."""
        results = self.backend.search("*", sort_by=["author"])
        authors = [getattr(r, "author", None) for r in results["results"]]
        # PostgreSQL sorts NULLs last in ASC
        non_none = [a for a in authors if a is not None]
        none_count = authors.count(None)
        assert none_count > 0
        assert authors == non_none + [None] * none_count

    def test_search_filtered_to_model(self):
        """Searching with a model filter should restrict results to that model."""
        results = self.backend.search("daniel3", models=[MockModel])
        for result in results["results"]:
            assert result.model == MockModel


# ---------------------------------------------------------------------------
# aligned_union
# ---------------------------------------------------------------------------


class TestAlignedUnion(TestCase):
    fixtures = ["bulk_data.json"]

    def setUp(self):
        self.mock_index = MockSearchIndex()
        self.another_index = AnotherMockSearchIndex()
        backend_setup(self, [self.mock_index, self.another_index])
        self.backend.update(self.mock_index, MockModel.objects.all())
        self.backend.update(self.another_index, AnotherMockModel.objects.all())
        self.mock_index_model = get_index_model(MockModel)
        self.another_index_model = get_index_model(AnotherMockModel)

    def tearDown(self):
        backend_teardown(self)

    def test_union_count_matches_sum(self):
        """Union of two querysets has count == sum of individual counts."""
        qs1 = self.mock_index_model.objects.all()
        qs2 = self.another_index_model.objects.all()
        result = qs1.aligned_union(qs2)
        assert result.count() == qs1.count() + qs2.count()

    def test_columns_are_superset(self):
        """Rows from the union contain all columns from both models."""
        qs1 = self.mock_index_model.objects.all()
        qs2 = self.another_index_model.objects.all()
        result = qs1.aligned_union(qs2)
        row = list(result)[0]
        assert "author" in row
        assert "pub_date" in row
        assert "text" in row

    def test_missing_columns_are_none(self):
        """AnotherMockModel rows have author=None in the union."""
        qs1 = self.mock_index_model.objects.all()
        qs2 = self.another_index_model.objects.all()
        result = qs1.aligned_union(qs2)
        another_ct = (
            f"{AnotherMockModel._meta.app_label}.{AnotherMockModel._meta.model_name}"
        )
        for row in result:
            if row["django_ct"] == another_ct:
                assert row["author"] is None

    def test_present_columns_retain_values(self):
        """MockModel rows have their actual author values preserved."""
        qs1 = self.mock_index_model.objects.all()
        qs2 = self.another_index_model.objects.all()
        result = qs1.aligned_union(qs2)
        mock_ct = f"{MockModel._meta.app_label}.{MockModel._meta.model_name}"
        for row in result:
            if row["django_ct"] == mock_ct:
                assert row["author"] is not None

    def test_order_by_shared_field(self):
        """.order_by('pub_date') works on the union."""
        qs1 = self.mock_index_model.objects.all()
        qs2 = self.another_index_model.objects.all()
        result = qs1.aligned_union(qs2)
        rows = list(result.order_by("pub_date"))
        dates = [r["pub_date"] for r in rows]
        assert dates == sorted(dates)

    def test_order_by_exclusive_field(self):
        """.order_by('author') works; NULLs sort last (ASC, PostgreSQL default)."""
        qs1 = self.mock_index_model.objects.all()
        qs2 = self.another_index_model.objects.all()
        result = qs1.aligned_union(qs2)
        rows = list(result.order_by("author"))
        authors = [r["author"] for r in rows]
        # PostgreSQL sorts NULLs last in ASC
        non_none = [a for a in authors if a is not None]
        none_count = authors.count(None)
        assert none_count > 0
        assert authors == non_none + [None] * none_count

    def test_slicing(self):
        """union_qs[2:5] returns exactly 3 rows."""
        qs1 = self.mock_index_model.objects.all()
        qs2 = self.another_index_model.objects.all()
        result = qs1.aligned_union(qs2)
        sliced = list(result.order_by("django_ct", "django_id")[2:5])
        assert len(sliced) == 3

    def test_pagination_no_duplicates(self):
        """Paging through the union yields no duplicate (django_ct, django_id) pairs."""
        qs1 = self.mock_index_model.objects.all()
        qs2 = self.another_index_model.objects.all()
        result = qs1.aligned_union(qs2)
        ordered = result.order_by("django_ct", "django_id")
        total = ordered.count()
        all_ids = []
        page_size = 5
        for offset in range(0, total, page_size):
            for row in ordered[offset : offset + page_size]:
                all_ids.append((row["django_ct"], row["django_id"]))
        assert len(all_ids) == total
        assert len(all_ids) == len(set(all_ids))

    def test_rank_annotation_preserved(self):
        """If querysets have rank annotations, values carry through the union."""
        qs1 = self.mock_index_model.objects.search("daniel3")
        qs2 = self.another_index_model.objects.search("daniel3")
        result = qs1.aligned_union(qs2)
        for row in result:
            assert "rank" in row
            assert row["rank"] is not None

    def test_union_with_rank_on_some(self):
        """One queryset has rank, other has Value(0); both appear."""
        qs1 = self.mock_index_model.objects.search("daniel3")
        qs2 = self.another_index_model.objects.all().annotate(
            rank=Value(0, output_field=FloatField())
        )
        result = qs1.aligned_union(qs2)
        ranks = [row["rank"] for row in result]
        assert any(r > 0 for r in ranks)
        assert any(r == 0 for r in ranks)


# ---------------------------------------------------------------------------
# Three-model aligned_union
# ---------------------------------------------------------------------------


class TestThreeModelAlignedUnion(TestCase):
    fixtures = ["bulk_data.json"]

    def setUp(self):
        self.mock_index = MockSearchIndex()
        self.another_index = AnotherMockSearchIndex()
        self.score_index = ScoreMockSearchIndex()
        backend_setup(self, [self.mock_index, self.another_index, self.score_index])
        self.backend.update(self.mock_index, MockModel.objects.all())
        self.backend.update(self.another_index, AnotherMockModel.objects.all())
        self.backend.update(self.score_index, ScoreMockModel.objects.all())
        self.mock_index_model = get_index_model(MockModel)
        self.another_index_model = get_index_model(AnotherMockModel)
        self.score_index_model = get_index_model(ScoreMockModel)

    def tearDown(self):
        backend_teardown(self)

    def test_three_way_union_count(self):
        """Union of three querysets has count == sum of individual counts."""
        qs1 = self.mock_index_model.objects.all()
        qs2 = self.another_index_model.objects.all()
        qs3 = self.score_index_model.objects.all()
        result = qs1.aligned_union(qs2).aligned_union(qs3)
        assert result.count() == qs1.count() + qs2.count() + qs3.count()

    def test_three_way_columns_are_superset(self):
        """Rows from the 3-way union contain all columns from all models."""
        qs1 = self.mock_index_model.objects.all()
        qs2 = self.another_index_model.objects.all()
        qs3 = self.score_index_model.objects.all()
        result = qs1.aligned_union(qs2).aligned_union(qs3)
        row = list(result)[0]
        # author only on MockModel, pub_date on Mock+Another, score only on ScoreMock
        assert "author" in row
        assert "pub_date" in row
        assert "score" in row
        assert "text" in row

    def test_three_way_missing_columns_are_none(self):
        """Models missing a column get None for that column."""
        qs1 = self.mock_index_model.objects.all()
        qs2 = self.another_index_model.objects.all()
        qs3 = self.score_index_model.objects.all()
        result = qs1.aligned_union(qs2).aligned_union(qs3)
        score_ct = f"{ScoreMockModel._meta.app_label}.{ScoreMockModel._meta.model_name}"
        another_ct = (
            f"{AnotherMockModel._meta.app_label}.{AnotherMockModel._meta.model_name}"
        )
        for row in result:
            if row["django_ct"] == score_ct:
                # ScoreMockModel has no author or pub_date
                assert row["author"] is None
                assert row["pub_date"] is None
            if row["django_ct"] == another_ct:
                # AnotherMockModel has no author or score
                assert row["author"] is None
                assert row["score"] is None

    def test_three_way_order_by(self):
        """Ordering by django_ct, django_id works on the 3-way union."""
        qs1 = self.mock_index_model.objects.all()
        qs2 = self.another_index_model.objects.all()
        qs3 = self.score_index_model.objects.all()
        result = qs1.aligned_union(qs2).aligned_union(qs3)
        rows = list(result.order_by("django_ct", "django_id"))
        keys = [(r["django_ct"], r["django_id"]) for r in rows]
        assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# Type round-trip
# ---------------------------------------------------------------------------


class TestTypeRoundTrip(TestCase):
    fixtures = ["bulk_data.json"]

    def setUp(self):
        self.index = MockSearchIndex()
        backend_setup(self, [self.index])
        self.backend.update(self.index, MockModel.objects.all())

    def tearDown(self):
        backend_teardown(self)

    def test_datetime_field_round_trip(self):
        """DateTimeField values should survive indexing and retrieval."""
        sqs = SearchQuerySet().all()
        result = sqs[0]
        assert isinstance(result.pub_date, datetime)

    def test_char_field_round_trip(self):
        """CharField values should survive indexing and retrieval."""
        sqs = SearchQuerySet().all()
        result = sqs[0]
        assert isinstance(result.author, str)
        assert len(result.author) > 0

    def test_stored_fields_populated(self):
        """All declared stored fields should be present on the result."""
        sqs = SearchQuerySet().all()
        result = sqs[0]
        assert hasattr(result, "author")
        assert hasattr(result, "pub_date")
        assert hasattr(result, "text")

    def test_datetime_value_matches_database(self):
        """Retrieved datetime should match the original database value."""
        sqs = SearchQuerySet().all().order_by("pub_date")
        result = sqs[0]
        obj = MockModel.objects.get(pk=result.pk)
        assert result.pub_date == obj.pub_date

    def test_char_value_matches_database(self):
        """Retrieved char field should match the original database value."""
        sqs = SearchQuerySet().all().order_by("pub_date")
        result = sqs[0]
        obj = MockModel.objects.get(pk=result.pk)
        assert result.author == obj.author


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling(TestCase):
    fixtures = ["bulk_data.json"]

    def setUp(self):
        self.index = MockSearchIndex()
        backend_setup(self, [self.index])
        self.backend.update(self.index, MockModel.objects.all())

    def tearDown(self):
        backend_teardown(self)

    def test_silently_fail_update(self):
        """With SILENTLY_FAIL=True, a DB error during update should be swallowed."""
        self.backend.silently_fail = True
        mock_model = MagicMock()
        mock_model.objects.bulk_create.side_effect = DatabaseError("test")
        with patch("postgres_fts_backend.get_index_model", return_value=mock_model):
            self.backend.update(self.index, MockModel.objects.all()[:1])

    def test_non_silent_fail_update(self):
        """With SILENTLY_FAIL=False, a DB error during update should raise."""
        self.backend.silently_fail = False
        mock_model = MagicMock()
        mock_model.objects.bulk_create.side_effect = DatabaseError("test")
        with patch("postgres_fts_backend.get_index_model", return_value=mock_model):
            with pytest.raises(DatabaseError):
                self.backend.update(self.index, MockModel.objects.all()[:1])


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestSchemaValidation(TestCase):
    fixtures = ["bulk_data.json"]

    def setUp(self):
        self.index = MockSearchIndex()
        backend_setup(self, [self.index])

    def tearDown(self):
        backend_teardown(self)

    def test_validate_schema_missing_table(self):
        """_validate_all_schemas should warn for a missing table."""
        with patch(
            "postgres_fts_backend._table_name", return_value="nonexistent_table_xyz"
        ):
            with pytest.warns(UserWarning, match="does not exist"):
                validate_all_schemas()


# ---------------------------------------------------------------------------
# build_schema
# ---------------------------------------------------------------------------


class TestBuildSchema(TestCase):
    def setUp(self):
        self.index = MockSearchIndex()
        self.another_index = AnotherMockSearchIndex()
        backend_setup(self, [self.index, self.another_index])

    def tearDown(self):
        backend_teardown(self)

    def test_keys_are_source_models(self):
        """generate_index_models() returns a dict keyed by the source Django models."""
        schema = generate_index_models()
        assert MockModel in schema
        assert AnotherMockModel in schema

    def test_generated_class_name(self):
        """The MockModel index model's __name__ is 'HaystackIndex_Core_Mockmodel'."""
        schema = generate_index_models()
        assert schema[MockModel].__name__ == "HaystackIndex_Core_Mockmodel"

    def test_table_name(self):
        """The MockModel index model's db_table is 'haystack_index_core_mockmodel'."""
        schema = generate_index_models()
        assert schema[MockModel]._meta.db_table == "haystack_index_core_mockmodel"

    def test_required_base_fields_exist(self):
        """Every generated model has django_id, django_ct, and search_vector."""
        schema = generate_index_models()
        for model_cls in schema.values():
            field_names = {f.name for f in model_cls._meta.get_fields()}
            assert "django_id" in field_names
            assert "django_ct" in field_names
            assert "search_vector" in field_names

            django_id_field = model_cls._meta.get_field("django_id")
            assert isinstance(django_id_field, models.CharField)

            django_ct_field = model_cls._meta.get_field("django_ct")
            assert isinstance(django_ct_field, models.CharField)

            sv_field = model_cls._meta.get_field("search_vector")
            assert isinstance(sv_field, SearchVectorField)

    def test_haystack_fields_are_mapped(self):
        """MockModel's index model has text (TextField), author (TextField), pub_date (DateTimeField)."""
        schema = generate_index_models()
        mock_model = schema[MockModel]

        text_field = mock_model._meta.get_field("text")
        assert isinstance(text_field, models.TextField)

        author_field = mock_model._meta.get_field("author")
        assert isinstance(author_field, models.TextField)

        pub_date_field = mock_model._meta.get_field("pub_date")
        assert isinstance(pub_date_field, models.DateTimeField)

    def test_unique_together(self):
        """_meta.unique_together is (('django_ct', 'django_id'),)."""
        schema = generate_index_models()
        assert schema[MockModel]._meta.unique_together == (("django_ct", "django_id"),)

    def test_gin_index_on_search_vector(self):
        """_meta.indexes has one GinIndex whose fields contains 'search_vector'."""
        schema = generate_index_models()
        indexes = schema[MockModel]._meta.indexes
        gin_indexes = [idx for idx in indexes if isinstance(idx, GinIndex)]
        assert len(gin_indexes) == 1
        assert "search_vector" in gin_indexes[0].fields

    def test_cache_returns_same_object(self):
        """Calling generate_index_models() twice returns identical model objects."""
        schema_a = generate_index_models()
        schema_b = generate_index_models()
        assert schema_a[MockModel] is schema_b[MockModel]
        assert schema_a[AnotherMockModel] is schema_b[AnotherMockModel]


# ---------------------------------------------------------------------------
# Migration change detection
# ---------------------------------------------------------------------------


class TestMigrationChangeDetection(TestCase):
    INITIAL_MIGRATION = os.path.join(
        os.path.dirname(__file__), "search_migrations", "0001_initial.py"
    )

    def setUp(self):
        self.models_module = models_module
        self._saved_cache = models_module._index_models_cache.copy()
        models_module._index_models_cache.clear()

        self.index = MockSearchIndex()
        self.another_index = AnotherMockSearchIndex()
        self.score_index = ScoreMockSearchIndex()

    def tearDown(self):
        self.models_module._index_models_cache.clear()
        self.models_module._index_models_cache.update(self._saved_cache)

    def _make_migration_package(self, module_name, copy_initial=True):
        """Create a temp dir importable as *module_name* with an __init__.py.

        If *copy_initial* is True the existing 0001_initial.py is copied in so
        the autodetector sees the current schema as its baseline.

        Returns the path to the package directory.
        """
        parent = tempfile.mkdtemp()
        pkg_dir = os.path.join(parent, module_name)
        os.mkdir(pkg_dir)
        with open(os.path.join(pkg_dir, "__init__.py"), "w") as f:
            f.write("")

        if copy_initial:
            shutil.copy(self.INITIAL_MIGRATION, pkg_dir)

        sys.path.insert(0, parent)
        # Ensure Python doesn't cache a missing/stale module entry
        sys.modules.pop(module_name, None)

        self.addCleanup(shutil.rmtree, parent, True)
        self.addCleanup(sys.path.remove, parent)
        self.addCleanup(sys.modules.pop, module_name, None)
        return pkg_dir

    def test_adding_field_generates_add_field_migration(self):
        """Registering an extended index with an extra IntegerField generates an AddField migration."""

        class ExtendedMockSearchIndex(MockSearchIndex):
            extra_field = haystack_indexes.IntegerField(default=0)

        backend_setup(
            self, [ExtendedMockSearchIndex(), self.another_index, self.score_index]
        )
        self.models_module._index_models_cache.clear()

        module_name = "temp_mig_addfield"
        pkg_dir = self._make_migration_package(module_name)

        with override_settings(MIGRATION_MODULES={"postgres_fts_backend": module_name}):
            call_command("build_postgres_schema", verbosity=0)

            migration_files = [
                f
                for f in os.listdir(pkg_dir)
                if f.endswith(".py") and f != "__init__.py" and f != "0001_initial.py"
            ]
            assert len(migration_files) > 0, "No migration file generated"

            migration_content = open(os.path.join(pkg_dir, migration_files[0])).read()
            assert "AddField" in migration_content
            assert "extra_field" in migration_content

        backend_teardown(self)

    def test_generated_migration_applies_cleanly(self):
        """After generating a migration for an added field, migrate succeeds and the column exists."""

        class ExtendedMockSearchIndex(MockSearchIndex):
            extra_field = haystack_indexes.IntegerField(default=0)

        backend_setup(
            self, [ExtendedMockSearchIndex(), self.another_index, self.score_index]
        )
        self.models_module._index_models_cache.clear()

        module_name = "temp_mig_apply"
        self._make_migration_package(module_name)

        with override_settings(MIGRATION_MODULES={"postgres_fts_backend": module_name}):
            call_command("build_postgres_schema", verbosity=0)
            call_command("migrate", "postgres_fts_backend", verbosity=0)

            # Verify the column exists
            with connection.cursor() as cursor:
                columns = {
                    info.name
                    for info in connection.introspection.get_table_description(
                        cursor, "haystack_index_core_mockmodel"
                    )
                }
            assert "extra_field" in columns

            # Roll back: restore original indexes, regenerate, and migrate
            self.models_module._index_models_cache.clear()
            backend_teardown(self)
            backend_setup(self, [self.index, self.another_index, self.score_index])
            self.models_module._index_models_cache.clear()

            call_command("build_postgres_schema", verbosity=0)
            call_command("migrate", "postgres_fts_backend", verbosity=0)

            # Verify extra_field is gone
            with connection.cursor() as cursor:
                columns = {
                    info.name
                    for info in connection.introspection.get_table_description(
                        cursor, "haystack_index_core_mockmodel"
                    )
                }
            assert "extra_field" not in columns

        backend_teardown(self)

    def test_no_changes_detected_when_schema_matches(self):
        """With standard indexes, build_postgres_schema outputs 'No changes detected'."""
        backend_setup(self, [self.index, self.another_index, self.score_index])

        module_name = "temp_mig_nochange"
        pkg_dir = self._make_migration_package(module_name)

        with override_settings(MIGRATION_MODULES={"postgres_fts_backend": module_name}):
            out = StringIO()
            call_command("build_postgres_schema", stdout=out, verbosity=1)
            output = out.getvalue()
            assert "No changes detected" in output

            # Verify no new migration files were written
            migration_files = [
                f
                for f in os.listdir(pkg_dir)
                if f.endswith(".py") and f != "__init__.py" and f != "0001_initial.py"
            ]
            assert len(migration_files) == 0

        backend_teardown(self)


# ---------------------------------------------------------------------------
# Spelling suggestions
# ---------------------------------------------------------------------------


class TestSpellingSuggestion(TestCase):
    fixtures = ["bulk_data.json"]

    def setUp(self):
        self.index = MockSearchIndex()
        backend_setup(self, [self.index])
        self.backend.update(self.index, MockModel.objects.all())

    def tearDown(self):
        backend_teardown(self)

    def test_backend_returns_spelling_suggestion_key(self):
        """search() result dict should contain 'spelling_suggestion' key set to None."""
        result = self.backend.search("indexing", spelling_query="indexxing")
        assert "spelling_suggestion" in result
        assert result["spelling_suggestion"] is None

    def test_sqs_spelling_suggestion_returns_none(self):
        """spelling_suggestion() through the SQS layer should return None."""
        result = SearchQuerySet().auto_query("indexing").spelling_suggestion()
        assert result is None


# ---------------------------------------------------------------------------
# Pagination regression — no duplicate results
# ---------------------------------------------------------------------------


class TestPaginationNoDuplicates(TestCase):
    fixtures = ["bulk_data.json"]

    def setUp(self):
        self.index = MockSearchIndex()
        backend_setup(self, [self.index])
        self.backend.update(self.index, MockModel.objects.all())

    def tearDown(self):
        backend_teardown(self)

    def test_no_duplicate_results_across_pages(self):
        """Paging through all results with page_size=10 should yield no duplicates."""
        page_size = 10
        all_ids = []
        offset = 0
        total = self.backend.search("*")["hits"]

        while offset < total:
            page = self.backend.search(
                "*", start_offset=offset, end_offset=offset + page_size
            )
            for result in page["results"]:
                all_ids.append((result.app_label, result.model_name, result.pk))
            offset += page_size

        assert len(all_ids) == total
        assert len(all_ids) == len(set(all_ids))


# ---------------------------------------------------------------------------
# Complex & / | with SQ and excludes
# ---------------------------------------------------------------------------


class TestComplexSQCombinations(TestCase):
    fixtures = ["bulk_data.json"]

    def setUp(self):
        self.index = MockSearchIndex()
        backend_setup(self, [self.index])
        self.backend.update(self.index, MockModel.objects.all())

    def tearDown(self):
        backend_teardown(self)

    def test_and_with_sq_excludes(self):
        """AND combining exclude+SQ filter with an exact author filter."""
        sqs1 = (
            SearchQuerySet()
            .exclude(author__exact="daniel1")
            .filter(SQ(content="index") | SQ(content="search"))
        )
        sqs2 = SearchQuerySet().filter(author__exact="daniel3")
        combined = sqs1 & sqs2
        assert len(combined) > 0
        for result in combined:
            obj = MockModel.objects.get(pk=result.pk)
            assert obj.author == "daniel3"

    def test_or_with_sq_excludes(self):
        """OR combining exclude+SQ filter with an exact author filter."""
        sqs1 = (
            SearchQuerySet()
            .exclude(author="daniel1")
            .filter(SQ(content="index") | SQ(content="search"))
        )
        sqs2 = SearchQuerySet().filter(author__exact="daniel3")
        combined = sqs1 | sqs2
        assert len(combined) > 0
        for result in combined:
            obj = MockModel.objects.get(pk=result.pk)
            is_side1 = obj.author != "daniel1"
            is_side2 = obj.author == "daniel3"
            assert is_side1 or is_side2


# ---------------------------------------------------------------------------
# Multiple narrow queries stacked
# ---------------------------------------------------------------------------


class TestMultipleNarrows(TestCase):
    fixtures = ["bulk_data.json"]

    def setUp(self):
        self.index = MockSearchIndex()
        backend_setup(self, [self.index])
        self.backend.update(self.index, MockModel.objects.all())

    def tearDown(self):
        backend_teardown(self)

    def test_single_narrow(self):
        """Narrowing to author:daniel1 should return 7 results."""
        sqs = SearchQuerySet().all().narrow('author:"daniel1"')
        assert len(sqs) == 7

    def test_stacked_narrows_are_anded(self):
        """Narrowing to two different authors should return 0 (ANDed, no record has both)."""
        sqs = (
            SearchQuerySet().all().narrow('author:"daniel1"').narrow('author:"daniel2"')
        )
        assert len(sqs) == 0


# ---------------------------------------------------------------------------
# Pickling
# ---------------------------------------------------------------------------


class TestPickling(TestCase):
    fixtures = ["bulk_data.json"]

    def setUp(self):
        self.index = MockSearchIndex()
        backend_setup(self, [self.index])
        self.backend.update(self.index, MockModel.objects.all())

    def tearDown(self):
        backend_teardown(self)

    def test_pickle_search_result(self):
        """A single SearchResult should survive pickle/unpickle."""
        results = self.backend.search("indexing")
        original = results["results"][0]

        restored = pickle.loads(pickle.dumps(original))

        assert restored.pk == original.pk
        assert restored.score == original.score
        assert restored.app_label == original.app_label
        assert restored.author == original.author

    def test_pickle_search_queryset_results(self):
        """A list of SQS results should survive pickle/unpickle."""
        original_results = list(SearchQuerySet().auto_query("indexing"))
        assert len(original_results) > 0

        restored_results = pickle.loads(pickle.dumps(original_results))

        assert len(restored_results) == len(original_results)
        assert [r.pk for r in restored_results] == [r.pk for r in original_results]


# ---------------------------------------------------------------------------
# Query logging
# ---------------------------------------------------------------------------


class TestQueryLogging(TestCase):
    fixtures = ["bulk_data.json"]

    def setUp(self):
        self.index = MockSearchIndex()
        backend_setup(self, [self.index])
        self.backend.update(self.index, MockModel.objects.all())

    def tearDown(self):
        backend_teardown(self)

    @override_settings(DEBUG=False)
    def test_no_logging_when_debug_false(self):
        """With DEBUG=False, connections['default'].queries should stay empty."""
        connections["default"].queries = []
        self.backend.search("indexing")
        assert len(connections["default"].queries) == 0

    @override_settings(DEBUG=True)
    def test_logging_when_debug_true(self):
        """With DEBUG=True, a search should log one query entry."""
        connections["default"].queries = []
        self.backend.search("indexing")
        assert len(connections["default"].queries) == 1
        assert connections["default"].queries[0]["query_string"] == "indexing"


# ---------------------------------------------------------------------------
# Non-silent fail for search, remove, clear
# ---------------------------------------------------------------------------


class TestNonSilentFail(TestCase):
    fixtures = ["bulk_data.json"]

    def setUp(self):
        self.index = MockSearchIndex()
        backend_setup(self, [self.index])
        self.backend.update(self.index, MockModel.objects.all())

    def tearDown(self):
        backend_teardown(self)

    def test_search_silently_fail_true(self):
        """search with silently_fail=True should return empty results on error."""
        self.backend.silently_fail = True
        mock_model = MagicMock()
        mock_model.objects.all.side_effect = DatabaseError("test")
        mock_model.objects.search.side_effect = DatabaseError("test")
        mock_model.objects.filter.side_effect = DatabaseError("test")
        with patch("postgres_fts_backend.get_index_model", return_value=mock_model):
            result = self.backend.search("indexing")
        assert result["hits"] == 0
        assert result["results"] == []

    def test_search_silently_fail_false(self):
        """search with silently_fail=False should raise DatabaseError."""
        self.backend.silently_fail = False
        mock_model = MagicMock()
        mock_model.objects.all.side_effect = DatabaseError("test")
        mock_model.objects.search.side_effect = DatabaseError("test")
        mock_model.objects.filter.side_effect = DatabaseError("test")
        with patch("postgres_fts_backend.get_index_model", return_value=mock_model):
            with pytest.raises(DatabaseError):
                self.backend.search("indexing")

    def test_remove_silently_fail_true(self):
        """remove with silently_fail=True should not raise on error."""
        self.backend.silently_fail = True
        mock_model = MagicMock()
        mock_model.objects.filter.return_value.delete.side_effect = DatabaseError(
            "test"
        )
        with patch("postgres_fts_backend.get_index_model", return_value=mock_model):
            self.backend.remove(MockModel.objects.first())

    def test_remove_silently_fail_false(self):
        """remove with silently_fail=False should raise DatabaseError."""
        self.backend.silently_fail = False
        mock_model = MagicMock()
        mock_model.objects.filter.return_value.delete.side_effect = DatabaseError(
            "test"
        )
        with patch("postgres_fts_backend.get_index_model", return_value=mock_model):
            with pytest.raises(DatabaseError):
                self.backend.remove(MockModel.objects.first())

    def test_clear_silently_fail_true(self):
        """clear with silently_fail=True should not raise on error."""
        self.backend.silently_fail = True
        mock_model = MagicMock()
        mock_model.objects.all.return_value.delete.side_effect = DatabaseError("test")
        with patch("postgres_fts_backend.get_index_model", return_value=mock_model):
            self.backend.clear()

    def test_clear_silently_fail_false(self):
        """clear with silently_fail=False should raise DatabaseError."""
        self.backend.silently_fail = False
        mock_model = MagicMock()
        mock_model.objects.all.return_value.delete.side_effect = DatabaseError("test")
        with patch("postgres_fts_backend.get_index_model", return_value=mock_model):
            with pytest.raises(DatabaseError):
                self.backend.clear()


# ---------------------------------------------------------------------------
# RelatedSearchQuerySet / load_all with queryset
# ---------------------------------------------------------------------------


class TestRelatedSearchQuerySet(TestCase):
    fixtures = ["bulk_data.json"]

    def setUp(self):
        self.index = MockSearchIndex()
        backend_setup(self, [self.index])
        self.backend.update(self.index, MockModel.objects.all())

    def tearDown(self):
        backend_teardown(self)

    def test_load_all_populates_objects(self):
        """RelatedSearchQuerySet.load_all() should populate .object on results."""
        sqs = RelatedSearchQuerySet().all().load_all()
        results = list(sqs)
        assert len(results) > 0
        for result in results:
            assert result.object is not None
            assert isinstance(result.object, MockModel)

    def test_load_all_queryset_filters_objects(self):
        """load_all_queryset should restrict which objects get loaded."""
        sqs = (
            RelatedSearchQuerySet()
            .all()
            .load_all()
            .load_all_queryset(MockModel, MockModel.objects.filter(author="daniel1"))
        )
        results = list(sqs)
        assert len(results) > 0
        for result in results:
            if result.object is not None:
                assert result.object.author == "daniel1"


# ---------------------------------------------------------------------------
# GIN index usage
# ---------------------------------------------------------------------------


class TestGinIndexUsage(TestCase):
    fixtures = ["bulk_data.json"]

    def setUp(self):
        self.index = MockSearchIndex()
        backend_setup(self, [self.index])
        self.backend.update(self.index, MockModel.objects.all())

    def tearDown(self):
        backend_teardown(self)

    def _explain_without_seqscan(self, qs):
        """Run EXPLAIN with sequential scans disabled to force index usage."""
        with connection.cursor() as cursor:
            cursor.execute("SET enable_seqscan = off")
        try:
            return qs.explain()
        finally:
            with connection.cursor() as cursor:
                cursor.execute("SET enable_seqscan = on")

    def test_search_uses_gin_index(self):
        """Full-text search should use the GIN index on search_vector."""
        index_model = get_index_model(MockModel)
        qs = index_model.objects.search("indexing")
        plan = self._explain_without_seqscan(qs)
        assert "haystack_index_core_mockmodel_sv_gin" in plan

    def test_ranked_query_uses_gin_index(self):
        """The ORM query path (filter + ranked) should use the GIN index."""
        index_model = get_index_model(MockModel)
        sq = SearchQuery("indexing", search_type="websearch", config="english")
        qs = index_model.objects.filter(search_vector=sq).ranked("indexing")
        plan = self._explain_without_seqscan(qs)
        assert "haystack_index_core_mockmodel_sv_gin" in plan


# ---------------------------------------------------------------------------
# Trigram (pg_trgm) index support
# ---------------------------------------------------------------------------


class TestTrigramSupport(TestCase):
    """Tests for pg_trgm trigram index support with EdgeNgramField."""

    fixtures = ["bulk_data.json"]

    INITIAL_MIGRATION = os.path.join(
        os.path.dirname(__file__), "search_migrations", "0001_initial.py"
    )

    def setUp(self):
        self.models_module = models_module
        self._saved_cache = models_module._index_models_cache.copy()
        models_module._index_models_cache.clear()
        self.another_index = AnotherMockSearchIndex()
        self.score_index = ScoreMockSearchIndex()

    def tearDown(self):
        self.models_module._index_models_cache.clear()
        self.models_module._index_models_cache.update(self._saved_cache)

    def _ngram_index(self):
        class NgramMockSearchIndex(MockSearchIndex):
            author = haystack_indexes.EdgeNgramField(model_attr="author")

        return NgramMockSearchIndex()

    def _make_migration_package(self, module_name, copy_initial=True):
        parent = tempfile.mkdtemp()
        pkg_dir = os.path.join(parent, module_name)
        os.mkdir(pkg_dir)
        with open(os.path.join(pkg_dir, "__init__.py"), "w") as f:
            f.write("")
        if copy_initial:
            shutil.copy(self.INITIAL_MIGRATION, pkg_dir)
        sys.path.insert(0, parent)
        sys.modules.pop(module_name, None)
        self.addCleanup(shutil.rmtree, parent, True)
        self.addCleanup(sys.path.remove, parent)
        self.addCleanup(sys.modules.pop, module_name, None)
        return pkg_dir

    def _explain_without_seqscan(self, qs):
        with connection.cursor() as cursor:
            cursor.execute("SET enable_seqscan = off")
        try:
            return qs.explain()
        finally:
            with connection.cursor() as cursor:
                cursor.execute("SET enable_seqscan = on")

    # --- Schema generation ---

    def test_generate_index_models_includes_trigram_index(self):
        """EdgeNgramField gets a GIN trigram index when pg_trgm is installed."""
        ngram_index = self._ngram_index()
        backend_setup(self, [ngram_index, self.another_index, self.score_index])
        try:
            schema = generate_index_models()
            indexes = schema[MockModel]._meta.indexes
            trgm = [i for i in indexes if "trgm" in i.name]
            assert len(trgm) == 1
            assert "author_trgm" in trgm[0].name
        finally:
            backend_teardown(self)

    # --- Migration generation ---

    def test_migration_adds_trigram_index(self):
        """Switching author to EdgeNgramField generates an AddIndex with gin_trgm_ops."""
        ngram_index = self._ngram_index()
        backend_setup(self, [ngram_index, self.another_index, self.score_index])
        try:
            module_name = "temp_mig_trgm_add"
            pkg_dir = self._make_migration_package(module_name)
            with override_settings(
                MIGRATION_MODULES={"postgres_fts_backend": module_name}
            ):
                call_command("build_postgres_schema", verbosity=0)
                migration_files = [
                    f
                    for f in os.listdir(pkg_dir)
                    if f.endswith(".py") and f not in ("__init__.py", "0001_initial.py")
                ]
                assert len(migration_files) > 0, "No migration generated"
                content = open(os.path.join(pkg_dir, migration_files[0])).read()
                assert "gin_trgm_ops" in content
                assert "author_trgm" in content
        finally:
            backend_teardown(self)

    # --- Index usage ---

    def test_trigram_index_used_for_similarity_query(self):
        """EXPLAIN shows the GIN trigram index for a trigram_similar lookup."""
        ngram_index = self._ngram_index()
        backend_setup(self, [ngram_index, self.another_index, self.score_index])
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS "
                    "haystack_index_core_mockmodel_author_trgm "
                    "ON haystack_index_core_mockmodel "
                    "USING gin (author gin_trgm_ops)"
                )

            self.backend.update(ngram_index, MockModel.objects.all())

            index_model = get_index_model(MockModel)
            qs = index_model.objects.filter(author__trigram_similar="daniel")
            plan = self._explain_without_seqscan(qs)
            assert "haystack_index_core_mockmodel_author_trgm" in plan
        finally:
            with connection.cursor() as cursor:
                cursor.execute(
                    "DROP INDEX IF EXISTS " "haystack_index_core_mockmodel_author_trgm"
                )
            backend_teardown(self)


# ---------------------------------------------------------------------------
# Adversarial inputs
# ---------------------------------------------------------------------------


class TestAdversarialInputs(TestCase):
    """Tests for malformed, unusual, or hostile inputs.

    These target error-handling gaps where exceptions other than DatabaseError
    escape the silently_fail guard.
    """

    fixtures = ["bulk_data.json"]

    def setUp(self):
        self.index = MockSearchIndex()
        self.another_index = AnotherMockSearchIndex()
        backend_setup(self, [self.index, self.another_index])
        self.backend.update(self.index, MockModel.objects.all())
        self.backend.update(self.another_index, AnotherMockModel.objects.all())
        self.backend.silently_fail = True

    def tearDown(self):
        backend_teardown(self)

    # --- Narrow query parsing ---

    def test_malformed_narrow_query(self):
        """A narrow query that doesn't match field:"value" should not crash."""
        result = self.backend.search("*", narrow_queries=["not a valid narrow"])
        assert result["hits"] == 0 or result["results"] == []

    def test_narrow_query_missing_quotes(self):
        """Narrow query without quotes should not crash."""
        result = self.backend.search("*", narrow_queries=["author:daniel1"])
        assert result["hits"] == 0 or result["results"] == []

    def test_narrow_query_empty_value(self):
        """Narrow query with empty quoted value should not crash."""
        result = self.backend.search("*", narrow_queries=['author:""'])
        assert result["hits"] == 0 or result["results"] == []

    # --- Colon edge cases in query strings ---

    def test_search_colon_only(self):
        """Searching for ':' alone should not crash."""
        result = self.backend.search(":")
        assert isinstance(result["hits"], int)

    def test_search_leading_colon(self):
        """Searching for ':value' (empty field name) should not crash."""
        result = self.backend.search(":value")
        assert isinstance(result["hits"], int)

    def test_search_trailing_colon(self):
        """Searching for 'field:' (empty value) should not crash."""
        result = self.backend.search("author:")
        assert isinstance(result["hits"], int)

    def test_search_multiple_colons(self):
        """Searching for 'a:b:c' should not crash."""
        result = self.backend.search("a:b:c")
        assert isinstance(result["hits"], int)

    # --- remove() with bad identifiers ---
    # Per haystack convention, invalid identifiers are programming errors
    # and should raise regardless of silently_fail.

    def test_remove_invalid_model_string(self):
        """remove('nonexistent.model.1') raises — invalid model is a programming error."""
        with pytest.raises(LookupError):
            self.backend.remove("nonexistent.model.1")

    def test_remove_unindexed_model_string(self):
        """remove() for a model not in the search index raises."""
        with pytest.raises(Exception):
            self.backend.remove("auth.user.1")

    def test_remove_empty_pk(self):
        """remove('core.mockmodel.') is a no-op (matches nothing)."""
        self.backend.remove("core.mockmodel.")

    def test_remove_too_few_parts(self):
        """remove('just_a_string') raises — invalid format is a programming error."""
        with pytest.raises(ValueError):
            self.backend.remove("just_a_string")

    # --- Websearch syntax edge cases ---

    def test_search_unmatched_quote(self):
        """Unmatched double-quote should not crash."""
        result = self.backend.search('"unclosed phrase')
        assert isinstance(result["hits"], int)

    def test_search_only_operators(self):
        """Search for 'OR' (a websearch operator) alone should not crash."""
        result = self.backend.search("OR")
        assert isinstance(result["hits"], int)

    def test_search_negation_only(self):
        """Search for '-everything' should not crash."""
        result = self.backend.search("-everything")
        assert isinstance(result["hits"], int)

    def test_search_special_characters(self):
        """Search with brackets, pipes, ampersands should not crash."""
        result = self.backend.search("(foo) & [bar] | <baz>")
        assert isinstance(result["hits"], int)

    def test_search_null_byte(self):
        """Null bytes in search text should not crash with silently_fail."""
        result = self.backend.search("test\x00injection")
        assert isinstance(result["hits"], int)

    def test_search_sql_injection_attempt(self):
        """SQL injection via search text is neutralized by parameterized queries."""
        result = self.backend.search("'; DROP TABLE haystack_index_core_mockmodel; --")
        assert isinstance(result["hits"], int)
        # Verify the table still exists
        index_model = get_index_model(MockModel)
        assert index_model.objects.count() > 0

    def test_search_backslash_heavy(self):
        """Backslashes should not break websearch parsing."""
        result = self.backend.search("test\\value\\\\more")
        assert isinstance(result["hits"], int)

    def test_search_very_long_query(self):
        """A very long query string should not crash."""
        result = self.backend.search("word " * 5000)
        assert isinstance(result["hits"], int)

    def test_search_unicode_emoji(self):
        """Emoji in search text should not crash."""
        result = self.backend.search("\U0001f600 \U0001f4a9 \U0001f680")
        assert isinstance(result["hits"], int)

    def test_search_unicode_cjk(self):
        """CJK characters in search text should not crash."""
        result = self.backend.search("\u4e16\u754c\u4f60\u597d")
        assert isinstance(result["hits"], int)

    def test_search_unicode_combining_chars(self):
        """Combining characters (e.g. accented e as e + combining accent) should not crash."""
        result = self.backend.search("caf\u0065\u0301")  # NFD form of café
        assert isinstance(result["hits"], int)

    def test_search_tabs_and_newlines(self):
        """Whitespace variations in query should not crash."""
        result = self.backend.search("test\t\nvalue\r\n")
        assert isinstance(result["hits"], int)

    def test_search_zero_width_chars(self):
        """Zero-width characters should not crash."""
        result = self.backend.search("te\u200bst\u200dval\u200cue")
        assert isinstance(result["hits"], int)

    # --- Multi-model edge cases ---

    def test_multi_model_highlight(self):
        """Highlighting in multi-model search should not crash."""
        result = self.backend.search("daniel3", highlight=True)
        assert result["hits"] > 0
        has_highlight = False
        for r in result["results"]:
            if hasattr(r, "highlighted") and r.highlighted:
                has_highlight = True
        assert has_highlight

    def test_multi_model_boost(self):
        """Boosting in multi-model search should not crash."""
        result = self.backend.search("daniel3", boost={"indexing": 2.0})
        assert result["hits"] > 0

    def test_multi_model_search_match_one_model_only(self):
        """Search matching only one model in a multi-model setup returns results."""
        # "daniel3" exists in both models via fixture data,
        # so narrow to author (only on MockModel) to get single-model results
        result = self.backend.search(
            "daniel3", narrow_queries=['author:"daniel3"']
        )
        assert result["hits"] > 0
        for r in result["results"]:
            assert r.model == MockModel

    def test_multi_model_facet_on_single_model_field(self):
        """Faceting on a field that only one model has should work."""
        result = self.backend.search("*", facets=["author"])
        assert "facets" in result
        assert "fields" in result["facets"]
        assert "author" in result["facets"]["fields"]

    # --- SearchQuerySet API edge cases ---

    def test_sqs_auto_query_sql_injection(self):
        """auto_query with SQL injection attempt should not crash."""
        sqs = SearchQuerySet().auto_query("'; DROP TABLE users; --")
        assert isinstance(len(sqs), int)

    def test_sqs_auto_query_all_negated(self):
        """auto_query where every term is negated should not crash."""
        sqs = SearchQuerySet().auto_query("-this -that -other")
        assert isinstance(len(sqs), int)

    def test_sqs_auto_query_only_quotes(self):
        """auto_query with only quote characters should not crash."""
        sqs = SearchQuerySet().auto_query('"""')
        assert isinstance(len(sqs), int)

    def test_sqs_filter_empty_string(self):
        """Filtering content for empty string should not crash."""
        sqs = SearchQuerySet().filter(content="")
        assert isinstance(len(sqs), int)

    def test_sqs_slice_beyond_results(self):
        """Slicing past the end of results returns empty, not error."""
        sqs = SearchQuerySet().all()
        sliced = list(sqs[99999:99999 + 10])
        assert sliced == []

    def test_sqs_order_then_filter(self):
        """Order followed by filter should not crash."""
        sqs = SearchQuerySet().order_by("pub_date").filter(content="indexing")
        assert isinstance(len(sqs), int)

    # --- Boost edge cases ---

    def test_boost_empty_dict(self):
        """Boost with empty dict should be a no-op."""
        result = self.backend.search("indexing", boost={})
        assert result["hits"] > 0

    def test_boost_zero_weight(self):
        """Boost with zero weight should not crash."""
        result = self.backend.search("indexing", boost={"test": 0.0})
        assert result["hits"] > 0

    def test_boost_negative_weight(self):
        """Boost with negative weight should not crash."""
        result = self.backend.search("indexing", boost={"test": -1.0})
        assert result["hits"] > 0

    # --- Update/remove edge cases ---

    def test_update_then_search_immediately(self):
        """Documents are searchable immediately after update."""
        result = self.backend.search("daniel1")
        assert result["hits"] > 0

    def test_remove_nonexistent_document(self):
        """Removing a document that doesn't exist is a no-op."""
        obj = MockModel.objects.first()
        self.backend.remove(obj)
        # Remove again — should not crash
        self.backend.remove(obj)

    def test_update_idempotent(self):
        """Updating the same documents twice doesn't create duplicates."""
        self.backend.update(self.index, MockModel.objects.all())
        self.backend.update(self.index, MockModel.objects.all())
        result = self.backend.search("*")
        assert result["hits"] == MockModel.objects.count() + AnotherMockModel.objects.count()


# ---------------------------------------------------------------------------
# Search and retrieval correctness
# ---------------------------------------------------------------------------


class TestSearchCorrectness(TestCase):
    """Verify that searches return the RIGHT results — correct documents,
    correct fields, correct scores, correct order."""

    fixtures = ["bulk_data.json"]

    def setUp(self):
        self.mock_index = MockSearchIndex()
        self.another_index = AnotherMockSearchIndex()
        self.score_index = ScoreMockSearchIndex()
        backend_setup(
            self,
            [self.mock_index, self.another_index, self.score_index],
        )
        self.backend.update(self.mock_index, MockModel.objects.all())
        self.backend.update(self.another_index, AnotherMockModel.objects.all())
        self.backend.update(self.score_index, ScoreMockModel.objects.all())

    def tearDown(self):
        backend_teardown(self)

    # --- Field value correctness ---

    def test_author_field_matches_database(self):
        """Every MockModel result's author field should match the database."""
        results = self.backend.search("*", models=[MockModel])
        assert results["hits"] == MockModel.objects.count()
        for r in results["results"]:
            db_obj = MockModel.objects.get(pk=r.pk)
            assert r.author == db_obj.author

    def test_pub_date_field_matches_database(self):
        """Every MockModel result's pub_date should match the database."""
        results = self.backend.search("*", models=[MockModel])
        for r in results["results"]:
            db_obj = MockModel.objects.get(pk=r.pk)
            assert r.pub_date == db_obj.pub_date

    def test_result_attributes(self):
        """Each result has correct app_label, model_name, and model class."""
        results = self.backend.search("*", models=[MockModel])
        for r in results["results"]:
            assert r.app_label == "core"
            assert r.model_name == "mockmodel"
            assert r.model == MockModel

    # --- Filter correctness ---

    def test_filter_exact_returns_correct_pks(self):
        """filter(author='daniel1') returns exactly the right PKs."""
        expected_pks = set(
            MockModel.objects.filter(author="daniel1").values_list("pk", flat=True)
        )
        sqs = SearchQuerySet().models(MockModel).filter(author__exact="daniel1")
        result_pks = {int(r.pk) for r in sqs}
        assert result_pks == expected_pks

    def test_filter_in_returns_correct_pks(self):
        """filter(author__in=['daniel1', 'daniel2']) returns their union."""
        expected_pks = set(
            MockModel.objects.filter(author__in=["daniel1", "daniel2"]).values_list(
                "pk", flat=True
            )
        )
        sqs = SearchQuerySet().models(MockModel).filter(
            author__in=["daniel1", "daniel2"]
        )
        result_pks = {int(r.pk) for r in sqs}
        assert result_pks == expected_pks

    def test_exclude_removes_correct_pks(self):
        """exclude(author='daniel1') returns everyone else."""
        excluded_pks = set(
            MockModel.objects.filter(author="daniel1").values_list("pk", flat=True)
        )
        sqs = SearchQuerySet().models(MockModel).exclude(author__exact="daniel1")
        result_pks = {int(r.pk) for r in sqs}
        assert result_pks & excluded_pks == set()
        assert len(result_pks) == MockModel.objects.exclude(author="daniel1").count()

    def test_date_boundary_filter(self):
        """pub_date__lt=2009-07-01 returns only the June documents."""
        cutoff = datetime(2009, 7, 1)
        expected_pks = set(
            MockModel.objects.filter(pub_date__lt=cutoff).values_list("pk", flat=True)
        )
        assert len(expected_pks) > 0  # Sanity: some docs are in June
        sqs = SearchQuerySet().models(MockModel).filter(pub_date__lt=cutoff)
        result_pks = {int(r.pk) for r in sqs}
        assert result_pks == expected_pks

    def test_combined_filters_intersect(self):
        """author='daniel3' AND pub_date before July returns the intersection."""
        cutoff = datetime(2009, 7, 1)
        expected_pks = set(
            MockModel.objects.filter(author="daniel3", pub_date__lt=cutoff).values_list(
                "pk", flat=True
            )
        )
        sqs = SearchQuerySet().models(MockModel).filter(
            author__exact="daniel3", pub_date__lt=cutoff
        )
        result_pks = {int(r.pk) for r in sqs}
        assert result_pks == expected_pks

    # --- Scoring correctness ---

    def test_scores_are_nonnegative(self):
        """All search scores should be >= 0."""
        results = self.backend.search("indexing")
        for r in results["results"]:
            assert r.score >= 0

    def test_scores_strictly_sorted(self):
        """Scores should be in non-increasing order."""
        results = self.backend.search("indexing")
        scores = [r.score for r in results["results"]]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    def test_wildcard_search_scores_are_zero(self):
        """Wildcard '*' search assigns score=0 to all results."""
        results = self.backend.search("*")
        for r in results["results"]:
            assert r.score == 0

    # --- Phrase search correctness ---

    def test_phrase_search_is_subset_of_word_search(self):
        """Phrase results should be a subset of individual word results."""
        word_results = self.backend.search("search backend")
        phrase_results = self.backend.search('"search backend"')
        word_pks = {(r.app_label, r.pk) for r in word_results["results"]}
        phrase_pks = {(r.app_label, r.pk) for r in phrase_results["results"]}
        assert phrase_pks <= word_pks

    def test_phrase_search_more_restrictive(self):
        """Phrase search should match fewer or equal documents than word search."""
        word_hits = self.backend.search("search backend")["hits"]
        phrase_hits = self.backend.search('"search backend"')["hits"]
        assert phrase_hits <= word_hits

    # --- Highlighting correctness ---

    def test_highlight_contains_markup(self):
        """Highlighted text should contain <b> tags from ts_headline."""
        results = self.backend.search("indexing", highlight=True, models=[MockModel])
        assert results["hits"] > 0
        found_markup = False
        for r in results["results"]:
            highlighted = getattr(r, "highlighted", None)
            if highlighted:
                for field, fragments in highlighted.items():
                    for frag in fragments:
                        if "<b>" in frag:
                            found_markup = True
        assert found_markup, "No <b> markup found in any highlighted result"

    def test_highlight_contains_search_term(self):
        """The highlighted fragment should contain the stemmed search term."""
        results = self.backend.search("indexing", highlight=True, models=[MockModel])
        found_term = False
        for r in results["results"]:
            highlighted = getattr(r, "highlighted", None)
            if highlighted:
                for fragments in highlighted.values():
                    for frag in fragments:
                        if "index" in frag.lower():
                            found_term = True
        assert found_term, "Search term not found in any highlighted fragment"

    def test_highlight_field_key_is_content_field(self):
        """The highlighted dict key should be the content field name."""
        results = self.backend.search("indexing", highlight=True, models=[MockModel])
        for r in results["results"]:
            highlighted = getattr(r, "highlighted", None)
            if highlighted:
                assert "text" in highlighted

    # --- Boost correctness ---

    def test_boost_changes_ranking(self):
        """Boosting a term should change the result order."""
        unboosted = self.backend.search("indexing", models=[MockModel])
        boosted = self.backend.search(
            "indexing", models=[MockModel], boost={"template": 10.0}
        )
        unboosted_pks = [r.pk for r in unboosted["results"]]
        boosted_pks = [r.pk for r in boosted["results"]]
        assert unboosted_pks != boosted_pks, "Boost had no effect on ordering"

    def test_boost_increases_score_of_matching_docs(self):
        """Documents matching the boosted term should score higher than without boost."""
        unboosted = self.backend.search("indexing", models=[MockModel])
        boosted = self.backend.search(
            "indexing", models=[MockModel], boost={"template": 10.0}
        )
        # Build pk→score maps
        unboosted_scores = {r.pk: r.score for r in unboosted["results"]}
        boosted_scores = {r.pk: r.score for r in boosted["results"]}
        # At least one doc that matches "template" should have a higher score
        any_increased = any(
            boosted_scores.get(pk, 0) > unboosted_scores.get(pk, 0)
            for pk in boosted_scores
        )
        assert any_increased

    # --- Facet correctness ---

    def test_facet_counts_sum_to_total(self):
        """Author facet counts should sum to total search hits."""
        results = self.backend.search("*", models=[MockModel], facets=["author"])
        total_hits = results["hits"]
        facet_values = results["facets"]["fields"]["author"]
        facet_sum = sum(count for _, count in facet_values)
        assert facet_sum == total_hits

    def test_facet_values_match_database(self):
        """Author facet values should match actual author counts in the database."""
        results = self.backend.search("*", models=[MockModel], facets=["author"])
        facet_dict = dict(results["facets"]["fields"]["author"])
        for author in ["daniel1", "daniel2", "daniel3"]:
            expected = MockModel.objects.filter(author=author).count()
            assert facet_dict[author] == expected, (
                f"Facet count for {author}: expected {expected}, got {facet_dict[author]}"
            )

    def test_facet_sorted_by_count_desc(self):
        """Facet values should be sorted by count descending."""
        results = self.backend.search("*", models=[MockModel], facets=["author"])
        facet_values = results["facets"]["fields"]["author"]
        counts = [count for _, count in facet_values]
        assert counts == sorted(counts, reverse=True)

    # --- Pagination correctness ---

    def test_paginated_results_reassemble_to_full(self):
        """Paginating with page_size=3 and reassembling gives the same set as full query."""
        full = self.backend.search("*", models=[MockModel])
        full_pks = {r.pk for r in full["results"]}

        reassembled_pks = set()
        offset = 0
        page_size = 3
        while offset < full["hits"]:
            page = self.backend.search(
                "*", models=[MockModel], start_offset=offset, end_offset=offset + page_size
            )
            for r in page["results"]:
                reassembled_pks.add(r.pk)
            offset += page_size

        assert reassembled_pks == full_pks

    def test_page_hits_stable_across_pages(self):
        """Total hits count should be the same on every page."""
        full_hits = self.backend.search("*", models=[MockModel])["hits"]
        for offset in range(0, full_hits, 5):
            page = self.backend.search(
                "*", models=[MockModel], start_offset=offset, end_offset=offset + 5
            )
            assert page["hits"] == full_hits

    # --- Multi-model field correctness ---

    def test_multi_model_mockmodel_has_author(self):
        """MockModel results in multi-model search have their author field."""
        results = self.backend.search("*")
        for r in results["results"]:
            if r.model == MockModel:
                db_obj = MockModel.objects.get(pk=r.pk)
                assert r.author == db_obj.author

    def test_multi_model_another_has_no_author_value(self):
        """AnotherMockModel results should have author=None (field doesn't exist on that index)."""
        results = self.backend.search("*")
        found = False
        for r in results["results"]:
            if r.model == AnotherMockModel:
                found = True
                assert r.author is None
        assert found, "No AnotherMockModel results found"

    def test_multi_model_score_model_has_text(self):
        """ScoreMockModel results should have their text field populated."""
        results = self.backend.search("*")
        found_score_model = False
        for r in results["results"]:
            if r.model == ScoreMockModel:
                found_score_model = True
                assert r.text is not None
        assert found_score_model, "No ScoreMockModel results found"

    # --- Negation correctness ---

    def test_negation_excludes_term(self):
        """'indexing -template' should not return docs matching 'template'."""
        positive = self.backend.search("template", models=[MockModel])
        negated = self.backend.search("indexing -template", models=[MockModel])
        positive_pks = {r.pk for r in positive["results"]}
        negated_pks = {r.pk for r in negated["results"]}
        # No result in negated should appear in positive (template-matching) set
        assert negated_pks & positive_pks == set()

    # --- Narrow correctness ---

    def test_narrow_returns_only_matching_values(self):
        """Narrowing to author:'daniel1' returns only daniel1 docs."""
        results = self.backend.search("*", narrow_queries=['author:"daniel1"'])
        for r in results["results"]:
            if r.model == MockModel:
                assert r.author == "daniel1"

    def test_narrow_hit_count_matches_database(self):
        """Narrow to author:'daniel1' should match the DB count."""
        results = self.backend.search(
            "*", models=[MockModel], narrow_queries=['author:"daniel1"']
        )
        expected = MockModel.objects.filter(author="daniel1").count()
        assert results["hits"] == expected
