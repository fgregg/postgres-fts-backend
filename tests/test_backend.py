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
from django.test import TestCase, override_settings
from haystack import connections
from haystack import indexes as haystack_indexes
from haystack.query import SQ, RelatedSearchQuerySet, SearchQuerySet
from haystack.utils.loading import UnifiedIndex

import postgres_fts_backend.models as models_module
from postgres_fts_backend import validate_all_schemas
from postgres_fts_backend.models import generate_index_models, get_index_model
from tests.core.models import AnotherMockModel, MockModel
from tests.mocks import MockSearchResult
from tests.search_indexes import AnotherMockSearchIndex, MockSearchIndex


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

    def test_search_rejects_unknown_field(self):
        """Searching field:value with a non-existent field should raise FieldError."""
        with pytest.raises(FieldError):
            self.backend.search("nonexistent_field:foo")

    def test_filter_rejects_unknown_field(self):
        """Filtering on a non-existent field should raise FieldError."""
        with pytest.raises(FieldError):
            list(SearchQuerySet().filter(nonexistent_field__exact="foo"))

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

    def test_search_across_models_not_supported(self):
        """Searching across multiple models should raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            self.backend.search("daniel3")

    def test_search_filtered_to_model(self):
        """Searching with a model filter should restrict results to that model."""
        results = self.backend.search("daniel3", models=[MockModel])
        for result in results["results"]:
            assert result.model == MockModel


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
        """_validate_all_schemas should raise RuntimeError for a missing table."""
        with patch(
            "postgres_fts_backend._table_name", return_value="nonexistent_table_xyz"
        ):
            with pytest.raises(RuntimeError) as ctx:
                validate_all_schemas()
            assert "does not exist" in str(ctx.value)


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

        backend_setup(self, [ExtendedMockSearchIndex(), self.another_index])
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

        backend_setup(self, [ExtendedMockSearchIndex(), self.another_index])
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
            backend_setup(self, [self.index, self.another_index])
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
        backend_setup(self, [self.index, self.another_index])

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
        backend_setup(self, [ngram_index, self.another_index])
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
        backend_setup(self, [ngram_index, self.another_index])
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
        backend_setup(self, [ngram_index, self.another_index])
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
