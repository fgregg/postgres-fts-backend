import pytest
from django.core.management import call_command
from haystack import connections
from haystack.utils.loading import UnifiedIndex

from tests.search_indexes import (
    AllFieldsSearchIndex,
    AnotherMockSearchIndex,
    MockSearchIndex,
    ScoreMockSearchIndex,
)


@pytest.fixture(scope="session")
def _build_search_schema(django_db_setup, django_db_blocker):
    """Generate and apply search index migrations before tests run."""
    # Register test indexes so build_postgres_schema can find them
    ui = UnifiedIndex()
    ui.build(
        indexes=[
            MockSearchIndex(),
            AnotherMockSearchIndex(),
            ScoreMockSearchIndex(),
            AllFieldsSearchIndex(),
        ]
    )
    connections["default"]._index = ui

    with django_db_blocker.unblock():
        call_command("build_postgres_schema", verbosity=0)
        call_command("migrate", "postgres_fts_backend", verbosity=0)


@pytest.fixture(autouse=True)
def _search_schema(_build_search_schema):
    """Ensure search schema is available for all tests."""
