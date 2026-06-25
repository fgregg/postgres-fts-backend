from haystack import indexes

from tests.core.models import (
    AllFieldsModel,
    AnotherMockModel,
    MockModel,
    ScoreMockModel,
)


class MockSearchIndex(indexes.SearchIndex, indexes.Indexable):
    text = indexes.CharField(document=True, use_template=True)
    author = indexes.CharField(model_attr="author")
    pub_date = indexes.DateTimeField(model_attr="pub_date")
    categories = indexes.MultiValueField()
    # Nullable faceted field — null for some records, so multi-model facet
    # merges can include a None value.
    region = indexes.CharField(faceted=True, null=True)

    def get_model(self):
        return MockModel

    def prepare_categories(self, obj):
        # Deterministic categories derived from author name.
        base = ["general"]
        if obj.author.endswith("1"):
            base.append("primary")
        if obj.author.endswith("2"):
            base.append("secondary")
        return base

    def prepare_region(self, obj):
        return "west" if obj.author.endswith("1") else None


class AnotherMockSearchIndex(indexes.SearchIndex, indexes.Indexable):
    text = indexes.CharField(document=True, model_attr="author")
    pub_date = indexes.DateTimeField(model_attr="pub_date")
    region = indexes.CharField(faceted=True, null=True)

    def get_model(self):
        return AnotherMockModel

    def prepare_region(self, obj):
        return None


class ScoreMockSearchIndex(indexes.SearchIndex, indexes.Indexable):
    text = indexes.CharField(document=True, model_attr="score")
    score = indexes.CharField(model_attr="score")

    def get_model(self):
        return ScoreMockModel


class AllFieldsSearchIndex(indexes.SearchIndex, indexes.Indexable):
    text = indexes.CharField(document=True, model_attr="name")
    name_ngram = indexes.NgramField(model_attr="name")
    name_edge = indexes.EdgeNgramField(model_attr="name")
    is_active = indexes.BooleanField(model_attr="is_active")
    count = indexes.IntegerField(model_attr="count")
    rating = indexes.FloatField(model_attr="rating")
    price = indexes.DecimalField(model_attr="price")
    created_date = indexes.DateField(model_attr="created_date")
    created_at = indexes.DateTimeField(model_attr="created_at")

    def get_model(self):
        return AllFieldsModel
