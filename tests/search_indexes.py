from haystack import indexes

from tests.core.models import AnotherMockModel, MockModel, ScoreMockModel


class MockSearchIndex(indexes.SearchIndex, indexes.Indexable):
    text = indexes.CharField(document=True, use_template=True)
    author = indexes.CharField(model_attr="author")
    pub_date = indexes.DateTimeField(model_attr="pub_date")

    def get_model(self):
        return MockModel


class AnotherMockSearchIndex(indexes.SearchIndex, indexes.Indexable):
    text = indexes.CharField(document=True, model_attr="author")
    pub_date = indexes.DateTimeField(model_attr="pub_date")

    def get_model(self):
        return AnotherMockModel


class ScoreMockSearchIndex(indexes.SearchIndex, indexes.Indexable):
    text = indexes.CharField(document=True, model_attr="score")
    score = indexes.CharField(model_attr="score")

    def get_model(self):
        return ScoreMockModel
