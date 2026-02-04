import datetime

from django.db import models


class MockTag(models.Model):
    name = models.CharField(max_length=32)

    def __str__(self):
        return self.name


class MockModel(models.Model):
    author = models.CharField(max_length=255)
    foo = models.TextField(blank=True)
    pub_date = models.DateTimeField(default=datetime.datetime.now)
    tag = models.ForeignKey(MockTag, models.CASCADE)

    def __str__(self):
        return self.author

    def hello(self):
        return "World!"


class ScoreMockModel(models.Model):
    score = models.CharField(max_length=10)

    def __str__(self):
        return self.score


class AnotherMockModel(models.Model):
    author = models.CharField(max_length=255)
    pub_date = models.DateTimeField(default=datetime.datetime.now)

    def __str__(self):
        return self.author


class AllFieldsModel(models.Model):
    name = models.CharField(max_length=255)
    is_active = models.BooleanField(default=True)
    count = models.IntegerField(default=0)
    rating = models.FloatField(default=0.0)
    price = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    created_date = models.DateField()
    created_at = models.DateTimeField()

    def __str__(self):
        return self.name
