"""Idempotent model operations for generated search-index migrations.

The search-index tables are derived state — their schema is regenerated from the
``SearchIndex`` definitions, and they can legitimately exist (or not) regardless
of what the migration history records. A plain ``CreateModel`` / ``DeleteModel``
assumes the migration state and the physical database agree, so it raises
"relation already exists" / "table does not exist" when they have drifted apart
(e.g. a table created out-of-band, or a regenerated chain whose recorded state
forgot a table that is still physically present).

These operations update the Django migration **state** unconditionally but make
the **database** step conditional on the table's actual presence, so generated
migrations apply cleanly on a fresh install, on production, and on any state the
two have drifted into.
"""

from django.db.migrations.operations.models import CreateModel, DeleteModel


def _table_exists(schema_editor, model):
    return model._meta.db_table in schema_editor.connection.introspection.table_names()


class CreateModelIfNotExists(CreateModel):
    """``CreateModel`` whose ``CREATE TABLE`` runs only if the table is absent."""

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        model = to_state.apps.get_model(app_label, self.name)
        if self.allow_migrate_model(schema_editor.connection.alias, model):
            if not _table_exists(schema_editor, model):
                super().database_forwards(
                    app_label, schema_editor, from_state, to_state
                )

    def database_backwards(self, app_label, schema_editor, from_state, to_state):
        model = from_state.apps.get_model(app_label, self.name)
        if self.allow_migrate_model(schema_editor.connection.alias, model):
            if _table_exists(schema_editor, model):
                super().database_backwards(
                    app_label, schema_editor, from_state, to_state
                )


class DeleteModelIfExists(DeleteModel):
    """``DeleteModel`` whose ``DROP TABLE`` runs only if the table is present."""

    def database_forwards(self, app_label, schema_editor, from_state, to_state):
        model = from_state.apps.get_model(app_label, self.name)
        if self.allow_migrate_model(schema_editor.connection.alias, model):
            if _table_exists(schema_editor, model):
                super().database_forwards(
                    app_label, schema_editor, from_state, to_state
                )

    def database_backwards(self, app_label, schema_editor, from_state, to_state):
        model = to_state.apps.get_model(app_label, self.name)
        if self.allow_migrate_model(schema_editor.connection.alias, model):
            if not _table_exists(schema_editor, model):
                super().database_backwards(
                    app_label, schema_editor, from_state, to_state
                )
