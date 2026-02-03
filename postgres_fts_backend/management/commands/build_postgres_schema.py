import os
import sys

from django.conf import settings
from django.contrib.postgres.operations import TrigramExtension
from django.core.management.base import BaseCommand
from django.db.migrations.autodetector import MigrationAutodetector
from django.db.migrations.loader import MigrationLoader
from django.db.migrations.state import ModelState, ProjectState
from django.db.migrations.writer import MigrationWriter

from postgres_fts_backend.models import generate_index_models

APP_LABEL = "postgres_fts_backend"


class Command(BaseCommand):
    help = (
        "Generate Django migrations for haystack search index tables. "
        "Run this after changing SearchIndex definitions, then run "
        "'manage.py migrate postgres_fts_backend'."
    )

    def handle(self, *args, **options):
        index_models = generate_index_models()

        if not index_models:
            self.stdout.write("No search indexes found.")
            return

        # Build the target state from the dynamic models
        new_state = ProjectState()
        for model_cls in index_models.values():
            model_state = ModelState.from_model(model_cls)
            new_state.add_model(model_state)

        # Load existing migrations to get the current state
        loader = MigrationLoader(None, ignore_no_migrations=True)
        old_state = loader.project_state()

        # Detect changes
        autodetector = MigrationAutodetector(old_state, new_state)
        changes = autodetector.changes(graph=loader.graph)

        if not changes.get(APP_LABEL):
            self.stdout.write("No changes detected.")
            return

        # Determine output directory
        migrations_module = getattr(settings, "MIGRATION_MODULES", {}).get(APP_LABEL)
        if migrations_module:
            migrations_dir = os.path.join(*migrations_module.split("."))
            # Make it relative to the project base if not absolute
            if not os.path.isabs(migrations_dir):
                # Find the root by looking at the first component on sys.path
                # that contains the module
                for path in sys.path:
                    candidate = os.path.join(path, migrations_dir)
                    if os.path.isdir(os.path.dirname(candidate)) or path == "":
                        migrations_dir = candidate
                        break
        else:
            # Default to the package's own migrations dir
            migrations_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "migrations",
            )

        os.makedirs(migrations_dir, exist_ok=True)

        # Ensure __init__.py exists
        init_path = os.path.join(migrations_dir, "__init__.py")
        if not os.path.exists(init_path):
            with open(init_path, "w") as f:
                f.write("")

        for migration in changes[APP_LABEL]:
            if getattr(migration, "initial", False):
                migration.operations.insert(0, TrigramExtension())
            writer = MigrationWriter(migration)
            migration_path = os.path.join(migrations_dir, f"{migration.name}.py")
            with open(migration_path, "w") as f:
                f.write(writer.as_string())
            self.stdout.write(f"Created {migration_path}")
