import json
import os
from django.core.management.base import BaseCommand
from django.db import transaction
from crud.models import Family, Bird  # Ganti 'myapp' dengan nama app Anda


class Command(BaseCommand):
    help = 'Load bird data from JSON file'

    def add_arguments(self, parser):
        parser.add_argument(
            '--file',
            type=str,
            help='Path to JSON file containing bird data',
            default='static/jenis_burung.json'
        )
        parser.add_argument(
            '--clear',
            action='store_true',
            help='Clear existing data before loading',
        )
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Show verbose output',
        )

    def handle(self, *args, **options):
        file_path = options['file']
        clear_data = options['clear']

        if not os.path.exists(file_path):
            self.stdout.write(
                self.style.ERROR(f'File {file_path} not found!')
            )
            return

        # Clear existing data if requested
        if clear_data:
            self.stdout.write('Clearing existing data...')
            Bird.objects.all().delete()
            Family.objects.all().delete()
            self.stdout.write(
                self.style.SUCCESS('Existing data cleared!')
            )

        # Load JSON data
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                bird_data = json.load(file)
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error reading JSON file: {e}')
            )
            return

        # Process data
        self.load_bird_data(bird_data)

    @transaction.atomic
    def load_bird_data(self, bird_data):
        """Load bird data with atomic transaction"""

        family_cache = {}
        created_birds = 0
        created_families = 0

        for bird_info in bird_data:
            try:
                # Get or create family
                family_name = bird_info.get('family')
                if family_name not in family_cache:
                    family, created = Family.objects.get_or_create(
                        family_nm=family_name,
                        defaults={
                            'description': f'Family {family_name} - Auto generated'
                        }
                    )
                    family_cache[family_name] = family
                    if created:
                        created_families += 1
                        self.stdout.write(f'Created family: {family_name}')

                family = family_cache[family_name]

                # Create bird
                bird, created = Bird.objects.get_or_create(
                    scientific_nm=bird_info.get('scientific_nm'),
                    defaults={
                        'bird_nm': bird_info.get('bird_nm'),
                        'family': family,
                        'description': bird_info.get('description', ''),
                        'habitat': bird_info.get('habitat', ''),
                    }
                )

                if created:
                    created_birds += 1
                    self.stdout.write(f'Created bird: {bird.bird_nm}')
                else:
                    self.stdout.write(f'Bird already exists: {bird.bird_nm}')

            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(
                        f'Error processing bird {bird_info.get("bird_nm", "Unknown")}: {e}'
                    )
                )
                continue

        # Summary
        self.stdout.write(
            self.style.SUCCESS(
                f'\nData loading completed!'
                f'\nFamilies created: {created_families}'
                f'\nBirds created: {created_birds}'
            )
        )