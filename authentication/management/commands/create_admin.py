from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model

User = get_user_model()


class Command(BaseCommand):
    help = 'Membuat user admin'

    def add_arguments(self, parser):
        parser.add_argument('--username', type=str, default='admin', help='Username untuk admin')
        parser.add_argument('--email', type=str, default='admin@mail.com', help='Email untuk admin')
        parser.add_argument('--password', type=str, default='12345678', help='Password untuk admin')

    def handle(self, *args, **options):
        username = options['username']
        email = options['email']
        password = options['password']

        if User.objects.filter(username=username).exists():
            self.stdout.write(self.style.WARNING(f'User dengan username {username} sudah ada'))
            return

        admin = User.objects.create_user(
            username=username,
            email=email,
            password=password,
            role='admin',
            is_staff=True,
            is_superuser=True
        )

        self.stdout.write(self.style.SUCCESS(f'Berhasil membuat admin: {admin.username}'))