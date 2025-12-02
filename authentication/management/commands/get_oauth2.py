from django.core.management.base import BaseCommand
from oauth2_provider.models import Application
import secrets
import string
import os
from django.contrib.auth import get_user_model
from dotenv import load_dotenv

User = get_user_model()


class Command(BaseCommand):
    help = 'Inisialisasi OAUTH2_CLIENT_ID dan OAUTH2_CLIENT_SECRET dan menyimpannya ke .env'

    def add_arguments(self, parser):
        parser.add_argument('--name', type=str, default='Thobias App', help='Nama aplikasi OAuth2')
        parser.add_argument('--admin_username', type=str, default='admin', help='Username admin')
        parser.add_argument('--force', action='store_true', help='Paksa pembuatan baru meskipun sudah ada')

    def generate_secret(self, length=50):
        """Generate random string untuk client secret"""
        alphabet = string.ascii_letters + string.digits
        return ''.join(secrets.choice(alphabet) for _ in range(length))

    def handle(self, *args, **options):
        name = options['name']
        admin_username = options['admin_username']
        force = options['force']

        # Pastikan admin sudah terdaftar
        try:
            admin_user = User.objects.get(username=admin_username)
        except User.DoesNotExist:
            self.stdout.write(self.style.ERROR(f'User admin dengan username {admin_username} tidak ditemukan'))
            self.stdout.write(self.style.WARNING('Jalankan perintah create_admin terlebih dahulu'))
            return

        # Cek apakah aplikasi oauth2 sudah ada
        existing_app = Application.objects.filter(name=name).first()

        if existing_app and not force:
            self.stdout.write(self.style.SUCCESS(f'Aplikasi OAuth2 sudah ada dengan nama {name}'))
            self.stdout.write(f'OAUTH2_CLIENT_ID: {existing_app.client_id}')
            self.stdout.write(f'OAUTH2_CLIENT_SECRET: {existing_app.client_secret}')

            # Update file .env
            self.update_env_file(existing_app.client_id, existing_app.client_secret)
            return

        # Generate client ID dan secret baru
        client_id = secrets.token_urlsafe(32)
        client_secret = self.generate_secret(50)

        # Buat atau update aplikasi OAuth2
        if existing_app:
            existing_app.client_id = client_id
            existing_app.client_secret = client_secret
            existing_app.save()
            app = existing_app
            self.stdout.write(self.style.SUCCESS(f'Aplikasi OAuth2 berhasil diupdate dengan ID dan Secret baru'))
        else:
            # Buat aplikasi OAuth2 baru
            app = Application.objects.create(
                name=name,
                user=admin_user,
                client_id=client_id,
                client_secret=client_secret,
                client_type='confidential',
                authorization_grant_type='password',
                skip_authorization=True,
                redirect_uris=''
            )
            self.stdout.write(self.style.SUCCESS(f'Aplikasi OAuth2 baru berhasil dibuat'))

        self.stdout.write(f'OAUTH2_CLIENT_ID: {app.client_id}')
        self.stdout.write(f'OAUTH2_CLIENT_SECRET: {app.client_secret}')

        # Update file .env
        self.update_env_file(app.client_id, app.client_secret)

    def update_env_file(self, client_id, client_secret):
        """Update file .env dengan credential OAuth2 baru"""
        env_path = os.path.join(os.getcwd(), '.env')

        # Baca file .env jika ada
        env_content = ""
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                env_content = f.read()

        # Fungsi untuk mengupdate atau menambahkan variabel ke env_content
        def update_env_variable(content, var_name, var_value):
            if f"{var_name}=" in content:
                # Update variabel yang sudah ada
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if line.startswith(f"{var_name}="):
                        lines[i] = f"{var_name}={var_value}"
                        break
                return '\n'.join(lines)
            else:
                # Tambahkan variabel baru
                return f"{content}\n{var_name}={var_value}"

        # Update OAUTH2_CLIENT_ID dan OAUTH2_CLIENT_SECRET
        env_content = update_env_variable(env_content, "OAUTH2_CLIENT_ID", client_id)
        env_content = update_env_variable(env_content, "OAUTH2_CLIENT_SECRET", client_secret)

        # Tulis kembali file .env
        with open(env_path, 'w') as f:
            f.write(env_content)

        self.stdout.write(self.style.SUCCESS(f'File .env berhasil diupdate dengan credential OAuth2 baru'))