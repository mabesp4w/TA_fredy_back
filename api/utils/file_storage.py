# utils/file_storage.py
import os
import uuid
import logging
from django.conf import settings
from django.core.files.storage import FileSystemStorage

logger = logging.getLogger(__name__)


class CustomAudioStorage:
    """
    Custom file storage untuk mengelola audio files
    """

    def __init__(self):
        self.temp_dir = getattr(settings, 'TEMP_AUDIO_DIR',
                                os.path.join(settings.MEDIA_ROOT, 'temp_audio'))
        self.uploads_dir = getattr(settings, 'UPLOADS_DIR',
                                   os.path.join(settings.MEDIA_ROOT, 'uploads'))

        # Pastikan direktori ada
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.uploads_dir, exist_ok=True)

    def save_temp_audio(self, audio_file):
        """
        Simpan audio file ke temporary directory
        """
        try:
            # Generate unique filename
            unique_id = uuid.uuid4().hex
            original_name = getattr(audio_file, 'name', 'audio.wav')
            file_extension = os.path.splitext(original_name)[1] or '.wav'
            filename = f"temp_audio_{unique_id}{file_extension}"

            # Full path
            file_path = os.path.join(self.temp_dir, filename)

            logger.info(f"Attempting to save audio to: {file_path}")
            logger.info(f"Temp directory exists: {os.path.exists(self.temp_dir)}")
            logger.info(f"Temp directory writable: {os.access(self.temp_dir, os.W_OK)}")

            # Reset file pointer
            if hasattr(audio_file, 'seek'):
                audio_file.seek(0)

            # Write file
            with open(file_path, 'wb') as destination:
                if hasattr(audio_file, 'chunks'):
                    # Django UploadedFile
                    for chunk in audio_file.chunks():
                        destination.write(chunk)
                else:
                    # Regular file object
                    destination.write(audio_file.read())

                # Force write to disk
                destination.flush()
                os.fsync(destination.fileno())

            # Verify file
            if not os.path.exists(file_path):
                logger.error(f"File not created: {file_path}")
                return None

            file_size = os.path.getsize(file_path)
            if file_size == 0:
                logger.error(f"File is empty: {file_path}")
                self.cleanup_file(file_path)
                return None

            logger.info(f"Audio saved successfully: {file_path} ({file_size} bytes)")
            return file_path

        except PermissionError as e:
            logger.error(f"Permission error: {e}")
            return None
        except OSError as e:
            logger.error(f"OS error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None

    def cleanup_file(self, file_path):
        """
        Hapus file temporary
        """
        try:
            if file_path and os.path.exists(file_path):
                os.unlink(file_path)
                logger.info(f"Cleaned up file: {file_path}")
                return True
        except Exception as e:
            logger.warning(f"Failed to cleanup file {file_path}: {e}")
        return False

    def cleanup_old_temps(self, max_age_hours=24):
        """
        Hapus file temporary yang sudah lama
        """
        try:
            import time
            now = time.time()
            count = 0

            for filename in os.listdir(self.temp_dir):
                if filename.startswith('temp_audio_'):
                    file_path = os.path.join(self.temp_dir, filename)
                    file_age_hours = (now - os.path.getctime(file_path)) / 3600

                    if file_age_hours > max_age_hours:
                        if self.cleanup_file(file_path):
                            count += 1

            logger.info(f"Cleaned up {count} old temporary files")

        except Exception as e:
            logger.error(f"Error cleaning up old temp files: {e}")


# Global instance
audio_storage = CustomAudioStorage()