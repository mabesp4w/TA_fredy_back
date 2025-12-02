# api/views.py
from django.http import JsonResponse
from django.conf import settings
import json
import os

def jenis_burung_api(request):
    """API endpoint untuk melayani data jenis burung"""
    try:
        file_path = os.path.join(settings.BASE_DIR, 'static', 'jenis_burung.json')
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return JsonResponse(data, safe=False)
    except FileNotFoundError:
        return JsonResponse({'error': 'File tidak ditemukan'}, status=404)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Format JSON tidak valid'}, status=400)