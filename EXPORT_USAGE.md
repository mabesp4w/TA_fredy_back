<!-- @format -->

# Export PDF dan Excel - Panduan Penggunaan

## Library yang Digunakan

1. **reportlab==4.0.7** - Untuk export PDF
2. **openpyxl==3.1.2** - Untuk export Excel

Library ini sudah ditambahkan ke `requirements.txt`.

## Instalasi di Docker Container

Karena backend berjalan di Docker container, Anda perlu rebuild image untuk menginstall library baru:

```bash
# Rebuild Docker image
docker exec back-web-1 pip install reportlab==4.0.7 openpyxl==3.1.2
# restart
docker restart back-web-1
```

Atau jika menggunakan script rebuild:

```bash
./rebuild_docker.sh
```

## Cara Menggunakan

### 1. Export Dashboard Statistics

#### PDF

```
GET /api/dashboard/export-pdf/
```

#### Excel

```
GET /api/dashboard/export-excel/
```

### 2. Export All Birds Data (API Endpoint) - **RECOMMENDED**

Endpoint ini lengkap dengan semua field birds. Dapat diakses secara umum di `/api/birds/`.

#### PDF

```
GET /api/birds/export-pdf/
GET /api/birds/export-pdf/?family=<family_id>  # Filter by family
GET /api/birds/export-pdf/?search=<keyword>    # Search by name, scientific name, description, or habitat
GET /api/birds/export-pdf/?ordering=<field>    # Order by field (e.g., bird_nm, -created_at)
```

**Field yang di-export:**

- Bird Name
- Scientific Name
- Family
- Description
- Habitat

#### Excel

```
GET /api/birds/export-excel/
GET /api/birds/export-excel/?family=<family_id>  # Filter by family
GET /api/birds/export-excel/?search=<keyword>    # Search by name, scientific name, description, or habitat
GET /api/birds/export-excel/?ordering=<field>    # Order by field (e.g., bird_nm, -created_at)
```

**Field yang di-export:**

- Bird Name
- Scientific Name
- Family
- Description (full text)
- Habitat (full text)

## Contoh Implementasi di View Lain

Jika ingin menambahkan export di view lain, gunakan utility yang sudah tersedia:

```python
from api.utils.export_utils import pdf_exporter, excel_exporter

@action(detail=False, methods=['get'])
def export_pdf(self, request):
    try:
        # Ambil data dari database
        queryset = YourModel.objects.all()

        # Format data
        data = []
        for item in queryset:
            data.append({
                'Field 1': item.field1,
                'Field 2': item.field2,
                # ... tambahkan field lainnya
            })

        headers = ['Field 1', 'Field 2']
        filename = f"report_{timezone.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        return pdf_exporter.export_to_pdf(
            data=data,
            filename=filename,
            title="Your Report Title",
            headers=headers
        )
    except Exception as e:
        return Response(
            {'error': f'Failed to generate PDF: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
```

## Format Data

Utility export mendukung dua format data:

1. **List of Dictionaries** (Recommended)

```python
data = [
    {'Name': 'John', 'Age': 30},
    {'Name': 'Jane', 'Age': 25}
]
headers = ['Name', 'Age']
```

2. **List of Lists**

```python
data = [
    ['Name', 'Age'],  # Header
    ['John', 30],
    ['Jane', 25]
]
```

## Testing

Setelah rebuild container, test endpoint export:

```bash
# Test Dashboard PDF export
curl -X GET http://localhost:8103/api/dashboard/export-pdf/ -o dashboard.pdf

# Test Dashboard Excel export
curl -X GET http://localhost:8103/api/dashboard/export-excel/ -o dashboard.xlsx

# Test Birds PDF export (API endpoint - recommended)
curl -X GET http://localhost:8103/api/birds/export-pdf/ -o birds_export.pdf

# Test Birds Excel export (API endpoint - recommended)
curl -X GET http://localhost:8103/api/birds/export-excel/ -o birds_export.xlsx

# Test dengan filter family
curl -X GET "http://localhost:8103/api/birds/export-pdf/?family=<family_id>" -o birds_filtered.pdf

# Test dengan search
curl -X GET "http://localhost:8103/api/birds/export-excel/?search=kakatua" -o birds_search.xlsx
```

## Catatan

- File akan otomatis di-download dengan nama yang unik berdasarkan timestamp
- PDF menggunakan format A4
- Excel menggunakan format .xlsx (Excel 2007+)
- Kolom Excel akan otomatis di-adjust sesuai lebar konten
