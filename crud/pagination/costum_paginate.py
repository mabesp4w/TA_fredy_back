# paginate/costume_paginate.py
from rest_framework.pagination import PageNumberPagination
from rest_framework.response import Response
from collections import OrderedDict
from math import ceil


class LaravelStylePagination(PageNumberPagination):
    page_size = 10  # Jumlah item per halaman
    page_size_query_param = 'per_page'  # Menggunakan nama parameter seperti Laravel
    max_page_size = 100  # Batasan maksimum untuk page_size
    page_query_param = 'page'  # Parameter untuk nomor halaman

    def get_paginated_response(self, data):
        count = self.page.paginator.count
        page_size = self.get_page_size(self.request)
        current_page = self.page.number
        total_pages = ceil(count / page_size)

        # Menghitung from dan to seperti Laravel
        from_item = (current_page - 1) * page_size + 1 if count > 0 else 0
        to_item = min(current_page * page_size, count)

        # Menentukan path tanpa query params
        path = self.request.build_absolute_uri().split('?')[0]

        # Membuat struktur respons mirip Laravel
        return Response(OrderedDict([
            ('data', data),
            ('links', OrderedDict([
                ('first', f"{path}?page=1"),
                ('last', f"{path}?page={total_pages}"),
                ('prev', f"{path}?page={current_page - 1}" if current_page > 1 else None),
                ('next', f"{path}?page={current_page + 1}" if current_page < total_pages else None),
            ])),
                ('current_page', current_page),
                ('from', from_item),
                ('last_page', total_pages),
                ('path', path),
                ('per_page', page_size),
                ('to', to_item),
                ('total', count),
                ('links', self._get_page_links(current_page, total_pages, path)),
        ]))

    def _get_page_links(self, current_page, total_pages, path):
        """
        Menghasilkan array link pagination untuk ditampilkan seperti Laravel
        """
        links = []

        # Tentukan rentang halaman yang akan ditampilkan
        window = 4  # Jumlah halaman di sebelah kiri dan kanan
        window_start = max(1, current_page - window)
        window_end = min(total_pages, current_page + window)

        # Tambahkan "..." jika perlu
        if window_start > 1:
            links.append({
                'url': f"{path}?page=1",
                'label': '1',
                'active': False
            })
            if window_start > 2:
                links.append({
                    'url': None,
                    'label': '...',
                    'active': False
                })

        # Tambahkan halaman dalam jendela
        for i in range(window_start, window_end + 1):
            links.append({
                'url': f"{path}?page={i}",
                'label': str(i),
                'active': i == current_page
            })

        # Tambahkan "..." dan halaman terakhir jika perlu
        if window_end < total_pages:
            if window_end < total_pages - 1:
                links.append({
                    'url': None,
                    'label': '...',
                    'active': False
                })
            links.append({
                'url': f"{path}?page={total_pages}",
                'label': str(total_pages),
                'active': False
            })

        return links