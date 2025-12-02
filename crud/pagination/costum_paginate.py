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

        # Build query parameters (exclude page parameter)
        query_params = self._get_query_params()

        # Membuat struktur respons mirip Laravel
        return Response(OrderedDict([
            ('data', data),
            ('meta', OrderedDict([
                ('current_page', current_page),
                ('from', from_item),
                ('last_page', total_pages),
                ('path', path),
                ('per_page', page_size),
                ('to', to_item),
                ('total', count),
            ])),
            ('links', OrderedDict([
                ('first', self._build_link(1, query_params)),
                ('last', self._build_link(total_pages, query_params)),
                ('prev', self._build_link(current_page - 1, query_params) if current_page > 1 else None),
                ('next', self._build_link(current_page + 1, query_params) if current_page < total_pages else None),
            ])),
            ('pagination_links', self._get_page_links(current_page, total_pages, query_params)),
        ]))

    def _get_query_params(self):
        """
        Mengambil semua query parameters kecuali 'page'
        """
        query_params = self.request.GET.copy()
        if 'page' in query_params:
            del query_params['page']
        return query_params

    def _build_link(self, page_number, query_params):
        """
        Membangun URL dengan parameter query
        """
        if page_number <= 0:
            return None

        path = self.request.build_absolute_uri().split('?')[0]
        params = query_params.copy()
        params['page'] = page_number

        if params:
            return f"{path}?{params.urlencode()}"
        else:
            return f"{path}?page={page_number}"

    def _get_page_links(self, current_page, total_pages, query_params):
        """
        Menghasilkan array link pagination untuk ditampilkan seperti Laravel
        """
        if total_pages <= 1:
            return []

        links = []
        path = self.request.build_absolute_uri().split('?')[0]

        # Previous button
        if current_page > 1:
            links.append({
                'url': self._build_link(current_page - 1, query_params),
                'label': '&laquo; Previous',
                'active': False
            })
        else:
            links.append({
                'url': None,
                'label': '&laquo; Previous',
                'active': False
            })

        # Tentukan rentang halaman yang akan ditampilkan
        window = 2  # Jumlah halaman di sebelah kiri dan kanan
        window_start = max(1, current_page - window)
        window_end = min(total_pages, current_page + window)

        # Tambahkan halaman pertama jika tidak dalam window
        if window_start > 1:
            links.append({
                'url': self._build_link(1, query_params),
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
                'url': self._build_link(i, query_params),
                'label': str(i),
                'active': i == current_page
            })

        # Tambahkan halaman terakhir jika tidak dalam window
        if window_end < total_pages:
            if window_end < total_pages - 1:
                links.append({
                    'url': None,
                    'label': '...',
                    'active': False
                })
            links.append({
                'url': self._build_link(total_pages, query_params),
                'label': str(total_pages),
                'active': False
            })

        # Next button
        if current_page < total_pages:
            links.append({
                'url': self._build_link(current_page + 1, query_params),
                'label': 'Next &raquo;',
                'active': False
            })
        else:
            links.append({
                'url': None,
                'label': 'Next &raquo;',
                'active': False
            })

        return links