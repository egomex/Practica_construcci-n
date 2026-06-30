from django.contrib import admin
from .models import RegistroConteo


@admin.register(RegistroConteo)
class RegistroConteoAdmin(admin.ModelAdmin):
    list_display = ('timestamp', 'cantidad_personas', 'fuente')
    list_filter = ('fuente', 'timestamp')
    ordering = ('-timestamp',)
    readonly_fields = ('timestamp',)
