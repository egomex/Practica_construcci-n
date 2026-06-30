from django.db import models


class RegistroConteo(models.Model):
    """Almacena cada registro de conteo de personas detectadas."""
    cantidad_personas = models.IntegerField(default=0)
    timestamp = models.DateTimeField(auto_now_add=True)
    captura = models.ImageField(upload_to='capturas/', null=True, blank=True)
    fuente = models.CharField(
        max_length=50,
        choices=[('camara', 'Cámara en vivo'), ('imagen', 'Imagen subida')],
        default='camara'
    )

    class Meta:
        ordering = ['-timestamp']
        verbose_name = 'Registro de Conteo'
        verbose_name_plural = 'Registros de Conteo'

    def __str__(self):
        return f"{self.timestamp.strftime('%d/%m/%Y %H:%M')} — {self.cantidad_personas} persona(s)"
