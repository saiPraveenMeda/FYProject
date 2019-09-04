# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models
import os

# Create your models here.

class File(models.Model):
  file = models.FileField(blank=False, null=False)

  def __str__(self):
        return os.path.basename(self.file.name)