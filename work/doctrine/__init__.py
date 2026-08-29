# -*- coding: utf-8 -*-
"""Библиотека доктрины кампании E-CUP 2026.

Управление неопределённостью при оптимизации под скрытую генеральную совокупность.
Формулы и выводы: work/reports/zhenya_K3_monograph.md
Тесты на исторических замерах: work/doctrine/tests/test_doctrine.py
"""
from . import transfer, dose, probes, slip, field   # noqa: F401
__all__ = ["transfer", "dose", "probes", "slip", "field"]
