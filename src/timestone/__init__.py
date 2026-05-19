"""TimeStone AI - evidence-based business transformation forecasting.

Public stable API:
    from timestone import assess_company, AssessOptions
"""
__version__ = "0.2.0"

from .application import assess_company, AssessOptions

__all__ = ["assess_company", "AssessOptions"]
