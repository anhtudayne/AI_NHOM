"""
Base pathfinder class for routing algorithms.
"""

from typing import List, Tuple, Dict, Set, Optional
import numpy as np
from .base_search import BaseSearch, SearchState

class BasePathFinder(BaseSearch):
    """Lớp cơ sở cho thuật toán tìm đường với ràng buộc nhiên liệu"""
    
    # Các hằng số
    FUEL_PER_MOVE = 0.5      # Nhiên liệu tiêu thụ mỗi bước (L)
    GAS_STATION_COST = 30.0  # Chi phí đổ xăng (đ)
    TOLL_COST = 5.0          # Chi phí qua trạm thu phí (đ)
    TOLL_PENALTY = 1000.0    # Phạt cho việc đi qua trạm thu phí 