"""
Định nghĩa các hằng số và trọng số cho hệ thống định tuyến
"""

from enum import IntEnum
from dataclasses import dataclass

class CellType(IntEnum):
    """Định nghĩa các loại ô trên bản đồ"""
    ROAD = 0        # Đường thông thường
    TOLL = 1        # Trạm thu phí
    GAS = 2         # Trạm xăng
    OBSTACLE = 3    # Vật cản

@dataclass
class MovementCosts:
    """Chi phí di chuyển cơ bản"""
    FUEL_PER_MOVE: float = 0.4      # Nhiên liệu tiêu thụ mỗi bước (L)
    MAX_FUEL: float = 3.0           # Dung tích bình xăng tối đa (L)
    LOW_FUEL_THRESHOLD: float = 1.0  # Ngưỡng xăng thấp (L)

@dataclass
class StationCosts:
    """Chi phí tại các trạm dịch vụ"""
    # Chi phí trạm xăng
    BASE_GAS_COST: float = 30.0     # Chi phí cơ bản khi đổ xăng
    GAS_DISCOUNT_FACTOR: float = 0.8 # Giảm giá khi nhiên liệu thấp
    
    # Chi phí trạm thu phí
    BASE_TOLL_COST: float = 50.0    # Chi phí cơ bản qua trạm
    TOLL_PENALTY: float = 100.0     # Phạt ban đầu khi qua trạm
    MAX_TOLL_DISCOUNT: float = 0.5  # Giảm giá tối đa cho trạm thu phí

@dataclass
class PathfindingWeights:
    """Trọng số cho thuật toán tìm đường"""
    # Trọng số cơ bản cho mỗi loại ô
    ROAD_WEIGHT: float = 1.0
    TOLL_BASE_WEIGHT: float = 8.0   # Trọng số cơ bản cho trạm thu phí
    GAS_BASE_WEIGHT: float = 3.0    # Trọng số cơ bản cho trạm xăng
    
    # Hệ số điều chỉnh theo điều kiện
    FUEL_URGENCY_FACTOR: float = 2.0  # Tăng ưu tiên trạm xăng khi nhiên liệu thấp
    TOLL_AVOIDANCE_FACTOR: float = 1.5 # Tăng né tránh trạm thu phí
    
    # Giới hạn chi phí
    MAX_TOTAL_COST: float = 5000.0  # Chi phí tối đa cho phép

def calculate_gas_station_weight(current_fuel: float, max_fuel: float) -> float:
    """Tính trọng số cho trạm xăng dựa trên lượng nhiên liệu hiện tại
    
    Args:
        current_fuel: Lượng nhiên liệu hiện tại
        max_fuel: Dung tích bình xăng tối đa
    
    Returns:
        float: Trọng số cho trạm xăng
    """
    fuel_ratio = current_fuel / max_fuel
    if fuel_ratio < 0.3:  # Nhiên liệu rất thấp
        return PathfindingWeights.GAS_BASE_WEIGHT * 0.5  # Giảm chi phí để ưu tiên ghé
    elif fuel_ratio < 0.5:  # Nhiên liệu thấp
        return PathfindingWeights.GAS_BASE_WEIGHT * 0.8
    else:  # Nhiên liệu đủ
        return PathfindingWeights.GAS_BASE_WEIGHT * 1.2  # Tăng chi phí vì không cần gấp

def calculate_toll_station_weight(visited_tolls: int) -> float:
    """Tính trọng số cho trạm thu phí dựa trên số trạm đã đi qua
    
    Args:
        visited_tolls: Số trạm thu phí đã đi qua
    
    Returns:
        float: Trọng số cho trạm thu phí
    """
    # Giảm trọng số nếu đã đi qua nhiều trạm (tối đa giảm 50%)
    discount = min(StationCosts.MAX_TOLL_DISCOUNT, visited_tolls * 0.1)
    return PathfindingWeights.TOLL_BASE_WEIGHT * (1.0 - discount) 