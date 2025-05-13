"""
Map display module for visualizing the routing environment.
Implements functions for drawing maps, routes, and animations using Streamlit.
"""

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
import io
import math
from core.algorithms.base_search import OBSTACLE_CELL, ROAD_CELL, TOLL_CELL, GAS_STATION_CELL

# Đường dẫn tới các icon
ICONS_PATH = os.path.join(os.path.dirname(__file__), 'units')

# Hằng số xác định loại ô
OBSTACLE_VALUE = OBSTACLE_CELL  # Đồng bộ với base_search.py
ROAD_VALUE = ROAD_CELL
TOLL_VALUE = TOLL_CELL
GAS_STATION_VALUE = GAS_STATION_CELL

def create_basic_icon(emoji, size=(64, 64), bg_color=None):
    """
    Tạo icon đơn giản với emoji
    
    Parameters:
    - emoji: Emoji text hiển thị 
    - size: Kích thước icon
    - bg_color: Màu nền
    
    Returns:
    - Đối tượng hình ảnh PIL
    """
    try:
        # Tạo hình nền
        img = Image.new("RGBA", size, bg_color if bg_color else (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Cố gắng load font emoji
        try:
            # Thử tải font có hỗ trợ emoji với kích thước lớn
            font_size = int(size[0] * 0.6)  # 60% kích thước icon
            try:
                # Thử các font phổ biến hỗ trợ emoji
                font_options = ["segoe ui emoji", "apple color emoji", "noto color emoji", "arial", "sans-serif"]
                font = None
                
                for font_name in font_options:
                    try:
                        font = ImageFont.truetype(font_name, font_size)
                        break
                    except:
                        continue
                
                if font is None:
                    font = ImageFont.load_default()
            except:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        # Tính toán vị trí để đặt emoji giữa hình
        try:
            # Phương thức mới trong Pillow mới hơn
            left, top, right, bottom = draw.textbbox((0, 0), emoji, font=font)
            text_width = right - left
            text_height = bottom - top
        except AttributeError:
            # Fallback cho Pillow cũ hơn
            try:
                text_width, text_height = draw.textsize(emoji, font=font)
            except:
                text_width, text_height = font_size, font_size
                
        position = ((size[0] - text_width) // 2, (size[1] - text_height) // 2)
        
        # Vẽ emoji
        draw.text(position, emoji, fill="white", font=font)
        
        return img
    except Exception as e:
        # Tạo một icon đơn giản nếu có lỗi
        img = Image.new("RGBA", size, bg_color if bg_color else (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Vẽ một hình vuông với chữ cái đầu tiên
        if emoji and len(emoji) > 0:
            letter = emoji[0] if isinstance(emoji, str) else "?"
            # Vẽ chữ cái ở giữa
            draw.text((size[0]//3, size[1]//3), letter, fill="white")
        
        return img

def load_modern_icons():
    """Tạo các icon hiện đại cho bản đồ"""
    icons = {}
    
    # Màu sắc theo phong cách Material Design
    colors = {
        'truck': (33, 150, 243, 255),    # Xanh dương - #2196F3
        'gas': (76, 175, 80, 255),       # Xanh lá - #4CAF50
        'toll': (244, 67, 54, 255),      # Đỏ - #F44336
        'brick': (121, 85, 72, 255),     # Nâu - #795548
        'road': (66, 66, 66, 255)        # Xám đậm - #424242
    }
    
    # Emoji text đơn giản 
    emojis = {
        'truck': '🚚',
        'gas': '⛽',
        'toll': '🚧',
        'brick': '🧱',
        'road': '🛣️'
    }
    
    # Tạo các icon
    for name, color in colors.items():
        try:
            icons[name] = create_basic_icon(
                emojis[name], 
                bg_color=color
            )
        except Exception as e:
            # Tạo icon dự phòng nếu gặp lỗi
            img = Image.new("RGBA", (64, 64), color)
            draw = ImageDraw.Draw(img)
            draw.rectangle([5, 5, 59, 59], fill=color, outline=(255, 255, 255, 128))
            icons[name] = img
    
    return icons

# Biến global để lưu trữ các icon đã tải
ICONS = None

def get_icons():
    """Trả về các icon đã tải hoặc tải mới nếu chưa có"""
    global ICONS
    if ICONS is None:
        try:
            ICONS = load_modern_icons()
        except Exception:
            # Tạo icons đơn giản nếu có lỗi
            ICONS = {
                'truck': Image.new("RGBA", (64, 64), (33, 150, 243, 255)),
                'gas': Image.new("RGBA", (64, 64), (76, 175, 80, 255)),
                'toll': Image.new("RGBA", (64, 64), (244, 67, 54, 255)),
                'brick': Image.new("RGBA", (64, 64), (121, 85, 72, 255)),
                'road': Image.new("RGBA", (64, 64), (66, 66, 66, 255))
            }
    return ICONS

def get_cell_type_name(cell_type):
    """Trả về tên loại ô dựa trên giá trị"""
    types = {
        -1: "Vị trí xe tải",
        0: "Đường thông thường",
        1: "Trạm thu phí",
        2: "Trạm xăng",
        3: "Vật cản"
    }
    return types.get(cell_type, "Không xác định")

def _draw_map_simple(map_data, start_pos=None):
    """Phiên bản đơn giản nhất cho hiển thị bản đồ khi có lỗi"""
    grid = map_data.grid
    size = grid.shape[0]
    
    # Sử dụng start_pos từ tham số nếu được cung cấp, nếu không lấy từ map_data
    if start_pos is None and hasattr(map_data, 'start_pos'):
        start_pos = map_data.start_pos
        
    # Lấy end_pos từ map_data nếu có
    end_pos = None
    if hasattr(map_data, 'end_pos'):
        end_pos = map_data.end_pos
    
    st.write("#### Bản đồ (Phiên bản đơn giản)")
    
    emojis = {
        'truck': '🚚',
        'gas': '⛽',
        'toll': '🚧',
        'brick': '🧱',
        'road': '🛣️',
        'end': '🏁'
    }
    
    # Tạo bảng dữ liệu để hiển thị bản đồ
    map_table = []
    for i in range(size):
        row = []
        for j in range(size):
            cell_type = grid[i][j]
            
            # Xác định emoji hiển thị
            if start_pos and (i, j) == start_pos:
                emoji = emojis['truck']
            elif end_pos and (i, j) == end_pos:
                emoji = emojis['end']
            elif cell_type == 1:
                emoji = emojis['toll']
            elif cell_type == 2:
                emoji = emojis['gas']
            elif cell_type == -1:
                emoji = emojis['brick']
            else:
                emoji = emojis['road']
                
            row.append(emoji)
        map_table.append(row)
    
    # Hiển thị bản đồ dạng bảng
    st.table(map_table)

def get_grid_from_map_data(map_data):
    """Trích xuất grid từ map_data một cách nhất quán."""
    if hasattr(map_data, 'grid'):
        return map_data.grid
    return map_data

def is_valid_position(grid, pos):
    """Kiểm tra một vị trí có nằm trong lưới hợp lệ không."""
    try:
        return (0 <= pos[0] < grid.shape[1] and 0 <= pos[1] < grid.shape[0])
    except:
        return False

def is_obstacle_cell(grid, pos):
    """Kiểm tra một ô có phải là chướng ngại vật không."""
    try:
        if not is_valid_position(grid, pos):
            return True  # Coi như ô ngoài biên là chướng ngại vật
        return grid[pos[1], pos[0]] == OBSTACLE_VALUE
    except Exception as e:
        print(f"Error checking cell at {pos}: {str(e)}")
        return True  # Coi như ô lỗi là chướng ngại vật để an toàn

def filter_obstacle_cells(grid, positions):
    """
    Lọc bỏ các ô chướng ngại vật từ danh sách vị trí đầu vào.
    Chỉ loại bỏ các ô là chướng ngại vật, không cố gắng sửa chữa tính liên tục.
    
    Args:
        grid: Lưới chứa thông tin về các ô
        positions: Danh sách các vị trí cần kiểm tra
        
    Returns:
        List[Tuple[int, int]]: Danh sách các vị trí mà không chứa ô chướng ngại vật
    """
    if not positions or len(positions) < 1:
        return positions
    
    # Lọc bỏ các ô chướng ngại vật
    filtered = []
    obstacles_found = False
    obstacles_count = 0
    
    for pos in positions:
        # Kiểm tra tính hợp lệ của vị trí
        if not is_valid_position(grid, pos):
            print(f"WARNING: Vị trí {pos} nằm ngoài lưới, bỏ qua.")
            obstacles_found = True
            obstacles_count += 1
            continue
            
        # Kiểm tra chắc chắn rằng ô không phải là chướng ngại vật
        if not is_obstacle_cell(grid, pos):
            filtered.append(pos)
        else:
            obstacles_found = True
            obstacles_count += 1
            print(f"CẢNH BÁO: Lọc bỏ ô chướng ngại vật tại {pos}")
    
    if obstacles_found:
        print(f"CẢNH BÁO: Đã phát hiện và lọc bỏ {obstacles_count} ô chướng ngại vật từ danh sách có {len(positions)} vị trí")
        if obstacles_count > 0:
            st.warning(f"⚠️ Đã phát hiện và lọc bỏ {obstacles_count} ô chướng ngại vật")
    
    return filtered

def find_path_between(grid, start, end):
    """
    Tìm đường đi ngắn nhất giữa hai điểm không liền kề sử dụng BFS.
    Chỉ đi qua các ô không phải chướng ngại vật.
    """
    if start == end:
        return [start]
    
    from collections import deque
    
    # Sử dụng BFS để tìm đường đi
    queue = deque([(start, [start])])
    visited = set([start])
    
    # Các hướng di chuyển: lên, phải, xuống, trái
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    
    while queue:
        (current, path) = queue.popleft()
        
        # Lấy các ô liền kề
        for dx, dy in directions:
            next_x, next_y = current[0] + dx, current[1] + dy
            next_pos = (next_x, next_y)
            
            # Kiểm tra ô có hợp lệ không
            if not is_valid_position(grid, next_pos):
                continue
                
            # Kiểm tra không phải ô chướng ngại vật
            if is_obstacle_cell(grid, next_pos):
                continue
                
            # Kiểm tra đã đến đích chưa
            if next_pos == end:
                return path + [end]
            
            # Kiểm tra đã thăm chưa
            if next_pos not in visited:
                visited.add(next_pos)
                queue.append((next_pos, path + [next_pos]))
                
                # Giới hạn tìm kiếm để tránh trường hợp không tìm thấy đường đi
                if len(visited) > 1000:
                    return None
    
    # Không tìm thấy đường đi
    return None

def draw_map(map_data, start_pos=None, visited=None, current_neighbors=None, current_pos=None, path=None):
    """
    Vẽ bản đồ với các icon sử dụng thành phần bản địa của Streamlit
    
    Parameters:
    - map_data: Đối tượng Map chứa thông tin bản đồ
    - start_pos: Tuple (row, col) chỉ vị trí bắt đầu của xe (nếu có) - LƯU Ý: API này nhận (row, col) nhưng nên nhất quán (x,y)
    - visited: List các vị trí đã thăm (x,y)
    - current_neighbors: List các vị trí hàng xóm đang xét (x,y)
    - current_pos: Tuple (x,y) chỉ vị trí hiện tại
    - path: List các vị trí (x,y) trên đường đi tìm được
    """
    try:
        # Lấy grid từ map_data một cách nhất quán
        grid = get_grid_from_map_data(map_data)
        size = grid.shape[0] # Giả sử là bản đồ vuông
        
        # Lấy vị trí bắt đầu từ map_data nếu không có trong tham số
        if start_pos is None and hasattr(map_data, 'start_pos'):
            start_pos = map_data.start_pos
            if start_pos is not None:
                print(f"INFO: Sử dụng start_pos từ map_data: {start_pos}")
        
        # Ghi log các thông tin debug giúp theo dõi vấn đề start_pos
        if start_pos is None:
            print("INFO: start_pos chưa được thiết lập")
            print(f"DEBUG: start_pos = {start_pos}, kiểu: {type(start_pos)}")
            if hasattr(map_data, 'start_pos'):
                print(f"DEBUG: map_data.start_pos = {map_data.start_pos}, kiểu: {type(map_data.start_pos)}")
            
        # KIỂM TRA KHẨN CẤP: Đảm bảo đường đi không chứa chướng ngại vật
        if path:
            obstacles_in_path = []
            for pos_xy in path: # path chứa (x,y)
                if is_obstacle_cell(grid, pos_xy): # is_obstacle_cell nhận (x,y)
                    obstacles_in_path.append(pos_xy)
            
            if obstacles_in_path:
                st.error(f"❌ LỖI NGHIÊM TRỌNG: Đường đi chứa {len(obstacles_in_path)} ô chướng ngại vật tại vị trí: {obstacles_in_path[:5]}{'...' if len(obstacles_in_path) > 5 else ''}")
                st.warning("⚠️ Đường đi không hợp lệ! Có lỗi nghiêm trọng trong thuật toán tìm đường! Kiểm tra lại thuật toán và phương thức validate_path_no_obstacles.")
        
        # Lọc tất cả các danh sách vị trí để đảm bảo không có ô chướng ngại vật
        # filter_obstacle_cells nhận (grid, list_of_xy_positions)
        if path:
            original_path_len = len(path)
            filtered_path_display_only = filter_obstacle_cells(grid, path) # Dùng để hiển thị lỗi, không thay đổi path gốc
            if len(filtered_path_display_only) < original_path_len:
                st.error(f"⚠️ LỖI HIỂN THỊ: Đường đi chứa {original_path_len - len(filtered_path_display_only)} ô chướng ngại vật bị lọc bởi filter_obstacle_cells!")
                print(f"CRITICAL: draw_map's filter_obstacle_cells removed {original_path_len - len(filtered_path_display_only)} obstacles from path for display purposes!")
            
        if visited:
            original_visited_len = len(visited)
            # visited được truyền vào là list (x,y)
            visited_for_display = filter_obstacle_cells(grid, visited) # visited_for_display giờ là list (x,y) đã lọc
            if len(visited_for_display) < original_visited_len:
                print(f"Đã lọc bỏ {original_visited_len - len(visited_for_display)} ô chướng ngại vật từ danh sách đã thăm (hiển thị)")
        else:
            visited_for_display = []

        if current_neighbors:
            original_neighbors_len = len(current_neighbors)
            # current_neighbors được truyền vào là list (x,y)
            current_neighbors_for_display = filter_obstacle_cells(grid, current_neighbors) # current_neighbors_for_display là (x,y) đã lọc
            if len(current_neighbors_for_display) < original_neighbors_len:
                print(f"Đã lọc bỏ {original_neighbors_len - len(current_neighbors_for_display)} ô chướng ngại vật từ danh sách lân cận (hiển thị)")
        else:
            current_neighbors_for_display = []
        
        # Kiểm tra vị trí hiện tại (x,y)
        if current_pos and is_obstacle_cell(grid, current_pos): # current_pos là (x,y)
            st.error(f"❌ Vị trí hiện tại {current_pos} là ô chướng ngại vật và sẽ bị bỏ qua!")
            current_pos_for_display = None # Không hiển thị current_pos nếu là chướng ngại vật
        else:
            current_pos_for_display = current_pos

        # start_pos và end_pos từ map_data có thể là (hàng, cột) hoặc (y,x) tùy thuộc vào cách nó được tạo
        # Giả định rằng start_pos được truyền vào hàm này là (x,y) để nhất quán
        # Tương tự cho end_pos từ map_data
        
        # Kiểm tra các thành phần khác (giả sử start_pos và end_pos là (x,y))
        if start_pos and is_obstacle_cell(grid, start_pos): 
            st.error(f"❌ Vị trí bắt đầu {start_pos} là ô chướng ngại vật!")
        
        end_pos_xy = None
        if hasattr(map_data, 'end_pos') and map_data.end_pos:
            # Giả định map_data.end_pos là (x,y)
            end_pos_xy = map_data.end_pos 
            if is_obstacle_cell(grid, end_pos_xy):
                st.error(f"❌ Vị trí kết thúc {end_pos_xy} là ô chướng ngại vật!")
        
        st.write("### 🗺️ Bản đồ")
        
        # CSS cho bản đồ và animation (chỉnh màu hiệu ứng lỗi)
        st.markdown("""
        <style>
        /* Reset styles để loại bỏ background từ mọi phần tử */
        .map-container, .map-container *, .map-container *:before, .map-container *:after {
            background: transparent !important;
            background-color: transparent !important;
            box-shadow: none !important;
            border: none !important;
        }
        
        .map-container {
            display: flex;
            justify-content: center;
            margin: 20px 0;
            padding: 25px;
            border-radius: 20px;
            transition: all 0.5s ease;
        }
        
        .map-container table {
            border-collapse: collapse;
            border-radius: 15px;
            overflow: hidden;
            transform: perspective(1200px) rotateX(2deg);
            transition: all 0.5s ease;
        }
        
        .map-container:hover table {
            transform: perspective(1200px) rotateX(0deg);
        }
        
        .map-container td {
            width: 64px;
            height: 64px;
            text-align: center;
            padding: 0;
            position: relative;
            border: none;
            transition: all 0.3s ease;
            overflow: hidden;
        }
        
        .map-container td > div {
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            transition: all 0.3s ease;
        }
        
        .visited-overlay {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(100, 181, 246, 0.05) !important;
            z-index: 1;
            animation: fadeIn 0.7s ease;
        }
        
        .neighbor-overlay {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(255, 215, 0, 0.05) !important;
            z-index: 2;
            animation: pulseGlow 1.5s infinite;
        }
        
        .current-overlay {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(255, 69, 0, 0.08) !important;
            z-index: 3;
            animation: highlightPulse 1.2s infinite;
        }
        
        .path-overlay {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(76, 175, 80, 0.05) !important;
            z-index: 2;
            animation: pathGlow 3s infinite;
        }
        
        .obstacle-in-path-overlay {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(220, 53, 69, 0.2) !important;
            z-index: 10 !important;
            animation: errorBlink 0.8s infinite;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: scale(0.95); }
            to { opacity: 1; transform: scale(1); }
        }
        
        @keyframes pulseGlow {
            0% { opacity: 0.4; }
            50% { opacity: 0.2; }
            100% { opacity: 0.4; }
        }
        
        @keyframes highlightPulse {
            0% { background-color: rgba(255, 69, 0, 0.08) !important; }
            50% { background-color: rgba(255, 69, 0, 0.15) !important; }
            100% { background-color: rgba(255, 69, 0, 0.08) !important; }
        }
        
        @keyframes pathGlow {
            0% { opacity: 0.4; }
            50% { opacity: 0.7; }
            100% { opacity: 0.4; }
        }
        
        @keyframes errorBlink {
            0% { background-color: rgba(220, 53, 69, 0.2) !important; }
            50% { background-color: rgba(220, 53, 69, 0.35) !important; }
            100% { background-color: rgba(220, 53, 69, 0.2) !important; }
        }
        
        .cell-content {
            position: relative;
            z-index: 4;
            font-size: 32px;
            line-height: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            width: 100%;
            height: 100%;
            transition: all 0.3s ease;
        }
        
        .cell-content:hover {
            transform: scale(1.1);
        }
        
        /* Hiệu ứng khi di chuột qua bản đồ */
        .map-container tr {
            transition: all 0.3s ease;
        }
        
        .map-container tr:hover {
            transform: translateY(-2px);
        }
        
        /* Xóa các đường kẻ giữa các ô */
        .map-container td::after {
            display: none;
        }
        
        .current-pos-cell .cell-content {
            animation: pulseTruck 1.2s infinite ease-in-out;
            transform-origin: center;
            z-index: 5;
        }
        
        @keyframes pulseTruck {
            0% { transform: scale(1); }
            50% { transform: scale(1.15); }
            100% { transform: scale(1); }
        }
        
        /* Xe tải luôn hiển thị rõ ràng */
        .truck-icon {
            font-size: 40px !important;
            filter: drop-shadow(0 2px 5px rgba(0,0,0,0.1));
            color: #FF5722;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Debug để kiểm tra start_pos có đúng format không
        print(f"DEBUG: start_pos = {start_pos}, kiểu: {type(start_pos) if start_pos else None}")
        if hasattr(map_data, 'start_pos'):
            print(f"DEBUG: map_data.start_pos = {map_data.start_pos}, kiểu: {type(map_data.start_pos)}")
        
        # Tạo bảng dữ liệu để hiển thị bản đồ
        # Vòng lặp i: hàng (y), j: cột (x)
        map_table = []
        for i_row in range(size): # i_row là y
            row = []
            for j_col in range(size): # j_col là x
                current_cell_xy = (j_col, i_row) # Tọa độ (x,y) của ô đang xét
                cell_type = grid[i_row, j_col] # Truy cập grid bằng (hàng, cột) tức (y,x)
                
                # Xác định loại ô và class CSS tương ứng
                cell_type_class = "cell-type-road"  # Mặc định là đường
                if cell_type == TOLL_VALUE:
                    cell_content = "🚧"
                    cell_type_class = "cell-type-toll"
                elif cell_type == GAS_STATION_VALUE:
                    cell_content = "⛽"
                    cell_type_class = "cell-type-gas"
                elif cell_type == OBSTACLE_VALUE:
                    cell_content = "🧱"
                    cell_type_class = "cell-type-obstacle"
                else: # ROAD_CELL
                    cell_content = ""  # Sử dụng chuỗi rỗng thay vì emoji vuông trắng
                    cell_type_class = "cell-type-road"
                
                # Kiểm tra nếu là vị trí bắt đầu hoặc kết thúc
                if end_pos_xy and current_cell_xy == end_pos_xy:
                    cell_content = "🏁"
                    cell_type_class = "cell-type-end"
                
                # *** THAY ĐỔI QUAN TRỌNG: Luôn ưu tiên hiển thị xe tải ở vị trí bắt đầu ***
                # Debug để kiểm tra có match không
                if start_pos and current_cell_xy == start_pos:
                    print(f"DEBUG: Đã tìm thấy vị trí bắt đầu tại {current_cell_xy}")
                    cell_content = """<span class='truck-icon' style='
                                     font-size: 40px !important; 
                                     filter: drop-shadow(0 2px 5px rgba(0,0,0,0.1)); 
                                     color: #FF5722;'>🚚</span>"""
                    cell_type_class = "cell-type-start current-pos-cell"
                
                # Vị trí hiện tại của xe tải (nếu khác vị trí bắt đầu)
                current_pos_class = ""
                if current_pos_for_display and current_cell_xy == current_pos_for_display:
                    # Chỉ hiển thị xe tải ở vị trí hiện tại nếu khác với vị trí bắt đầu
                    if not start_pos or current_cell_xy != start_pos:
                        cell_content = "<span class='truck-icon'>🚚</span>"
                        current_pos_class = "current-pos-cell"
                
                # Thêm overlay cho các hiệu ứng
                overlays = ""
                # visited_for_display chứa (x,y)
                if visited_for_display and current_cell_xy in visited_for_display:
                    overlays += '<div class="visited-overlay"></div>'
                # current_neighbors_for_display chứa (x,y)
                if current_neighbors_for_display and current_cell_xy in current_neighbors_for_display:
                    overlays += '<div class="neighbor-overlay"></div>'
                # current_pos_for_display là (x,y)
                if current_pos_for_display and current_cell_xy == current_pos_for_display:
                    overlays += '<div class="current-overlay"></div>'
                
                # Xử lý đường đi (path chứa (x,y))
                if path and current_cell_xy in path:
                    # Kiểm tra xem ô này có phải là chướng ngại vật không
                    # is_obstacle_cell nhận (grid, (x,y))
                    if is_obstacle_cell(grid, current_cell_xy):
                        overlays += '<div class="obstacle-in-path-overlay"></div>'
                        cell_content = "❌"  # Đánh dấu lỗi
                    else:
                        overlays += '<div class="path-overlay"></div>'
                        # Không hiển thị mũi tên nếu là vị trí hiện tại của xe hoặc vị trí bắt đầu
                        if (not current_pos_for_display or current_cell_xy != current_pos_for_display) and (not start_pos or current_cell_xy != start_pos):
                            # Không hiển thị mũi tên ở điểm cuối của đường đi
                            if current_cell_xy != path[-1]: 
                                try:
                                    idx = path.index(current_cell_xy)
                                    if idx + 1 < len(path):
                                        next_pos_xy = path[idx + 1] # next_pos_xy là (x,y)
                                        # Xác định hướng mũi tên từ current_cell_xy (x,y) đến next_pos_xy (x,y)
                                        delta_x = next_pos_xy[0] - current_cell_xy[0]
                                        delta_y = next_pos_xy[1] - current_cell_xy[1]

                                        if delta_y < 0: # Đi lên (giảm y)
                                            cell_content = "⬆️"
                                        elif delta_y > 0: # Đi xuống (tăng y)
                                            cell_content = "⬇️"
                                        elif delta_x < 0: # Đi trái (giảm x)
                                            cell_content = "⬅️"
                                        elif delta_x > 0: # Đi phải (tăng x)
                                            cell_content = "➡️"
                                except ValueError:
                                    # current_cell_xy có thể không nằm trong path nếu path bị lọc
                                    pass 
                
                # Tạo cell với class và overlay, thêm inline style để đảm bảo trong suốt
                cell = f'<td class="{cell_type_class} {current_pos_class}" style="background: transparent !important;"><div style="background: transparent !important;">{overlays}<div class="cell-content" style="background: transparent !important;">{cell_content}</div></div></td>'
                row.append(cell)
            map_table.append(row)
        
        # Hiển thị thông tin debug
        if start_pos:
            print(f"Vị trí xe tải (start_pos): {start_pos}")
        if current_pos:
            print(f"Vị trí hiện tại (current_pos): {current_pos}")
        
        # Hiển thị bản đồ dạng bảng với HTML
        st.markdown(
            f"""
            <div class="map-container" style="background: transparent !important;">
                <table style="background: transparent !important;">
                    {''.join(f"<tr style='background: transparent !important;'>{''.join(cell for cell in row)}</tr>" for row in map_table)}
                </table>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    except Exception as e:
        st.error(f"Lỗi khi hiển thị bản đồ: {str(e)}")
        # Hiển thị traceback đầy đủ cho debug
        import traceback
        st.error(traceback.format_exc())
        # Cung cấp map_data, và start_pos (nếu có, giả sử là (x,y)) cho hàm fallback
        _draw_map_simple(map_data, start_pos if start_pos else None)

def draw_route(map_data, route):
    """
    Vẽ tuyến đường trên bản đồ sử dụng thành phần bản địa của Streamlit
    
    Parameters:
    - map_data: Đối tượng Map chứa thông tin bản đồ
    - route: Danh sách các vị trí [(row1, col1), (row2, col2), ...] thể hiện tuyến đường
    """
    try:
        # Lấy grid từ map_data
        grid = get_grid_from_map_data(map_data)
        
        # Kiểm tra xem đường đi có chứa ô chướng ngại vật không
        if route:
            obstacle_positions = []
            for pos in route:
                if is_obstacle_cell(grid, pos):
                    obstacle_positions.append(pos)
            
            if obstacle_positions:
                st.error(f"⚠️ LỖI NGHIÊM TRỌNG: Đường đi chứa {len(obstacle_positions)} ô chướng ngại vật tại vị trí: {obstacle_positions[:5]}{'...' if len(obstacle_positions) > 5 else ''}")
                st.warning("Đường đi không hợp lệ! Vui lòng kiểm tra lại thuật toán tìm đường.")
        
        if not route or len(route) < 2:
            st.warning("⚠️ Không có đủ điểm để hiển thị tuyến đường!")
            draw_map(map_data, route[0] if route and len(route) > 0 else None)
            return
        
        st.write("### 🗺️ Bản đồ với Tuyến Đường")
        
        # Sử dụng emoji trực tiếp thay vì icon
        emojis = {
            'truck': '🚚',
            'gas': '⛽',
            'toll': '🚧',
            'brick': '🧱',
            'road': '🛣️',
            'route': '📍', # Emoji cho các bước trên tuyến đường
            'start': '🚩', # Emoji cho điểm bắt đầu
            'end': '🏁',   # Emoji cho điểm kết thúc
            'error': '❌'   # Emoji cho ô lỗi (đi qua chướng ngại vật)
        }
        
        # Lấy kích thước của grid
        size = grid.shape[0] if hasattr(grid, 'shape') else len(grid)
        
        # Tạo DataFrame để hiển thị bản đồ với tuyến đường
        map_data_display = []
        for i in range(size):
            row = []
            for j in range(size):
                cell_type = grid[i][j]
                route_marker = ""
                is_obstacle = is_obstacle_cell(grid, (i, j))
                
                # Xác định vị trí trong tuyến đường
                if (i, j) in route:
                    pos_index = route.index((i, j))
                    if pos_index == 0:
                        # Điểm bắt đầu
                        emoji = emojis['truck']
                        route_marker = "1"
                    elif pos_index == len(route) - 1:
                        # Điểm kết thúc
                        emoji = emojis['end']
                        route_marker = str(len(route))
                    else:
                        # Điểm trên tuyến đường
                        if is_obstacle:
                            emoji = emojis['error']  # Đánh dấu lỗi
                        elif cell_type == 1:
                            emoji = emojis['toll']
                        elif cell_type == 2:
                            emoji = emojis['gas']
                        else:
                            emoji = emojis['road']
                        route_marker = str(pos_index + 1)
                else:
                    # Vị trí không nằm trên tuyến đường
                    if is_obstacle:
                        emoji = emojis['brick']
                    elif cell_type == 1:
                        emoji = emojis['toll']
                    elif cell_type == 2:
                        emoji = emojis['gas']
                    else:
                        emoji = emojis['road']
                
                # Thêm số thứ tự vào các ô trên tuyến đường
                if route_marker:
                    if (i, j) == route[0]:
                        cell_display = f"{emoji} {emojis['start']}{route_marker}"
                    elif (i, j) == route[-1]:
                        cell_display = f"{emoji} {emojis['end']}{route_marker}"
                    else:
                        if is_obstacle:
                            cell_display = f"{emoji} {emojis['error']}{route_marker}"
                        else:
                            cell_display = f"{emoji} {emojis['route']}{route_marker}"
                else:
                    cell_display = emoji
                
                row.append(cell_display)
            map_data_display.append(row)
        
        # Hiển thị bản đồ dạng bảng
        st.table(map_data_display)
        
        # Hiển thị thông tin tuyến đường
        st.info("📍 Thông tin tuyến đường")
        total_toll = sum(1 for pos in route if grid[pos[1]][pos[0]] == 1)
        total_gas = sum(1 for pos in route if grid[pos[1]][pos[0]] == 2)
        total_obstacles = sum(1 for pos in route if is_obstacle_cell(grid, pos))
        
        route_info_cols = st.columns(4)
        with route_info_cols[0]:
            st.metric("Độ dài tuyến đường", f"{len(route) - 1} bước")
        with route_info_cols[1]:
            st.metric("Trạm thu phí", total_toll)
        with route_info_cols[2]:
            st.metric("Trạm xăng", total_gas)
        with route_info_cols[3]:
            if total_obstacles > 0:
                st.metric("Ô chướng ngại vật", total_obstacles, delta=-total_obstacles, delta_color="inverse")
            else:
                st.metric("Ô chướng ngại vật", 0)
        
        if total_obstacles > 0:
            st.error("⚠️ Đường đi qua chướng ngại vật không hợp lệ!")
        
    except Exception as e:
        st.error(f"Lỗi khi hiển thị tuyến đường: {str(e)}")
        # Hiển thị bản đồ bình thường khi có lỗi
        draw_map(map_data, route[0] if route and len(route) > 0 else None)

def draw_animation(map_data, states):
    """
    Tạo animation cho quá trình di chuyển sử dụng thành phần bản địa của Streamlit
    
    Parameters:
    - map_data: Đối tượng Map chứa thông tin bản đồ
    - states: Danh sách các trạng thái [(pos1, fuel1), (pos2, fuel2), ...]
    """
    if not states or len(states) < 2:
        st.warning("⚠️ Không đủ trạng thái để tạo animation!")
        if states and len(states) > 0:
            # Ưu tiên sử dụng start_pos từ map_data nếu có
            start_pos = None
            if hasattr(map_data, 'start_pos'):
                start_pos = map_data.start_pos
            draw_map(map_data, start_pos if start_pos else states[0][0])  # Hiển thị vị trí bắt đầu hoặc vị trí đầu tiên
        else:
            # Nếu không có states, vẫn cố gắng hiển thị bản đồ với vị trí bắt đầu
            start_pos = None
            if hasattr(map_data, 'start_pos'):
                start_pos = map_data.start_pos
            draw_map(map_data, start_pos)
        return
    
    try:
        # Lấy vị trí bắt đầu từ map_data nếu có
        start_pos = None
        if hasattr(map_data, 'start_pos'):
            start_pos = map_data.start_pos
        
        # Lấy danh sách các vị trí từ states
        positions = [state[0] for state in states]
        fuels = [state[1] for state in states]
        
        # Hiển thị bảng điều khiển animation
        st.subheader("🎬 Animation")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            step = st.slider("Bước di chuyển", 0, len(states) - 1, 0)
        
        with col2:
            st.text("Step: " + str(step+1) + "/" + str(len(states)))
        
        # Hiển thị trạng thái hiện tại
        current_pos = positions[step]
        current_fuel = fuels[step]
        
        # Tạo bản đồ với vị trí hiện tại được đánh dấu
        grid = map_data.grid
        size = grid.shape[0]
        
        # Sử dụng emoji trực tiếp thay vì icon
        emojis = {
            'truck': '🚚',
            'gas': '⛽',
            'toll': '🚧',
            'brick': '🧱',
            'road': '🛣️',
            'route': '📍', # Emoji cho các bước đã đi qua
            'current': '📌' # Emoji cho vị trí hiện tại
        }
        
        # CSS cho bản đồ animation
        st.markdown("""
        <style>
        /* Reset styles for elements *inside* the map-container if necessary, but not map-container itself */
        .map-anim-container, .map-anim-container *, .map-anim-container *:before, .map-anim-container *:after {
            background: transparent !important;
            background-color: transparent !important;
            box-shadow: none !important;
            border: none !important;
        }
        
        .map-anim-container table {
            border-collapse: collapse;
        }
        
        .anim-cell {
            padding: 10px;
            text-align: center;
            font-size: 32px;
            position: relative;
        }
        
        .anim-truck {
            font-size: 35px !important; 
            filter: drop-shadow(0 2px 5px rgba(0,0,0,0.1));
        }
        
        .anim-marker {
            font-size: 18px;
            color: #1E88E5;
        }
        
        .anim-current {
            color: #FF5722;
        }
        
        .anim-path {
            color: #4CAF50;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Tạo DataFrame để hiển thị bản đồ với tuyến đường và vị trí hiện tại
        map_table = []
        for i_row in range(size): # i_row là y
            row = []
            for j_col in range(size): # j_col là x
                current_cell_xy = (j_col, i_row) # Tọa độ (x,y) của ô đang xét
                cell_type = grid[i_row, j_col] # Truy cập grid bằng (hàng, cột) tức (y,x)
                
                # Xác định loại ô và class CSS tương ứng
                cell_type_class = "cell-type-road" 
                if cell_type == TOLL_VALUE:
                    cell_content = "🚧"
                    cell_type_class = "cell-type-toll"
                elif cell_type == GAS_STATION_VALUE:
                    cell_content = "⛽"
                    cell_type_class = "cell-type-gas"
                elif cell_type == OBSTACLE_VALUE:
                    cell_content = "🧱"
                    cell_type_class = "cell-type-obstacle"
                else: # ROAD_CELL
                    cell_content = ""  # Sử dụng chuỗi rỗng thay vì emoji vuông trắng
                    cell_type_class = "cell-type-road"
                
                # IMPORTANT: Hiển thị xe tải ở vị trí bắt đầu mọi lúc
                if start_pos and current_cell_xy == start_pos:
                    cell_content = "<span class='anim-truck'>🚚</span>"
                    cell_type_class = "cell-type-start"
                    position_marker = "<span class='anim-marker'>Bắt đầu</span>"
                # Vị trí hiện tại của xe (nếu khác vị trí bắt đầu)
                elif current_cell_xy == current_pos and (not start_pos or current_pos != start_pos):
                    cell_content = "<span class='anim-truck'>🚚</span>"
                    position_marker = "<span class='anim-marker anim-current'>Hiện tại</span>"
                # Vị trí đã đi qua (ngoại trừ vị trí bắt đầu)
                elif current_cell_xy in positions[:step] and (not start_pos or current_cell_xy != start_pos):
                    position_marker = "<span class='anim-marker anim-path'>Đã đi</span>"
                else:
                    position_marker = ""
                
                # Tạo cell HTML
                cell = f"""<td class="anim-cell {cell_type_class}" style="background: transparent !important;">
                    <div style="background: transparent !important;">
                        {cell_content}
                        <div>{position_marker}</div>
                    </div>
                </td>"""
                row.append(cell)
            map_table.append(row)
        
        # Hiển thị bản đồ dạng bảng với HTML
        st.markdown(
            f"""
            <div class="map-anim-container" style="background: transparent !important;">
                <table style="background: transparent !important;">
                    {''.join(f"<tr style='background: transparent !important;'>{''.join(cell for cell in row)}</tr>" for row in map_table)}
                </table>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Hiển thị lượng nhiên liệu
        st.subheader("🚚 Trạng thái hiện tại")
        status_cols = st.columns(2)
        
        with status_cols[0]:
            st.metric("Vị trí", f"[{current_pos[0]}, {current_pos[1]}]")
        
        with status_cols[1]:
            st.metric("Lượng nhiên liệu", f"{current_fuel:.1f}")
            
        # Thanh nhiên liệu
        fuel_percentage = current_fuel * 100 / 20.0  # Giả sử nhiên liệu tối đa là 10
        
        # Sử dụng progress bar của Streamlit
        fuel_color = "normal"
        if fuel_percentage <= 10:
            st.error(f"Nhiên liệu: {fuel_percentage:.1f}%")
        elif fuel_percentage <= 30:
            st.warning(f"Nhiên liệu: {fuel_percentage:.1f}%")
        else:
            st.success(f"Nhiên liệu: {fuel_percentage:.1f}%")
            
        st.progress(fuel_percentage/100.0)  # Streamlit progress nhận giá trị từ 0-1
    
    except Exception as e:
        st.error(f"Lỗi khi tạo animation: {str(e)}")
        # Hiển thị bản đồ thông thường với vị trí bắt đầu
        start_pos = None
        if hasattr(map_data, 'start_pos'):
            start_pos = map_data.start_pos
        draw_map(map_data, start_pos if start_pos else (states[0][0] if states and len(states) > 0 else None))