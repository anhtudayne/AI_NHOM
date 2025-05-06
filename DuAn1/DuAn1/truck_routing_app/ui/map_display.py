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


# Đường dẫn tới các icon
ICONS_PATH = os.path.join(os.path.dirname(__file__), 'units')

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
            elif cell_type == 3:
                emoji = emojis['brick']
            else:
                emoji = emojis['road']
                
            row.append(emoji)
        map_table.append(row)
    
    # Hiển thị bản đồ dạng bảng
    st.table(map_table)

def draw_map(map_data, start_pos=None, visited=None, current_neighbors=None, current_pos=None, path=None):
    """
    Vẽ bản đồ với các icon sử dụng thành phần bản địa của Streamlit
    
    Parameters:
    - map_data: Đối tượng Map chứa thông tin bản đồ
    - start_pos: Tuple (row, col) chỉ vị trí bắt đầu của xe (nếu có)
    - visited: List các vị trí đã thăm
    - current_neighbors: List các vị trí hàng xóm đang xét
    - current_pos: Tuple (row, col) chỉ vị trí hiện tại
    - path: List các vị trí trên đường đi tìm được
    """
    try:
        grid = map_data.grid
        size = grid.shape[0]
        
        # Sử dụng start_pos từ tham số nếu được cung cấp, nếu không lấy từ map_data
        if start_pos is None and hasattr(map_data, 'start_pos'):
            start_pos = map_data.start_pos
            
        # Lấy end_pos từ map_data nếu có
        end_pos = None
        if hasattr(map_data, 'end_pos'):
            end_pos = map_data.end_pos
        
        st.write("### 🗺️ Bản đồ")
        
        # CSS cho bản đồ và animation
        st.markdown("""
        <style>
        .map-container {
            display: flex;
            justify-content: center;
            margin: 20px 0;
            padding: 20px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .map-container table {
            border-collapse: collapse;
        }
        .map-container td {
            width: 60px;
            height: 60px;
            text-align: center;
            font-size: 24px;
            padding: 0;
            position: relative;
            transition: all 0.3s ease;
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
            background-color: rgba(100, 181, 246, 0.6) !important;
            z-index: 1;
            animation: fadeIn 0.5s ease;
        }
        .neighbor-overlay {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(255, 215, 0, 0.3);
            border: 2px dashed #ffd700;
            z-index: 2;
            animation: pulse 1s infinite;
        }
        .current-overlay {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(255, 69, 0, 0.5);
            z-index: 3;
            animation: highlight 1s infinite;
        }
        .path-overlay {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(76, 175, 80, 0.5);
            border: 2px solid #4CAF50;
            z-index: 2;
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        @keyframes pulse {
            0% { transform: scale(1); opacity: 0.8; }
            50% { transform: scale(1.05); opacity: 0.5; }
            100% { transform: scale(1); opacity: 0.8; }
        }
        @keyframes highlight {
            0% { background-color: rgba(255, 69, 0, 0.5); }
            50% { background-color: rgba(255, 69, 0, 0.8); }
            100% { background-color: rgba(255, 69, 0, 0.5); }
        }
        .cell-content {
            position: relative;
            z-index: 4;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Tạo bảng dữ liệu để hiển thị bản đồ
        map_table = []
        for i in range(size):
            row = []
            for j in range(size):
                cell_type = grid[i][j]
                
                # Xác định emoji và background color cho ô
                if current_pos and (i, j) == current_pos:
                    cell_content = "🚚"
                    bg_color = "#e3f2fd"
                elif start_pos and (i, j) == start_pos and not current_pos and not visited:  # Chỉ hiển thị xe ở vị trí bắt đầu khi chưa bắt đầu trực quan
                    cell_content = "🚚"
                    bg_color = "#e3f2fd"
                elif end_pos and (i, j) == end_pos:
                    cell_content = "🏁"
                    bg_color = "#fff9c4"
                elif cell_type == 1:
                    cell_content = "🚧"
                    bg_color = "#ffebee"
                elif cell_type == 2:
                    cell_content = "⛽"
                    bg_color = "#e8f5e9"
                elif cell_type == 3:
                    cell_content = "🧱"
                    bg_color = "#efebe9"
                else:
                    cell_content = "⬜"
                    bg_color = "#ffffff"
                
                # Thêm overlay cho các hiệu ứng
                overlays = ""
                if visited and (i, j) in visited:
                    overlays += '<div class="visited-overlay"></div>'
                if current_neighbors and (i, j) in current_neighbors:
                    overlays += '<div class="neighbor-overlay"></div>'
                if current_pos and (i, j) == current_pos:
                    overlays += '<div class="current-overlay"></div>'
                if path and (i, j) in path:
                    overlays += '<div class="path-overlay"></div>'
                    if not current_pos or (i, j) != current_pos:  # Không hiển thị mũi tên nếu là vị trí hiện tại
                        if (i, j) != path[-1]:  # Không hiển thị mũi tên ở điểm cuối
                            next_pos = path[path.index((i, j)) + 1]
                            # Xác định hướng mũi tên
                            if next_pos[0] < i:
                                cell_content = "⬆️"  # Lên
                            elif next_pos[0] > i:
                                cell_content = "⬇️"  # Xuống
                            elif next_pos[1] < j:
                                cell_content = "⬅️"  # Trái
                            else:
                                cell_content = "➡️"  # Phải
                
                # Tạo cell với background color và overlay
                cell = f'<div style="background-color: {bg_color};">{overlays}<div class="cell-content">{cell_content}</div></div>'
                row.append(cell)
            map_table.append(row)
        
        # Hiển thị bản đồ dạng bảng với HTML
        st.markdown(
            f"""
            <div class="map-container">
                <table>
                    {''.join(f"<tr>{''.join(f'<td>{cell}</td>' for cell in row)}</tr>" for row in map_table)}
                </table>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    except Exception as e:
        st.error(f"Lỗi khi hiển thị bản đồ: {str(e)}")
        _draw_map_simple(map_data, start_pos)

def draw_route(map_data, route):
    """
    Vẽ tuyến đường trên bản đồ sử dụng thành phần bản địa của Streamlit
    
    Parameters:
    - map_data: Đối tượng Map chứa thông tin bản đồ
    - route: Danh sách các vị trí [(row1, col1), (row2, col2), ...] thể hiện tuyến đường
    """
    if not route or len(route) < 2:
        st.warning("⚠️ Chưa có tuyến đường để hiển thị!")
        draw_map(map_data, route[0] if route else None)
        return
    
    try:
        grid = map_data.grid
        size = grid.shape[0]
        
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
            'end': '🏁'    # Emoji cho điểm kết thúc
        }
        
        # Tạo DataFrame để hiển thị bản đồ với tuyến đường
        map_data_display = []
        for i in range(size):
            row = []
            for j in range(size):
                cell_type = grid[i][j]
                route_marker = ""
                
                # Xác định vị trí trong tuyến đường
                if (i, j) == route[0]:
                    # Điểm bắt đầu
                    emoji = emojis['truck']
                    route_marker = "1"
                elif (i, j) == route[-1]:
                    # Điểm kết thúc
                    if cell_type == 1:
                        emoji = emojis['toll']
                    elif cell_type == 2:
                        emoji = emojis['gas']
                    elif cell_type == 3:
                        emoji = emojis['brick']
                    else:
                        emoji = emojis['road']
                    route_marker = str(len(route))
                elif (i, j) in route:
                    # Điểm trên tuyến đường
                    if cell_type == 1:
                        emoji = emojis['toll']
                    elif cell_type == 2:
                        emoji = emojis['gas']
                    elif cell_type == 3:
                        emoji = emojis['brick']
                    else:
                        emoji = emojis['road']
                    route_marker = str(route.index((i, j)) + 1)
                else:
                    # Điểm không nằm trên tuyến đường
                    if cell_type == 1:
                        emoji = emojis['toll']
                    elif cell_type == 2:
                        emoji = emojis['gas']
                    elif cell_type == 3:
                        emoji = emojis['brick']
                    else:
                        emoji = emojis['road']
                
                # Thêm số thứ tự vào các ô trên tuyến đường
                if route_marker:
                    if (i, j) == route[0]:
                        cell_display = f"{emoji} {emojis['start']}{route_marker}"
                    elif (i, j) == route[-1]:
                        cell_display = f"{emoji} {emojis['end']}{route_marker}"
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
        total_toll = sum(1 for pos in route if grid[pos[0]][pos[1]] == 1)
        total_gas = sum(1 for pos in route if grid[pos[0]][pos[1]] == 2)
        
        route_info_cols = st.columns(3)
        with route_info_cols[0]:
            st.metric("Độ dài tuyến đường", f"{len(route) - 1} bước")
        with route_info_cols[1]:
            st.metric("Trạm thu phí", total_toll)
        with route_info_cols[2]:
            st.metric("Trạm xăng", total_gas)
        
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
            draw_map(map_data, states[0][0])  # Chỉ hiển thị vị trí đầu tiên
        return
    
    try:
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
        
        # Tạo DataFrame để hiển thị bản đồ với tuyến đường và vị trí hiện tại
        map_data_display = []
        for i in range(size):
            row = []
            for j in range(size):
                cell_type = grid[i][j]
                position_marker = ""
                
                # Xác định vị trí hiện tại và đã đi qua
                if (i, j) == current_pos:
                    # Vị trí hiện tại
                    emoji = emojis['truck']
                    position_marker = "current"
                elif (i, j) in positions[:step]:
                    # Vị trí đã đi qua
                    if cell_type == 1:
                        emoji = emojis['toll']
                    elif cell_type == 2:
                        emoji = emojis['gas']
                    elif cell_type == 3:
                        emoji = emojis['brick']
                    else:
                        emoji = emojis['road']
                    position_marker = "past"
                else:
                    # Vị trí bình thường
                    if cell_type == 1:
                        emoji = emojis['toll']
                    elif cell_type == 2:
                        emoji = emojis['gas']
                    elif cell_type == 3:
                        emoji = emojis['brick']
                    else:
                        emoji = emojis['road']
                
                # Thêm đánh dấu vào các ô đã đi qua và vị trí hiện tại
                if position_marker == "current":
                    cell_display = f"{emoji} {emojis['current']}"
                elif position_marker == "past":
                    cell_display = f"{emoji} {emojis['route']}"
                else:
                    cell_display = emoji
                
                row.append(cell_display)
            map_data_display.append(row)
        
        # Hiển thị bản đồ dạng bảng
        st.table(map_data_display)
        
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
        # Hiển thị bản đồ thông thường
        if states and len(states) > 0:
            draw_map(map_data, states[0][0])  # Chỉ hiển thị vị trí đầu tiên