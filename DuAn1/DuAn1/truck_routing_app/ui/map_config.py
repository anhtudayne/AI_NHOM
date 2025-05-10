import streamlit as st
import numpy as np
from core.map import Map
from ui.map_display import draw_map
import os
import pandas as pd

def render_map_config():
    st.title("🗺️ Tạo Bản Đồ & Cấu Hình")
    
    # Layout chính
    col_sidebar, col_main = st.columns([2, 3])
    
    with col_sidebar:
        with st.container():
            st.subheader("⚙️ Cấu hình bản đồ")
            
            # Chọn phương thức tạo bản đồ đầu tiên
            map_type = st.radio(
                "Phương thức tạo bản đồ",
                ["Tạo tự động", "Vẽ thủ công"],
                help="Chọn cách bạn muốn tạo bản đồ"
            )
            
            # Thông số chung cho cả hai phương thức
            map_size = st.slider(
                "Kích thước bản đồ (n x n)", 
                min_value=8, 
                max_value=15, 
                value=10,
                help="Chọn kích thước của bản đồ (số hàng và cột)"
            )
            
            # Thiết lập cho phương thức tạo tự động
            if map_type == "Tạo tự động":
                st.write("#### Tỷ lệ các loại ô")
                
                st.info("""
                Các loại ô trên bản đồ:
                - Đường thông thường: Xe có thể đi qua
                - Trạm thu phí: Xe đi qua phải trả phí
                - Trạm xăng: Điểm xe có thể dừng để đổ xăng
                - Vật cản: Ô không thể đi qua
                """)
                
                # Thiết lập giá trị mặc định hợp lý hơn
                toll_ratio = st.slider(
                    "Tỷ lệ trạm thu phí (%)", 
                    min_value=0, 
                    max_value=15, 
                    value=5,
                    help="Tỷ lệ ô trạm thu phí trên bản đồ"
                )
                
                gas_ratio = st.slider(
                    "Tỷ lệ trạm xăng (%)", 
                    min_value=2, 
                    max_value=10, 
                    value=4,
                    help="Tỷ lệ ô trạm xăng trên bản đồ"
                )
                
                brick_ratio = st.slider(
                    "Tỷ lệ vật cản (%)", 
                    min_value=0, 
                    max_value=30, 
                    value=15, 
                    help="Tỷ lệ ô không đi được (gạch) trên bản đồ"
                )
                
                # Tự động tính toán tỷ lệ đường thông thường và hiển thị cho người dùng
                road_ratio = 100 - toll_ratio - gas_ratio - brick_ratio
                if road_ratio < 50:
                    st.warning(f"⚠️ Tỷ lệ đường thông thường ({road_ratio}%) khá thấp, có thể gây khó khăn cho việc tìm đường!")
                else:
                    st.info(f"Tỷ lệ đường thông thường: {road_ratio}%")
                
                # Tùy chỉnh random seed
                random_seed = st.number_input(
                    "Seed ngẫu nhiên", 
                    min_value=0, 
                    max_value=9999, 
                    value=42,
                    help="Dùng seed để tạo bản đồ ngẫu nhiên có thể tái hiện lại"
                )
                
                # Tab cho các chế độ tạo bản đồ
                map_mode = st.radio(
                    "Chế độ tạo bản đồ", 
                    ["Ngẫu nhiên thông minh", "Bản đồ mẫu"],
                    help="Chọn chế độ tạo bản đồ"
                )
                
                if map_mode == "Ngẫu nhiên thông minh":
                    # Nút tạo bản đồ mới
                    if st.button("🔄 Tạo bản đồ", use_container_width=True):
                        with st.spinner("Đang tạo bản đồ mới..."):
                            np.random.seed(random_seed)
                            st.session_state.map = Map.generate_random(
                                map_size, 
                                toll_ratio/100, 
                                gas_ratio/100, 
                                brick_ratio/100
                            )
                            # Sử dụng các vị trí đã được tự động tạo trong map
                            st.session_state.start_pos = st.session_state.map.start_pos
                            st.session_state.end_pos = st.session_state.map.end_pos
                            
                            st.toast("✅ Đã tạo bản đồ mới thành công!")
                else:
                    # Nút tạo bản đồ mẫu
                    if st.button("🎮 Tạo bản đồ mẫu", use_container_width=True):
                        with st.spinner("Đang tạo bản đồ mẫu..."):
                            # Sử dụng phương thức create_demo_map mới
                            st.session_state.map = Map.create_demo_map(map_size)
                            # Vị trí bắt đầu và điểm đích được đặt trong phương thức create_demo_map
                            st.session_state.start_pos = st.session_state.map.start_pos
                            st.session_state.end_pos = st.session_state.map.end_pos
                            st.toast("✅ Đã tạo bản đồ mẫu thành công!")
                    
                    # Hiển thị hướng dẫn về bản đồ mẫu
                    st.info("""
                    Bản đồ mẫu được thiết kế với:
                    - Trạm xăng ở các góc và trung tâm
                    - Trạm thu phí dọc theo đường vành đai
                    - Vật cản tạo thành mê cung đơn giản
                    - Vị trí bắt đầu ở góc trên bên trái
                    - Điểm đích ở góc dưới bên phải
                    """)
        
        # Phần lưu/tải bản đồ
        with st.container():
            st.divider()
            st.subheader("💾 Lưu & Tải bản đồ")
            
            # Lưu bản đồ
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Lưu bản đồ", use_container_width=True):
                    if 'map' in st.session_state:
                        # Kiểm tra đã có vị trí bắt đầu và điểm đích chưa
                        if st.session_state.start_pos is None:
                            st.warning("⚠️ Chưa thiết lập vị trí bắt đầu của xe!")
                        elif st.session_state.end_pos is None:
                            st.warning("⚠️ Chưa thiết lập điểm đích!")
                        else:
                            # Lưu cả vị trí bắt đầu và điểm đích
                            st.session_state.map.start_pos = st.session_state.start_pos
                            st.session_state.map.end_pos = st.session_state.end_pos
                            filename = st.session_state.map.save()
                            st.success(f"✅ Đã lưu bản đồ vào: {os.path.basename(filename)}")
                    else:
                        st.error("❌ Chưa có bản đồ để lưu!")
            
            with col2:
                # Tạo bản đồ trống
                if st.button("🧹 Tạo bản đồ trống", use_container_width=True):
                    st.session_state.map = Map(map_size)
                    st.session_state.start_pos = None
                    st.session_state.end_pos = None
                    st.toast("✅ Đã tạo bản đồ trống!")
            
            # Tải bản đồ đã lưu
            if os.path.exists('maps'):
                map_files = [f for f in os.listdir('maps') if f.endswith('.json') and f != 'latest_map.json']
                
                if map_files:
                    # Sắp xếp file theo thời gian tạo (mới nhất lên đầu)
                    map_files.sort(key=lambda x: os.path.getmtime(os.path.join('maps', x)), reverse=True)
                    
                    st.write("#### Chọn bản đồ đã lưu:")
                    selected_map = st.selectbox(
                        "Bản đồ đã lưu",
                        options=map_files,
                        format_func=lambda x: f"{x.split('_')[1]} - {' '.join(x.split('_')[2:]).split('.')[0]}"
                    )
                    
                    if st.button("📂 Tải bản đồ", use_container_width=True):
                        with st.spinner("Đang tải bản đồ..."):
                            loaded_map = Map.load(selected_map)
                            if loaded_map:
                                st.session_state.map = loaded_map
                                st.session_state.start_pos = loaded_map.start_pos
                                st.session_state.end_pos = loaded_map.end_pos
                                st.success(f"✅ Đã tải bản đồ: {selected_map}")
                            else:
                                st.error(f"❌ Không thể tải bản đồ: {selected_map}")
                else:
                    st.info("ℹ️ Chưa có bản đồ được lưu trữ")
    
    with col_main:
        # Khởi tạo bản đồ nếu chưa có
        if 'map' not in st.session_state:
            st.session_state.map = Map(map_size)
            
        # Khởi tạo vị trí bắt đầu và điểm đích nếu chưa có
        if 'start_pos' not in st.session_state:
            st.session_state.start_pos = None
        if 'end_pos' not in st.session_state:
            st.session_state.end_pos = None
        
        # Sử dụng tabs cho nội dung chính
        tab1, tab2, tab3 = st.tabs([
            "🗺️ Hiển thị bản đồ", 
            "✏️ Chỉnh sửa bản đồ",
            "📊 Thống kê"
        ])
        
        with tab1:
            with st.container():
                st.subheader("🚚 Thiết lập vị trí bắt đầu và điểm đích")
                
                # Phần thiết lập vị trí bắt đầu
                st.write("##### Vị trí bắt đầu")
                st.info("Vị trí bắt đầu là nơi xe tải sẽ xuất phát. Vị trí này phải nằm trên ô đường thông thường.")
                
                col1, col2, col3 = st.columns([2, 2, 3])
                with col1:
                    start_row = st.number_input(
                        "Hàng (bắt đầu)", 
                        min_value=0, 
                        max_value=st.session_state.map.size-1, 
                        value=0 if st.session_state.start_pos is None else st.session_state.start_pos[0],
                        key="start_row"
                    )
                
                with col2:
                    start_col = st.number_input(
                        "Cột (bắt đầu)", 
                        min_value=0, 
                        max_value=st.session_state.map.size-1, 
                        value=0 if st.session_state.start_pos is None else st.session_state.start_pos[1],
                        key="start_col"
                    )
                
                with col3:
                    # Nút thiết lập vị trí bắt đầu
                    if st.button("🚩 Thiết lập vị trí bắt đầu", use_container_width=True, key="set_start_pos"):
                        # Kiểm tra xem vị trí có phải là đường đi không (loại 0)
                        if st.session_state.map.grid[start_row][start_col] == 0:
                            st.session_state.start_pos = (start_row, start_col)
                            st.success(f"✅ Đã đặt vị trí bắt đầu tại ({start_row}, {start_col})")
                        else:
                            st.error("❌ Vị trí bắt đầu phải là ô đường thông thường!")
                
                # Phần thiết lập điểm đích
                st.write("##### Điểm đích")
                st.info("Điểm đích là nơi xe tải cần đến. Điểm đích phải nằm trên ô đường thông thường.")
                
                col1, col2, col3 = st.columns([2, 2, 3])
                with col1:
                    end_row = st.number_input(
                        "Hàng (đích)", 
                        min_value=0, 
                        max_value=st.session_state.map.size-1, 
                        value=st.session_state.map.size-1 if st.session_state.end_pos is None else st.session_state.end_pos[0],
                        key="end_row"
                    )
                
                with col2:
                    end_col = st.number_input(
                        "Cột (đích)", 
                        min_value=0, 
                        max_value=st.session_state.map.size-1, 
                        value=st.session_state.map.size-1 if st.session_state.end_pos is None else st.session_state.end_pos[1],
                        key="end_col"
                    )
                
                with col3:
                    # Nút thiết lập điểm đích
                    if st.button("🏁 Thiết lập điểm đích", use_container_width=True, key="set_end_pos"):
                        # Kiểm tra xem vị trí có phải là đường đi không (loại 0)
                        if st.session_state.map.grid[end_row][end_col] == 0:
                            # Kiểm tra xem vị trí đích có trùng với vị trí bắt đầu không
                            if st.session_state.start_pos and (end_row, end_col) == st.session_state.start_pos:
                                st.error("❌ Điểm đích không thể trùng với vị trí bắt đầu!")
                            else:
                                st.session_state.end_pos = (end_row, end_col)
                                st.success(f"✅ Đã đặt điểm đích tại ({end_row}, {end_col})")
                        else:
                            st.error("❌ Điểm đích phải là ô đường thông thường!")
                
                # Nút ngẫu nhiên vị trí bắt đầu và điểm đích
                if st.button("🎲 Ngẫu nhiên vị trí bắt đầu và điểm đích", use_container_width=True):
                    # Tìm tất cả các ô đường thông thường
                    road_cells = []
                    for i in range(st.session_state.map.size):
                        for j in range(st.session_state.map.size):
                            if st.session_state.map.grid[i][j] == 0:
                                road_cells.append((i, j))
                    
                    if len(road_cells) >= 2:
                        # Chọn ngẫu nhiên 2 ô khác nhau
                        selected_indices = np.random.choice(len(road_cells), 2, replace=False)
                        st.session_state.start_pos = road_cells[selected_indices[0]]
                        st.session_state.end_pos = road_cells[selected_indices[1]]
                        st.success(f"✅ Đã ngẫu nhiên đặt vị trí bắt đầu tại {st.session_state.start_pos} và điểm đích tại {st.session_state.end_pos}")
                    else:
                        st.error("❌ Không đủ ô đường thông thường để đặt vị trí bắt đầu và điểm đích!")
            
            # Hiển thị bản đồ với vị trí bắt đầu và điểm đích
            with st.container():
                st.divider()
                st.subheader("🗺️ Bản đồ hiện tại")
                
                if st.session_state.start_pos is None:
                    st.warning("⚠️ Chưa thiết lập vị trí bắt đầu của xe - Hãy thiết lập vị trí bắt đầu để tiếp tục")
                elif st.session_state.end_pos is None:
                    st.warning("⚠️ Chưa thiết lập điểm đích - Hãy thiết lập điểm đích để tiếp tục")
                
                draw_map(st.session_state.map)
        
        with tab2:
            if map_type == "Vẽ thủ công":
                with st.container():
                    st.subheader("✏️ Vẽ bản đồ thủ công")
                    
                    st.info("""
                    Chọn loại ô và nhấp vào bản đồ để thay đổi:
                    - Đường thông thường: Xe có thể đi qua
                    - Trạm thu phí: Xe đi qua phải trả phí
                    - Trạm xăng: Điểm xe có thể dừng để đổ xăng
                    - Vật cản: Ô không thể đi qua
                    """)
                    
                    # Tạo bảng chọn loại ô với icon
                    st.write("Chọn loại ô cần vẽ:")
                    cell_type = st.radio(
                        "",
                        options=[0, 1, 2, -1],
                        format_func=lambda x: {
                            0: "🛣️ Đường thông thường",
                            1: "🚧 Trạm thu phí",
                            2: "⛽ Trạm xăng",
                            -1: "🧱 Vật cản"
                        }[x],
                        horizontal=True
                    )
                    
                    # Tạo bản đồ có thể chỉnh sửa
                    st.write("Nhấp vào ô để thay đổi loại:")
                    
                    # Hiển thị bản đồ hiện tại với chức năng chỉnh sửa
                    edited_df = st.data_editor(
                        pd.DataFrame(st.session_state.map.grid),
                        use_container_width=True,
                        hide_index=False,
                        column_config={i: st.column_config.SelectboxColumn(
                            f"Cột {i}",
                            options=[0, 1, 2, -1],
                            required=True,
                            width="small",
                            help="0: Đường, 1: Thu phí, 2: Xăng, -1: Vật cản"
                        ) for i in range(st.session_state.map.size)}
                    )
                    
                    # Cập nhật bản đồ nếu có thay đổi
                    if not edited_df.equals(pd.DataFrame(st.session_state.map.grid)):
                        st.session_state.map.grid = edited_df.values
                        
                        # Kiểm tra và cập nhật vị trí bắt đầu và điểm đích nếu cần
                        if st.session_state.start_pos:
                            row, col = st.session_state.start_pos
                            if st.session_state.map.grid[row][col] != 0:
                                st.session_state.start_pos = None
                                st.warning("⚠️ Vị trí bắt đầu đã bị xóa do ô không còn là đường thông thường")
                        
                        if st.session_state.end_pos:
                            row, col = st.session_state.end_pos
                            if st.session_state.map.grid[row][col] != 0:
                                st.session_state.end_pos = None
                                st.warning("⚠️ Điểm đích đã bị xóa do ô không còn là đường thông thường")
                    
                    # Thêm nút để làm sạch hoặc đảo ngẫu nhiên
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🧹 Xóa tất cả", use_container_width=True):
                            st.session_state.map.grid = np.zeros((st.session_state.map.size, st.session_state.map.size), dtype=int)
                            st.session_state.start_pos = None
                            st.session_state.end_pos = None
                            st.experimental_rerun()
                    
                    with col2:
                        if st.button("🔄 Đảo ngẫu nhiên", use_container_width=True):
                            rng = np.random.RandomState(42)
                            st.session_state.map.grid = rng.permutation(st.session_state.map.grid.flatten()).reshape(st.session_state.map.size, st.session_state.map.size)
                            st.session_state.start_pos = None
                            st.session_state.end_pos = None
                            st.experimental_rerun()
            else:
                st.info("ℹ️ Để chỉnh sửa bản đồ, hãy chọn chế độ 'Vẽ thủ công' ở menu bên trái")
        
        with tab3:
            with st.container():
                st.subheader("📊 Thống kê bản đồ")
                
                # Thống kê dạng card
                stats = st.session_state.map.get_statistics()
                total_cells = st.session_state.map.size * st.session_state.map.size
                
                # Tạo thống kê với metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    # Hiển thị thống kê dạng bảng
                    data = {
                        "Loại ô": ["Đường thông thường", "Trạm thu phí", "Trạm xăng", "Vật cản"],
                        "Số lượng": [
                            stats['normal_roads'],
                            stats['toll_stations'],
                            stats['gas_stations'],
                            stats['brick_cells']
                        ],
                        "Tỷ lệ": [
                            f"{stats['normal_roads']/total_cells*100:.1f}%",
                            f"{stats['toll_stations']/total_cells*100:.1f}%",
                            f"{stats['gas_stations']/total_cells*100:.1f}%",
                            f"{stats['brick_cells']/total_cells*100:.1f}%"
                        ]
                    }
                    
                    st.table(pd.DataFrame(data))
                
                with col2:
                    # Hiển thị thanh progress bar
                    st.write("##### Tỷ lệ các loại ô:")
                    st.progress(stats['normal_roads']/total_cells, text=f"🛣️ Đường thông thường: {stats['normal_roads']/total_cells*100:.1f}%")
                    st.progress(stats['toll_stations']/total_cells, text=f"🚧 Trạm thu phí: {stats['toll_stations']/total_cells*100:.1f}%")
                    st.progress(stats['gas_stations']/total_cells, text=f"⛽ Trạm xăng: {stats['gas_stations']/total_cells*100:.1f}%")
                    st.progress(stats['brick_cells']/total_cells, text=f"🧱 Vật cản: {stats['brick_cells']/total_cells*100:.1f}%")
                
                # Thông tin kiểm tra
                st.divider()
                st.subheader("🔍 Kiểm tra bản đồ")
                
                # Kiểm tra vị trí bắt đầu và điểm đích
                if st.session_state.start_pos:
                    st.success(f"✅ Vị trí bắt đầu: ({st.session_state.start_pos[0]}, {st.session_state.start_pos[1]})")
                else:
                    st.warning("⚠️ Chưa thiết lập vị trí bắt đầu")
                
                if st.session_state.end_pos:
                    st.success(f"✅ Điểm đích: ({st.session_state.end_pos[0]}, {st.session_state.end_pos[1]})")
                else:
                    st.warning("⚠️ Chưa thiết lập điểm đích")
                
                # Kiểm tra số lượng trạm xăng tối thiểu
                if stats['gas_stations'] < 1:
                    st.error("❌ Bản đồ cần có ít nhất 1 trạm xăng!")
                else:
                    st.success(f"✅ Có {stats['gas_stations']} trạm xăng trên bản đồ")
                
                # Kiểm tra đường đi
                if stats['normal_roads'] < total_cells * 0.3:
                    st.warning(f"⚠️ Tỷ lệ đường đi ({stats['normal_roads']/total_cells*100:.1f}%) quá thấp!")
                else:
                    st.success(f"✅ Tỷ lệ đường đi ({stats['normal_roads']/total_cells*100:.1f}%) hợp lý") 