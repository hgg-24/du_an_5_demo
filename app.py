import streamlit as st
import plotly.graph_objects as go
import numpy as np
import requests
from rdkit import Chem
from rdkit.Chem import Draw
from stmol import showmol
import py3Dmol
from PIL import Image
import io

# ==========================================
# 1. CẤU HÌNH TRANG & GIAO DIỆN (UI/UX)
# ==========================================
st.set_page_config(page_title="ED-ODYSSEY | Chem-Lab", layout="wide")

def inject_custom_css():
    st.markdown(r"""
    <style>
        :root {
            --primary-color: #007AFF;
        }
        .stApp {
            background-color: #F4F7FA;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #FFFFFF !important;
            border-radius: 16px !important;
            box-shadow: 0px 4px 20px rgba(0, 0, 0, 0.05) !important;
            border: 1px solid #E5E5EA !important;
            padding: 1.2rem;
        }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# ==========================================
# 2. HÀM XỬ LÝ HÓA TIN HỌC (DATA ENGINES)
# ==========================================
@st.cache_data
def get_smiles_from_formula(formula):
    """Sử dụng PubChem API để lấy SMILES từ công thức phân tử."""
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastformula/{formula}/property/CanonicalSMILES/JSON"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            # Bổ sung dòng kiểm tra an toàn: Chỉ lấy dữ liệu nếu tồn tại key 'PropertyTable'
            if 'PropertyTable' in data and 'Properties' in data['PropertyTable']:
                return data['PropertyTable']['Properties'][0]['CanonicalSMILES']
                
        # Nếu không có data hoặc lỗi, trả về None để hệ thống tự hiện cảnh báo vàng
        return None
        
    except Exception:
        return None

def draw_bohr_model(atomic_number):
    """Vẽ mô hình Bohr 2D bằng Plotly."""
    # Quy tắc phân bố electron cơ bản (Klechkowski giản lược)
    shells = [2, 8, 18, 32]
    electrons_left = atomic_number
    config = []
    
    for capacity in shells:
        if electrons_left <= 0: break
        if electrons_left >= capacity:
            config.append(capacity)
            electrons_left -= capacity
        else:
            config.append(electrons_left)
            electrons_left = 0

    fig = go.Figure()
    
    # Vẽ hạt nhân
    fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers+text', 
                             marker=dict(size=40, color='#FF3B30'),
                             text=[f"+{atomic_number}"], textfont=dict(color='white', size=14),
                             name="Hạt nhân"))
    
    # Vẽ các lớp vỏ và electron
    for i, num_e in enumerate(config):
        radius = i + 1
        # Vẽ vòng tròn quỹ đạo
        theta = np.linspace(0, 2*np.pi, 100)
        fig.add_trace(go.Scatter(x=radius*np.cos(theta), y=radius*np.sin(theta), 
                                 mode='lines', line=dict(color='gray', dash='dash'), hoverinfo='skip', showlegend=False))
        
        # Phân bố electron trên quỹ đạo
        angles = np.linspace(0, 2*np.pi, num_e, endpoint=False)
        e_x = radius * np.cos(angles)
        e_y = radius * np.sin(angles)
        fig.add_trace(go.Scatter(x=e_x, y=e_y, mode='markers', 
                                 marker=dict(size=12, color='#007AFF'), name=f"Lớp {i+1} ({num_e}e)"))

    fig.update_layout(width=400, height=400, template="plotly_white", 
                      xaxis=dict(visible=False, range=[-5, 5]), yaxis=dict(visible=False, range=[-5, 5]),
                      margin=dict(l=10, r=10, t=10, b=10))
    return fig, config

# ==========================================
# 3. GIAO DIỆN CHÍNH (MAIN APP)
# ==========================================
st.title("🧪 Chem-Lab: Molecular & Thermo Engine")
st.markdown("Hệ thống trực quan hóa cấu trúc nguyên tử và phân tử đa chiều của ED-ODYSSEY.")

# Thanh tìm kiếm thông minh với Unique Key. Đã thêm .strip() để tự động xóa khoảng trắng thừa do người dùng lỡ tay bấm.
search_query = st.text_input("🔍 Nhập Ký hiệu nguyên tố (VD: C, Fe) hoặc Công thức phân tử (VD: H2O, C6H12O6):", 
                             value="H2O", key="chem_search_input").strip()

# Tạo 3 Tabs (3-Panel UI) để tối ưu không gian hiển thị
tab1, tab2, tab3 = st.tabs(["⚛️ Cấu tạo nguyên tử", "🔗 Cấu trúc 2D (Lewis)", "🌐 Mô hình 3D (WebGL)"])

# Khu vực giữ chỗ (Placeholders) chống lỗi DOM removeChild
bohr_placeholder = tab1.empty()
lewis_placeholder = tab2.empty()
model3d_placeholder = tab3.empty()
thermo_placeholder = st.empty() # Dành cho tính năng Enthalpy

if search_query:
    try:
        # Xử lý nếu người dùng nhập nguyên tố (1-2 ký tự, bắt đầu bằng chữ hoa)
        is_element = len(search_query) <= 2 and search_query[0].isupper()
        
        smiles = None
        if is_element:
            # Tạm lập bản đồ tra cứu nhanh nguyên tử khối & Z (Trong thực tế dùng mendeleev)
            # Demo với một vài nguyên tố phổ biến
            elements_db = {"H": 1, "C": 6, "N": 7, "O": 8, "Na": 11, "Fe": 26}
            atomic_num = elements_db.get(search_query, 6) # Mặc định là Carbon nếu ko tìm thấy
            smiles = search_query # RDKit hiểu ký hiệu nguyên tố
            
            with bohr_placeholder.container():
                st.subheader(f"Mô hình nguyên tử: {search_query}")
                fig_bohr, e_config = draw_bohr_model(atomic_num)
                st.plotly_chart(fig_bohr, use_container_width=False, key="bohr_chart")
                st.latex(rf"\text{{Cấu hình Electron: }} " + " \\ ".join([f"{i+1}s^... " for i in range(len(e_config))]))
                st.caption(f"Tổng số electron: {sum(e_config)}")
        else:
            # Tra cứu phân tử qua PubChem
            smiles = get_smiles_from_formula(search_query)
            bohr_placeholder.info("Mô hình nguyên tử chỉ áp dụng cho đơn chất nguyên tố. Vui lòng xem cấu trúc 2D và 3D cho phân tử.")

        if smiles:
            # --- PANEL 2: RDKit 2D Lewis Structure ---
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                img = Draw.MolToImage(mol, size=(400, 400), fitImage=True)
                with lewis_placeholder.container():
                    st.image(img, caption=f"Cấu trúc 2D của {search_query}", use_container_width=False)
            
            # --- PANEL 3: py3Dmol Interactive 3D ---
            with model3d_placeholder.container():
                st.subheader("Trình diễn WebGL 3D")
                st.markdown("*Lưu ý: Dùng chuột xoay 360° và cuộn chuột để phóng to/thu nhỏ.*")
                
                # Khởi tạo view 3D
                view = py3Dmol.view(query=f'smiles:{smiles}', width=600, height=400)
                view.setStyle({'stick': {'radius': 0.15}, 'sphere': {'scale': 0.3}})
                view.zoomTo()
                
                # Render 3D component bằng stmol
                showmol(view, height=400, width=600)
                
            # --- BONUS: TÍNH NĂNG NHIỆT HÓA HỌC ---
            with thermo_placeholder.container(border=True):
                st.subheader("🔥 Động cơ Nhiệt hóa học (Enthalpy Engine)")
                st.latex(rf"\Delta_r H_{{298}}^0 = \sum \Delta_f H_{{298}}^0 (\text{{Sản phẩm}}) - \sum \Delta_f H_{{298}}^0 (\text{{Chất tham gia}})")
                st.info("Tính năng tính toán biến thiên Enthalpy tự động đang được kết nối với cơ sở dữ liệu nhiệt động học...")

        else:
            st.warning(f"Chúng tôi đang cập nhật thêm dữ liệu cho chất '{search_query}'. Vui lòng thử lại với các chất phổ biến như H2O, CO2, CH4...")

    except Exception as e:
        st.error("Có lỗi xảy ra trong quá trình xử lý đồ họa Hóa học.")
        st.caption(f"Mã lỗi hệ thống: {e}")

# ==========================================
# 4. FOOTER
# ==========================================
st.markdown("---")
st.caption("© 2026 ED-ODYSSEY Hub | Khởi chạy bằng RDKit & py3Dmol Engine")
