# config.py

# Đường dẫn tới file model
MODEL_PATH = 'best_plant_classifier_vgg16.h5'

# Danh sách tên lớp model có thể nhận diện (PHẢI KHỚP THỨ TỰ HUẤN LUYỆN)
# Ví dụ: ['Pothos', 'Monstera'] - Bạn cần xác nhận lại!
CLASS_NAMES = ['Pothos', 'Monstera']

CLASS_TO_SCIENTIFIC = {
    'Pothos': 'Epipremnum_aureum', # Sử dụng gạch dưới cho tên thư mục
    'Monstera': 'Monstera_deliciosa'
}

# Ngưỡng tin cậy để coi là chắc chắn (%)
CONFIDENCE_THRESHOLD = 90.0 # Sử dụng dạng phần trăm

# Thư mục lưu dữ liệu người dùng phản hồi
COLLECTED_DATA_DIR = "collected_data"

# --- iNaturalist Config ---
INAT_API_BASE_URL = "https://api.inaturalist.org/v1"

# Dictionary các loài gợi ý ban đầu (có thể không cần dùng nhiều nếu có autocomplete)
# Dùng để tham khảo hoặc gợi ý nhanh nếu muốn
SUGGESTED_SPECIES_VN = {
    "Trầu Bà Vàng (Pothos)": "Epipremnum aureum",
    "Monstera (Trầu Bà Lá Xẻ Lớn)": "Monstera deliciosa",
    "Kim Tiền (ZZ Plant)": "Zamioculcas zamiifolia",
    "Lan Ý (Peace Lily)": "Spathiphyllum wallisii",
    "Hồng Môn": "Anthurium andraeanum",
    "Vạn Niên Thanh": "Dieffenbachia seguine",
    "Trầu Bà Lá Xẻ (Syngonium)": "Syngonium podophyllum",
    "Lưỡi Hổ": "Dracaena trifasciata", # Tên mới của Sansevieria
    "Bàng Singapore": "Ficus lyrata",
    "Đuôi Công (Calathea/Goeppertia)": "Goeppertia ornata", # Ví dụ 1 loài
    "Ráy Voi (Alocasia)": "Alocasia macrorrhizos",
    # Thêm các loài khác phổ biến ở VN...
}