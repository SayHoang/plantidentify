# utils.py

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
from PIL import Image
import io
import requests
import os
from datetime import datetime
import re # Thêm thư viện regular expression để làm sạch tên file/thư mục

# Import base URL từ config
from config import INAT_API_BASE_URL

# --- Model Loading ---
@st.cache_resource
def load_keras_model(model_path):
    """Tải mô hình Keras từ đường dẫn."""
    try:
        model = load_model(model_path, compile=False)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Lỗi nghiêm trọng khi tải mô hình tại đường dẫn '{model_path}': {e}")
        print(f"Error loading model from {model_path}: {e}")
        return None

# --- Image Processing ---
def preprocess_image(image_data):
    # ... (Giữ nguyên hàm này như trước) ...
    try:
        img = Image.open(io.BytesIO(image_data))
        if img.format == 'GIF':
             img = img.convert('RGB')
        img = img.convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        return preprocess_input(img_array_expanded)
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        st.error(f"Không thể xử lý ảnh này. Vui lòng thử ảnh khác. Lỗi: {e}")
        return None


# --- iNaturalist API Interaction ---
@st.cache_data(ttl=3600)
def get_taxon_id(scientific_name):
    print(f"UTILS: Attempting to get Taxon ID for: '{scientific_name}'") # DEBUG
    if not scientific_name: return None
    try:
        api_url = f"{INAT_API_BASE_URL}/taxa"
        params = {'q': scientific_name, 'is_active': 'true', 'rank': 'species'}
        print(f"UTILS: Calling Taxa API: {api_url} with params: {params}") # DEBUG
        response = requests.get(api_url, params=params, timeout=10)
        print(f"UTILS: Taxa API status code: {response.status_code}") # DEBUG
        response.raise_for_status()
        data = response.json()
        # print(f"UTILS: Taxa API response data: {data}") # DEBUG - có thể rất dài
        if data.get('results') and data['total_results'] > 0:
            taxon_id = data['results'][0]['id']
            print(f"UTILS: Found Taxon ID: {taxon_id}") # DEBUG
            return taxon_id
        else:
            print(f"UTILS: No Taxon ID found for: {scientific_name}") # DEBUG
            return None
    except requests.exceptions.RequestException as e:
        print(f"UTILS: API Error finding Taxon ID for {scientific_name}: {e}") # DEBUG
        st.warning(f"Không thể kết nối đến iNaturalist để tìm ID cho {scientific_name}.")
        return None
    except Exception as e:
        print(f"UTILS: Unknown error finding Taxon ID for {scientific_name}: {e}") # DEBUG
        return None


@st.cache_data(ttl=3600)
def get_inat_image_urls(taxon_id, count=10):
    print(f"UTILS: Attempting to get image URLs for Taxon ID: {taxon_id}") # DEBUG
    if not taxon_id: return []
    image_urls = []
    try:
        api_url = f"{INAT_API_BASE_URL}/observations"
        params = {
            'taxon_id': taxon_id, 'photos': 'true', 'quality_grade': 'research',
            'per_page': count, 'order': 'desc', 'order_by': 'votes'
        }
        print(f"UTILS: Calling Observations API: {api_url} with params: {params}") # DEBUG
        response = requests.get(api_url, params=params, timeout=15) # Tăng nhẹ timeout nếu cần
        print(f"UTILS: Observations API status code: {response.status_code}") # DEBUG
        response.raise_for_status()
        data = response.json()
        # print(f"UTILS: Observations API response data: {data}") # DEBUG - có thể rất dài

        if data.get('results') and data['total_results'] > 0:
            print(f"UTILS: Found {len(data['results'])} observations with photos.") # DEBUG
            for obs in data['results']:
                if obs.get('photos'):
                    photo_url = obs['photos'][0].get('url')
                    if photo_url:
                        medium_url = photo_url.replace('square', 'medium')
                        print(f"UTILS: Adding image URL: {medium_url}") # DEBUG
                        image_urls.append(medium_url)
                    if len(image_urls) >= count:
                        break
        else:
             print(f"UTILS: No 'results' or 'total_results' > 0 in API response for taxon_id: {taxon_id}") # DEBUG

        print(f"UTILS: Returning {len(image_urls)} image URLs.") # DEBUG
        return image_urls
    except requests.exceptions.RequestException as e:
        print(f"UTILS: API Error getting observation photos for taxon_id {taxon_id}: {e}") # DEBUG
        st.warning("Không thể kết nối đến iNaturalist để lấy ảnh tham khảo.")
        return []
    except Exception as e:
        print(f"UTILS: Unknown error getting observation photos for taxon_id {taxon_id}: {e}") # DEBUG
        return []

# *** HÀM MỚI CHO AUTOCOMPLETE ***
@st.cache_data(ttl=600) # Cache ngắn hơn cho autocomplete (10 phút)
def search_taxa_autocomplete(query):
    """Tìm kiếm gợi ý loài trên iNaturalist."""
    if not query or len(query) < 3: # Chỉ tìm khi có ít nhất 3 ký tự
        return []
    suggestions = []
    try:
        api_url = f"{INAT_API_BASE_URL}/taxa/autocomplete"
        params = {'q': query, 'is_active': 'true', 'rank': 'species,genus,family'} # Mở rộng rank
        response = requests.get(api_url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data.get('results') and data['total_results'] > 0:
            for result in data['results']:
                taxon_id = result.get('id')
                scientific_name = result.get('name') # Tên khoa học
                display_name = result.get('preferred_common_name') or scientific_name # Ưu tiên tên thường gọi
                rank = result.get('rank')
                # Tạo một chuỗi hiển thị rõ ràng
                suggestion_display = f"{display_name} ({scientific_name}) - Rank: {rank}"
                suggestions.append({
                    "id": taxon_id,
                    "scientific_name": scientific_name,
                    "display_name": display_name,
                    "rank": rank,
                    "formatted_display": suggestion_display # Chuỗi để hiển thị trong selectbox/radio
                })
        return suggestions
    except requests.exceptions.RequestException as e:
        print(f"API Error during taxa autocomplete for query '{query}': {e}")
        # Không hiện lỗi trực tiếp lên UI để tránh làm phiền
        return []
    except Exception as e:
        print(f"Unknown error during taxa autocomplete for query '{query}': {e}")
        return []

# --- File Saving for Feedback ---
def save_feedback_image(image_bytes, original_filename, label, base_dir):
    """Lưu ảnh được phản hồi (dạng bytes) vào thư mục tương ứng."""
    try:
        # --- THÊM KIỂM TRA VÀ MẶC ĐỊNH CHO original_filename ---
        if original_filename is None:
            original_filename = "unknown_original_name.jpg" # Hoặc một tên mặc định khác
            print("Warning: original_filename was None, using default.")
        # -------------------------------------------------------

        # Làm sạch tên label
        safe_label = re.sub(r'[^\w\s-]', '', label).strip().replace(' ', '_')
        safe_label = safe_label[:100]
        if not safe_label: safe_label = "unknown_label"
        label_dir = os.path.join(base_dir, safe_label)
        os.makedirs(label_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
        # Làm sạch tên file gốc (được truyền vào và đã kiểm tra None)
        safe_original_filename = re.sub(r'[^\w\.-]', '_', original_filename)
        safe_original_filename = safe_original_filename[:50]
        filename = f"{timestamp}_{safe_original_filename}"
        save_path = os.path.join(label_dir, filename)

        # Ghi dữ liệu bytes trực tiếp
        with open(save_path, "wb") as f:
            f.write(image_bytes)
        return True, safe_label
    except TypeError as te: # Bắt cụ thể lỗi TypeError
        print(f"TypeError in save_feedback_image (label: '{label}'): {te}")
        print(f"Image bytes type: {type(image_bytes)}, Original filename type: {type(original_filename)}")
        st.error(f"Lỗi kiểu dữ liệu khi lưu ảnh: {te}")
        return False, None
    except Exception as e:
        print(f"Error saving feedback image to label '{label}' (safe: '{safe_label}'): {e}")
        st.error(f"Lỗi khi lưu ảnh phản hồi: {e}")
        return False, None