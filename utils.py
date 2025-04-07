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

import firebase_admin
from firebase_admin import credentials, storage, firestore

import json

# Import base URL từ config
from config import INAT_API_BASE_URL, COLLECTED_DATA_DIR

# --- Firebase Initialization ---
SERVICE_ACCOUNT_KEY_PATH = 'plantidentify-ca6f7-firebase-adminsdk-fbsvc-25fb51dcb6.json'
CORRECT_BUCKET_NAME = "plantidentify-ca6f7.firebasestorage.app"

@st.cache_resource
def initialize_firebase():
    """Khởi tạo Firebase Admin SDK một cách an toàn."""
    if firebase_admin._apps:
        print("Firebase app already initialized (checked at start).")
        return True

    firebase_creds_dict = None # Biến để lưu dictionary cuối cùng
    try:
        # --- LUÔN THỬ ĐỌC TỪ SECRETS TRƯỚC ---
        if hasattr(st, 'secrets'):
            print("Attempting to read from st.secrets...")
            retrieved_secret_value = st.secrets.get("FIREBASE_SERVICE_ACCOUNT")
            if retrieved_secret_value:
                print(f"DEBUG: Type of retrieved secret: {type(retrieved_secret_value)}")
                # <<< THỬ PARSE JSON THỦ CÔNG >>>
                if isinstance(retrieved_secret_value, str):
                    print("DEBUG: Secret value is a string, attempting JSON parse...")
                    try:
                        firebase_creds_dict = json.loads(retrieved_secret_value)
                        print("DEBUG: Successfully parsed JSON string from secret.")
                    except json.JSONDecodeError as json_e:
                        print(f"Error: Could not parse secret value as JSON: {json_e}")
                        st.error(f"Lỗi định dạng JSON trong Secret 'FIREBASE_SERVICE_ACCOUNT': {json_e}")
                        # Không gán gì cho firebase_creds_dict
                elif isinstance(retrieved_secret_value, dict):
                    # Nếu Streamlit đã tự parse thành công (trường hợp lý tưởng)
                    print("DEBUG: Secret value is already a dict.")
                    firebase_creds_dict = retrieved_secret_value
                else:
                     print(f"Warning: Unexpected type for secret value: {type(retrieved_secret_value)}")

            else:
                print("DEBUG: FIREBASE_SERVICE_ACCOUNT secret not found or empty.")
        else:
            print("st.secrets not available (running locally?).")

    except Exception as secrets_e:
        print(f"Error accessing st.secrets: {secrets_e}")
        # Tiếp tục thử file local nếu lỗi secrets

    # --- ƯU TIÊN KHỞI TẠO TỪ DICTIONARY ĐÃ PARSE (TỪ SECRETS) ---
    if firebase_creds_dict: # Chỉ cần kiểm tra khác None vì đã parse/gán ở trên
        print("Initializing Firebase using credentials obtained from Secrets...")
        try:
            # Truyền dictionary đã parse vào Certificate
            cred_obj = credentials.Certificate(firebase_creds_dict)
            project_id = cred_obj.project_id if hasattr(cred_obj, 'project_id') else firebase_creds_dict.get('project_id')
            if not project_id:
                 st.error("Không thể xác định Project ID từ Secrets.")
                 print("Error: Could not determine Project ID from Secrets.")
                 return False

            firebase_admin.initialize_app(cred_obj, {
                'storageBucket': CORRECT_BUCKET_NAME
            })
            print("Firebase initialized from Secrets.")
            return True
        # ... (Các khối except ValueError, Exception cho nhánh secrets như cũ) ...
        except ValueError as ve:
            if "The default Firebase app already exists" in str(ve): return True
            else: print(f"ValueError during init from Secrets dict: {ve}"); return False
        except Exception as e_init_secrets:
            print(f"Error during init from Secrets dict: {e_init_secrets}"); return False


    # --- KHỞI TẠO TỪ FILE LOCAL NẾU SECRETS KHÔNG DÙNG ĐƯỢC/LỖI ---
    elif os.path.exists(SERVICE_ACCOUNT_KEY_FILENAME): # Đổi biến SERVICE_ACCOUNT_KEY_PATH thành SERVICE_ACCOUNT_KEY_FILENAME nếu cần
        print(f"Initializing Firebase using local key file: {SERVICE_ACCOUNT_KEY_FILENAME}...")
        try:
            cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_FILENAME)
            # project_id = cred.project_id # Không cần lấy project_id nữa nếu dùng tên bucket cứng
            firebase_admin.initialize_app(cred, {
                'storageBucket': CORRECT_BUCKET_NAME
            })
            print("Firebase initialized from local file.")
            return True
        # ... (Các khối except ValueError, Exception cho nhánh local như cũ) ...
        except ValueError as ve:
             if "The default Firebase app already exists" in str(ve): return True
             else: print(f"ValueError during init from local file: {ve}"); return False
        except Exception as e_init_local:
            print(f"Error during init from local file: {e_init_local}"); return False

    # --- KHÔNG TÌM THẤY CẢ SECRETS HỢP LỆ VÀ FILE LOCAL ---
    else:
        st.error("Không tìm thấy thông tin xác thực Firebase hợp lệ.")
        print("Firebase credentials not found (invalid Secrets and no local file).")
        st.info(f"Đảm bảo Secret 'FIREBASE_SERVICE_ACCOUNT' có giá trị JSON đúng hoặc file key '{SERVICE_ACCOUNT_KEY_FILENAME}' tồn tại khi chạy local.")
        return False

# Gọi hàm khởi tạo ngay khi load utils.py (nhờ cache nên chỉ chạy 1 lần)
firebase_initialized = initialize_firebase()
print(f"UTILS: Firebase initialized status after attempt: {firebase_initialized}")

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
def save_feedback_image(image_bytes, original_filename, label, base_dir=COLLECTED_DATA_DIR): # base_dir giờ là tiền tố trên Storage
    """Lưu ảnh phản hồi (dạng bytes) lên Firebase Cloud Storage."""
    print(f"SAVE_FEEDBACK: Called. Firebase initialized status: {firebase_initialized}")
    if not firebase_initialized:
        st.error("Firebase chưa được khởi tạo, không thể lưu ảnh.")
        print("SAVE_FEEDBACK: Firebase not initialized, aborting save.")
        return False, None

    try:
        # Làm sạch tên label để tạo đường dẫn trên Storage
        safe_label = re.sub(r'[^\w\s-]', '', label).strip().replace(' ', '_')
        safe_label = safe_label[:100]
        if not safe_label: safe_label = "unknown_label"

        # Tạo timestamp và tên file duy nhất
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
        # Cố gắng lấy đuôi file từ tên gốc
        file_extension = ".jpg" # Mặc định
        if original_filename and isinstance(original_filename, str):
             try:
                 _, ext = os.path.splitext(original_filename)
                 if ext: file_extension = ext.lower()
             except Exception: pass # Bỏ qua nếu lỗi lấy đuôi file

        # Tạo đường dẫn đầy đủ trên Firebase Storage
        # Ví dụ: collected_data/Epipremnum_aureum/20230407_193005123456.jpg
        destination_blob_name = f"{base_dir}/{safe_label}/{timestamp}{file_extension}"

        # Lấy bucket từ Firebase Storage
        bucket = storage.bucket() # Lấy bucket mặc định đã cấu hình khi initialize

        # Tạo blob (đối tượng file) trên Storage
        blob = bucket.blob(destination_blob_name)

        # Xác định content type dựa trên đuôi file (quan trọng để trình duyệt hiển thị đúng)
        content_type = 'image/jpeg' # Mặc định
        if file_extension == '.png':
            content_type = 'image/png'
        elif file_extension == '.gif':
            content_type = 'image/gif'
        # Thêm các loại khác nếu cần

        # Upload dữ liệu ảnh (dạng bytes)
        print(f"UTILS: Uploading to Firebase Storage: {destination_blob_name} (Content-Type: {content_type})")
        blob.upload_from_string(
            image_bytes,
            content_type=content_type
        )

        print(f"UTILS: Successfully uploaded to {destination_blob_name}")

        # --- (TÙY CHỌN) Lưu metadata vào Firestore ---
        try:
            db = firestore.client()
            doc_ref = db.collection(u'plant_feedback').document(f'{timestamp}') # Dùng timestamp làm ID document
            doc_ref.set({
                u'label': label, # Lưu cả nhãn gốc người dùng nhập/chọn
                u'storage_path': destination_blob_name,
                u'original_filename': original_filename,
                u'timestamp': firestore.SERVER_TIMESTAMP # Tự động lấy giờ server
                # Thêm các trường khác nếu muốn (ví dụ: dự đoán ban đầu, confidence...)
            })
            print(f"UTILS: Metadata saved to Firestore collection 'plant_feedback', doc ID: {timestamp}")
        except Exception as fs_e:
            print(f"UTILS: Error saving metadata to Firestore: {fs_e}")
            st.warning("Lưu ảnh thành công nhưng không thể lưu thông tin vào CSDL.")
            # Không coi đây là lỗi nghiêm trọng làm hỏng việc upload ảnh

        return True, safe_label # Trả về thành công và tên thư mục (label) đã dùng

    except Exception as e:
        print(f"Error uploading image to Firebase Storage for label '{label}' (safe: '{safe_label}'): {e}")
        st.error(f"Lỗi khi tải ảnh lên bộ nhớ Cloud: {e}")
        return False, None