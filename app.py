# app.py

import streamlit as st
import numpy as np
import os
import uuid
# import requests # Không cần import trực tiếp ở đây nữa nếu st.image xử lý URL

# Import các hàm và hằng số
from config import MODEL_PATH, CLASS_NAMES, CONFIDENCE_THRESHOLD, COLLECTED_DATA_DIR
from utils import (load_keras_model, preprocess_image, search_taxa_autocomplete,
                   get_inat_image_urls, save_feedback_image)

# --- Cấu hình trang ---
st.set_page_config(
    page_title="Nhận diện Cây Kiểng Lá VN",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Khởi tạo Session State ---
default_states = {
    'file_identifier': None,
    'widget_key_prefix': None,
    'image_data': None,
    'prediction_done': False,
    'predicted_class': None,
    'confidence': 0.0,
    'user_feedback': None,
    'inat_search_term': "",
    'inat_suggestions': [],
    'selected_inat_suggestion': None,
    'inat_image_urls': [],
    'final_label_confirmed': None,
    'image_saved': False,
    'last_search_term': "" # Thêm state để tránh gọi API liên tục
}
if 'original_filename' not in st.session_state: st.session_state.original_filename = None
for key, default_value in default_states.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# --- Tải Model ---
model = load_keras_model(MODEL_PATH)

# --- Sidebar ---
st.sidebar.header("Tải ảnh lên")
uploaded_file = st.sidebar.file_uploader(
    "Chọn một file ảnh (JPG, PNG, GIF)",
    type=["jpg", "jpeg", "png", "gif"],
    key="file_uploader" # Key cố định
)

# --- Logic chính ---
st.title("🌿 Nhận diện Cây Kiểng Lá")

# Xử lý khi có file mới được tải lên
if uploaded_file is not None:
    current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    # Nếu là file mới, reset toàn bộ trạng thái và tạo key mới
    if st.session_state.file_identifier != current_file_id:
        print(f"New file uploaded: {uploaded_file.name}")
        st.session_state.file_identifier = current_file_id
        st.session_state.widget_key_prefix = f"file_{uuid.uuid4().hex[:10]}"
        st.session_state.image_data = uploaded_file.getvalue()
        # Reset tất cả các trạng thái liên quan đến xử lý file cũ
        for key in default_states:
            if key not in ['file_identifier', 'widget_key_prefix', 'image_data']: # Giữ lại 3 cái này
                st.session_state[key] = default_states[key]
        # Đặt lại last_search_term cho key mới
        st.session_state[f"last_search_{st.session_state.widget_key_prefix}"] = ""


else:
    # Reset nếu không có file
    if st.session_state.file_identifier is not None:
        print("No file uploaded, resetting state.")
        for key in default_states:
            st.session_state[key] = default_states[key]
    st.info('⬆️ Hãy tải lên một hình ảnh ở thanh bên trái để bắt đầu!')


# --- Hiển thị và xử lý khi có file và model đã tải ---
if st.session_state.image_data is not None and model is not None:
    key_prefix = st.session_state.widget_key_prefix # Dùng key prefix đã lưu

    # Hiển thị ảnh gốc
    st.image(st.session_state.image_data, caption='Ảnh bạn đã tải lên.', use_container_width=True)

    # --- Nút Phân loại ---
    if not st.session_state.prediction_done:
        if st.button('Phân loại cây này!', key=f"classify_{key_prefix}"):
            with st.spinner('Đang phân tích hình ảnh...'):
                processed_image = preprocess_image(st.session_state.image_data)
                if processed_image is not None:
                    try:
                        prediction = model.predict(processed_image)
                        pred_index = np.argmax(prediction[0])
                        pred_conf = np.max(prediction[0]) * 100

                        st.session_state.confidence = pred_conf
                        if 0 <= pred_index < len(CLASS_NAMES):
                            st.session_state.predicted_class = CLASS_NAMES[pred_index]
                        else:
                            st.session_state.predicted_class = "Lớp không xác định"
                        st.session_state.prediction_done = True
                        st.rerun() # Chạy lại để hiển thị kết quả
                    except Exception as e:
                        st.error(f"Lỗi trong quá trình dự đoán: {e}")
                        print(f"Prediction error: {e}")
                else:
                    # Lỗi đã được hiển thị trong preprocess_image
                    pass

    # --- Khối chính xử lý SAU KHI ĐÃ PHÂN LOẠI và CHƯA LƯU ---
    if st.session_state.prediction_done and not st.session_state.image_saved:
        st.markdown("---") # Phân cách với ảnh
        pred_class = st.session_state.predicted_class
        pred_conf = st.session_state.confidence

        # --- Định nghĩa biến điều kiện ---
        is_confident_prediction = pred_conf >= CONFIDENCE_THRESHOLD and pred_class in CLASS_NAMES
        is_known_class_prediction = pred_class in CLASS_NAMES

        # Hiển thị kết quả dự đoán
        if is_confident_prediction:
            st.success(f"**Kết quả: Có vẻ là cây {pred_class}!** (Độ chắc chắn: {pred_conf:.1f}%)")
        elif is_known_class_prediction:
             st.warning(f"**Hmm, không chắc chắn lắm.** Dự đoán gần nhất: **{pred_class}** (Độ chắc chắn: {pred_conf:.1f}%)")
        else:
            st.error("**Không thể nhận diện chắc chắn.**")

        # --- Logic Hỏi Feedback Mới ---
        st.markdown("---") # Phân cách trước câu hỏi feedback
        if is_confident_prediction:
            st.write(f"**Hệ thống dự đoán là {pred_class}. Thông tin này có chính xác không?**")
            feedback_cols = st.columns(2)
            with feedback_cols[0]:
                if st.button("✅ Đúng rồi", key=f"feedback_correct_{key_prefix}"):
                    st.session_state.user_feedback = 'Correct_Confident'
                    st.rerun()
            with feedback_cols[1]:
                 if st.button("❌ Sai rồi", key=f"feedback_incorrect_{key_prefix}"):
                    st.session_state.user_feedback = 'Incorrect_Confident'
                    # Reset tìm kiếm khi bấm sai
                    st.session_state.inat_search_term = ""
                    st.session_state.selected_inat_suggestion = None
                    st.session_state.inat_image_urls = []
                    st.rerun()
        else: # Trường hợp không chắc chắn hoặc lớp lạ
            st.write(f"**Hệ thống không chắc chắn về ảnh này. Bạn có thể giúp xác định không?**")
            feedback_cols = st.columns(2)
            with feedback_cols[0]:
                if is_known_class_prediction: # Chỉ hiện nút này nếu biết lớp dự đoán gần nhất
                    if st.button(f"✅ Đúng, đó là {pred_class}", key=f"feedback_confirm_unsure_{key_prefix}"):
                        st.session_state.user_feedback = 'Confirmed_Unsure'
                        st.session_state.final_label_confirmed = pred_class # Xác nhận nhãn dự đoán ban đầu là đúng
                        st.rerun()
                # else: Cột này trống nếu không biết lớp
            with feedback_cols[1]:
                 if st.button("🔍 Tìm loại cây khác", key=f"feedback_search_unsure_{key_prefix}"):
                    st.session_state.user_feedback = 'Search_Unsure'
                    # Reset tìm kiếm
                    st.session_state.inat_search_term = ""
                    st.session_state.selected_inat_suggestion = None
                    st.session_state.inat_image_urls = []
                    st.rerun()

        # --- Xử lý dựa trên Feedback ---
        # Chỉ hiển thị các phần này nếu user_feedback đã được đặt
        if st.session_state.user_feedback == 'Correct_Confident':
            st.success("🎉 Cảm ơn bạn đã xác nhận!")
            # Tùy chọn: Lưu ảnh này với nhãn đúng vào collected_data
            # if st.button("Lưu ảnh này làm dữ liệu tốt", key=f"save_correct_{key_prefix}"):
            #    saved_ok, saved_label_dir = save_feedback_image(st.session_state.image_data, pred_class, COLLECTED_DATA_DIR)
            #    if saved_ok:
            #        st.info(f"Đã lưu ảnh vào thư mục '{saved_label_dir}'.")
            #        st.session_state.image_saved = True # Đánh dấu đã lưu
            #        st.rerun()

        elif st.session_state.user_feedback == 'Confirmed_Unsure':
            st.success(f"🎉 Cảm ơn bạn đã xác nhận là **{st.session_state.final_label_confirmed}**!")
            # Lưu ảnh với nhãn đã xác nhận
            if not st.session_state.image_saved: # Chỉ lưu 1 lần
                 # Sử dụng image_data từ session state để lưu
                 saved_ok, saved_label_dir = save_feedback_image(
                     st.session_state.image_data,             # Dữ liệu ảnh
                     st.session_state.original_filename,    # Tên file gốc
                     st.session_state.final_label_confirmed, # Nhãn đã xác nhận
                     COLLECTED_DATA_DIR
                 )
                 if saved_ok:
                     st.info(f"Đã lưu ảnh vào thư mục '{saved_label_dir}' để cải thiện độ chắc chắn cho model sau này.")
                     st.session_state.image_saved = True
                     st.rerun() # Rerun để hiển thị trạng thái đã lưu

        elif st.session_state.user_feedback in ['Incorrect_Confident', 'Search_Unsure']:
            # Hiển thị giao diện tìm kiếm
            st.markdown("---")
            st.subheader("Tìm và xác nhận loài cây:")
            st.write("Hãy thử tìm tên cây bạn nghĩ đến:")

            # Ô tìm kiếm
            search_term = st.text_input(
                "Nhập tên cây (tên khoa học bằng tiếng Anh)",
                value=st.session_state.inat_search_term,
                key=f"search_input_{key_prefix}"
            )
            # Cập nhật state search term nếu có thay đổi
            if search_term != st.session_state.inat_search_term:
                st.session_state.inat_search_term = search_term
                # Reset gợi ý khi gõ mới để tránh hiển thị gợi ý cũ
                st.session_state.inat_suggestions = []
                # Không cần rerun ở đây, rerun sẽ xảy ra khi kiểm tra độ dài

            # Gọi API autocomplete nếu có từ khóa mới và đủ dài
            last_search_key = f"last_search_{key_prefix}"
            if search_term != st.session_state.get(last_search_key, ""):
                 if len(search_term) >= 3:
                     with st.spinner("Đang tìm gợi ý..."):
                          st.session_state.inat_suggestions = search_taxa_autocomplete(search_term)
                 else:
                     # Xóa gợi ý nếu từ khóa quá ngắn
                     if len(search_term) < 3 and st.session_state.inat_suggestions:
                         st.session_state.inat_suggestions = []
                 st.session_state[last_search_key] = search_term
                 # Chỉ rerun nếu thực sự có thay đổi gợi ý hoặc cần xóa gợi ý
                 st.rerun()

            # Hiển thị gợi ý dạng nút bấm
            if st.session_state.inat_suggestions:
                st.write("Gợi ý:")
                num_suggestions = len(st.session_state.inat_suggestions)
                cols_per_row = 3
                num_rows = (num_suggestions + cols_per_row - 1) // cols_per_row

                suggestion_list = st.session_state.inat_suggestions
                idx = 0
                for r in range(num_rows):
                    cols = st.columns(cols_per_row)
                    for c in range(cols_per_row):
                        if idx < num_suggestions:
                            suggestion = suggestion_list[idx]
                            button_label = suggestion['formatted_display']
                            # Giới hạn độ dài nhãn nút nếu quá dài
                            max_label_len = 35
                            if len(button_label) > max_label_len:
                                button_label = button_label[:max_label_len-3] + "..."

                            if cols[c].button(button_label, key=f"suggestion_button_{suggestion['id']}_{key_prefix}", help=suggestion['formatted_display']):
                                # Chỉ cập nhật nếu chọn gợi ý khác với cái đang chọn
                                current_selection_id = st.session_state.selected_inat_suggestion.get('id') if st.session_state.selected_inat_suggestion else None
                                if current_selection_id != suggestion['id']:
                                    st.session_state.selected_inat_suggestion = suggestion
                                    st.session_state.inat_image_urls = [] # Reset ảnh cũ
                                    # Cập nhật ô search với tên được chọn (tùy chọn UX)
                                    st.session_state.inat_search_term = suggestion['display_name']
                                    st.session_state[last_search_key] = suggestion['display_name'] # Cập nhật last search để tránh gọi API lại
                                    st.rerun()
                            idx += 1


            # Hiển thị ảnh tham khảo nếu đã chọn 1 gợi ý
            if st.session_state.selected_inat_suggestion:
                 selected_sug = st.session_state.selected_inat_suggestion
                 st.markdown("---") # Phân cách với các nút gợi ý
                 st.write(f"Bạn đang xem: **{selected_sug['display_name']} ({selected_sug['scientific_name']})**")

                 # Tải ảnh nếu chưa có
                 if not st.session_state.inat_image_urls:
                     print(f"APP: Fetching images for {selected_sug['id']}")
                     with st.spinner("Đang tải ảnh tham khảo..."):
                          st.session_state.inat_image_urls = get_inat_image_urls(selected_sug.get('id'), count=10)
                          print(f"APP: Got {len(st.session_state.inat_image_urls)} image URLs")

                 # Hiển thị ảnh
                 if st.session_state.inat_image_urls:
                     st.write("Ảnh tham khảo (từ iNaturalist.org):")
                     cols = st.columns(5)
                     for i, img_url in enumerate(st.session_state.inat_image_urls):
                         with cols[i % 5]:
                             try:
                                 st.image(img_url, width=100) # Để Streamlit tự xử lý URL
                             except Exception as img_e:
                                 print(f"APP: Error displaying image {img_url}: {img_e}")
                                 # st.caption("Lỗi ảnh")
                 else:
                      print("APP: No image URLs to display.")
                      st.warning("Không tìm thấy ảnh tham khảo cho loài này.")

                 # Nút xác nhận cuối cùng
                 st.markdown("---")
                 confirm_button_label = f"✅ Xác nhận đây là cây: {selected_sug['display_name']}"
                 if len(confirm_button_label) > 50: # Rút gọn nếu quá dài
                     confirm_button_label = f"✅ Xác nhận: {selected_sug['display_name'][:30]}..."

                 if st.button(confirm_button_label, key=f"confirm_label_{key_prefix}", help=f"Xác nhận là {selected_sug['display_name']} ({selected_sug['scientific_name']})"):
                      final_label_to_save = selected_sug['scientific_name'] # Dùng tên khoa học để lưu
                      # Lưu ảnh dùng image_data từ session state
                      saved_ok, saved_label_dir = save_feedback_image(
                          st.session_state.image_data,          # Dữ liệu ảnh
                          st.session_state.original_filename, # Tên file gốc
                          final_label_to_save,              # Nhãn cuối cùng
                          COLLECTED_DATA_DIR
                      )
                      if saved_ok:
                          st.success(f"Đã lưu ảnh vào thư mục '{saved_label_dir}' để huấn luyện sau. Cảm ơn bạn!")
                          st.balloons()
                          st.session_state.image_saved = True
                          st.rerun() # Chạy lại để hiển thị trạng thái cuối
            


    # --- Hiển thị khi đã lưu ảnh thành công ---
    # Khối này chỉ chạy nếu prediction_done=True VÀ image_saved=True
    elif st.session_state.prediction_done and st.session_state.image_saved:
         st.success("Đã lưu phản hồi của bạn. Cảm ơn bạn đã đóng góp!")
         st.info("Bạn có thể tải lên ảnh khác ở thanh bên trái.")

# --- Chân trang ---
st.markdown("---")
st.markdown("Xây dựng với Streamlit & TensorFlow/Keras. Dữ liệu tham khảo từ iNaturalist.org.")