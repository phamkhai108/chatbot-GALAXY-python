import streamlit as st
import time
from chat import chat

st.title('Chatbot GALAXY')

# Kiểm tra và khởi tạo session_state nếu cần
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị tin nhắn có trong session_state
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Lấy dữ liệu từ chat_input và xử lý
if prompt := st.chat_input("Hãy nhập vào yêu cầu?"):
    # Thêm tin nhắn của người dùng vào session_state
    st.session_state.messages.append(
        {
            "role": 'user',
            "content": prompt
        }
    )

    # hiển thị tin nhắn người dùng vừa nhập trong giao diện Streamlit.
    with st.chat_message('user'):
        st.markdown(prompt)

    # tạo một phần tin nhắn mới với vai trò là 'assistant'. 
    with st.chat_message('assistant'):
        full_res = ""
        holder = st.empty()
        
        # Gọi hàm chat và xử lý kết quả
        response = chat(prompt)
        
        # Chạy animation tạo cảm giác trả lời của bo
        for word in response.split():
            full_res += word + " "
            time.sleep(0.05)
            holder.markdown(full_res + "█")
            
        # Hiển thị ra câu trả lời
        holder.markdown(full_res)

    # Thêm tin nhắn của bot vào session_state
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_res
        }
    )
