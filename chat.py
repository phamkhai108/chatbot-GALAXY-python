import random
import json
import torch
import os
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from getWeather import weather
from unidecode import unidecode
from tom_tat import tom_tat_van_ban
from acronym.stand_words import normalize_text, dictions

# Chọn thiết bị chạy mô hình PyTorch 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

intents = {"intents": []}

# Lặp qua tất cả các file JSON trong thư mục "stories" và mở các file json trong đó
folder_path = "stories"
for file_name in os.listdir(folder_path):
    if file_name.endswith(".json"):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            intents_data = json.load(file)
            intents["intents"].extend(intents_data["intents"])

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Yasuo"
waiting_for_summary = False

def chat(user_input):
    global waiting_for_summary
    # Xử lý input từ người dùng
    processed_input = unidecode(user_input).lower()
    processed_input = normalize_text(processed_input, dictions)
    processed_input = tokenize(processed_input)
    
    X = bag_of_words(processed_input, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # Nếu đang trong trạng thái chờ tóm tắt
    if waiting_for_summary:
        if prob.item() > 0.95:
            waiting_for_summary = False
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    return random.choice(intent['responses'])
        summary = tom_tat_van_ban(user_input)
        return summary

    # Nếu xác suất dự đoán cao hơn ngưỡng (0.75), chọn phản hồi tương ứng
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                if tag == "tom_tat": 
                    waiting_for_summary = True
                    return random.choice(intent['responses'])
                elif tag == 'weather':
                    return weather(user_input)
                else:
                    return random.choice(intent['responses'])
    else:
            # Ngay lập tức chọn một phản hồi ngẫu nhiên từ intent có tag là "khong_hieu"
        intent = next((i for i in intents['intents'] if i["tag"] == "khong_hieu"), None)
        if intent:
            return random.choice(intent['responses'])
        else:
            return "I do not understand..."



        
print("Let's chat! (type 'quit' to exit)")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        print(f"{bot_name}: Goodbye!")
        break
    response = chat(user_input)
    print(f"{bot_name}: {response}")