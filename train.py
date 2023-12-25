import numpy as np
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet


intents = {"intents": []}

# Lặp qua tất cả các file JSON trong thư mục "stories" và mở các file json trong đó
folder_path = "stories"
for file_name in os.listdir(folder_path):
    if file_name.endswith(".json"):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            intents_data = json.load(file)
            intents["intents"].extend(intents_data["intents"])
            

# Tạo danh sách các từ 
all_words = []

# lưu danh sách các nhãn tương ứng greeting, goodbye,...
tags = []

#  Chứa [các cặp từ đã xử lý và nhãn
xy = []

# Lặp qua mỗi phần tử trong intents của json
for intent in intents['intents']:
    # Lấy tag và lưu vào danh sách tags
    tag = intent['tag']
    tags.append(tag)

    # Lặp qua mỗi pattern xử lý từ 
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        # lưu vào danh sách các từ
        all_words.extend(w)
        # Lưu các từ tương ứng với tag
        xy.append((w, tag))

# Xử lý loại bỏ các dấu trong thư viện từ 
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]

# Loại bỏ các từ trùng lặp
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# print(len(xy), "patterns")
# print(len(tags), "tags:", tags)
# print(len(all_words), "Các từ đã được xử lý:", all_words)

# danh sách biểu diễn bag of words
X_train = []
# danh sách tag tương ứng
Y_train = []

# Xây dựng dữ liệu, mỗi mẫu dữ liệu trong xy chuyển về vecto
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    # Lưu các vị trí nhãn
    label = tags.index(tag)
    Y_train.append(label)

# Chuyển thành mảng để xử lý 
X_train = np.array(X_train)
Y_train = np.array(Y_train)

#  Số lượng lần duyện qua toàn bộ tập tin
num_epochs = 1000
# Số lượng mẫu dữ liệu được sử dụng trong mỗi lần cập nhật
batch_size = 8
# Tốc độ học của mô hình
learning_rate = 0.001
# Kích thước vecto đầu vào
input_size = len(X_train[0])
# Kích thước lớp ẩn trong neural
hidden_size = 8
# Số lượng đầu ra tương ứng với tags
output_size = len(tags)

# Tạo đối tượng tập dữ liệu cho quá trình đào tạo
class ChatDataset(Dataset):
    # Khởi tạo đối tượng
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train

    # Lấy dữ liệu
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    #Tổng số lượng mẫu
    def __len__(self):
        return self.n_samples

# Tạo đối tượng dataset
dataset = ChatDataset()

#Tạo đối tượng Dataloader chia dữ liệu thành các mini_batch (tập con nhỏ của dữ liệu đào tạo)
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

# Kiểm tra và thiết lập thiết bị tính toán GPU hoặc CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Khởi tạo mô hình neural network và chuyển nó lên device
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Khởi tạo hàm mất mát đo sự chênh lệch giữa dự đoán của mô hình và thực thể
criterion = nn.CrossEntropyLoss()

# Khởi tạo tối ưu hóa 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Lặp qua các epochs (1000)
for epoch in range(num_epochs):
    #Lặp qua các mini-batch trong train_loader
    for (words, labels) in train_loader:

        # Chuyển dữ liệu của mini-batch lên thiết bị đã chọn
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Chuyển tiếp đến Neural để dự đoán output
        outputs = model(words)

        # Tính toán hàm mất mát để đo lường và điều chỉnh trọng số
        loss = criterion(outputs, labels)
        
        # Thực hiện quá trình lùi lại và cập nhật trọng số
        # Làm sạch gradient của tất cả các tham số trong mô hình trước khi tính toán gradient mới.
        # gradient tốc độ mà hàm mất mát thay đổi
        optimizer.zero_grad()
        #  Tính gradient của hàm mất mát theo tất cả các tham số
        loss.backward()
        # Cập nhật trọng số của mô hình sử dụng giải thuật tối ưu hóa
        optimizer.step()
    
    # Theo dõi và in ra màn hình thông tin về mất mát sau mỗi 100 epochs
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# In ra màn hình giá trị cuối cùng của hàm mất mát
print(f'final loss: {loss.item():.4f}')

# Tạo ra một dictionary 
data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

# Lưu trữ thông tin về mô hình và dữ liệu liên quan
FILE = "data.pth"
torch.save(data, FILE)

# In ra thông báo thông báo rằng quá trình huấn luyện đã hoàn tất 
print(f'training complete. file saved to {FILE}')
