from dexnet.grasp_model import fakeSuctionFCGQCNN, DexNet3FCGQCNN, DexNet3
import torch
import time

gqcnn_path = ""
fcgqcnn_path = ""


gqcnn = torch.load(gqcnn_path)
model1 = fakeSuctionFCGQCNN(gqcnn)
model2 = DexNet3FCGQCNN()

model1.load_state_dict(gqcnn_path)
model2.load_state_dict(fcgqcnn_path)

model1, model2 = model1.to("cuda"), model2.to("cuda")

start = time.time()
for i in range(100):
    data = torch.random((64, 1, 512, 512)).to("cuda")
    model1(data)
print("fake-fcgqcnn")
print(time.time() - start)

start = time.time()
for i in range(100):
    data = torch.random((64, 1, 512, 512)).to("cuda")
    model2(data)
print("fcgqcnn")
print(time.time() - start)

