from dexnet.grasp_model import fakeSuctionFCGQCNN, DexNet3FCGQCNN, DexNet3
import torch
import time

def inference_benchmark(gqcnn_path, fcgqcnn_path):
    gqcnn_w = torch.load(gqcnn_path)
    gqcnn = DexNet3()
    gqcnn.load_state_dict(gqcnn_w)

    model1 = fakeSuctionFCGQCNN(gqcnn)
    model2 = DexNet3FCGQCNN()

    model2.load_state_dict(torch.load(fcgqcnn_path))

    model1, model2 = model1.to("cuda"), model2.to("cuda")
    model1.eval()
    model2.eval()

    with torch.no_grad():
        start = time.time()
        for i in range(10):
            data = torch.rand((64, 1, 70, 70)).to("cuda")
            model1(data)
        print("fake-fcgqcnn")
        print(time.time() - start)

        start = time.time()
        for i in range(10):
            data = torch.rand((64, 1, 70, 70)).to("cuda")
            model2(data)
        print("fcgqcnn")
        print(time.time() - start)


if __name__ == "__main__":
    gqcnn_path = "model_zoo/Dex-Net-3-gqcnn.pth"
    fcgqcnn_path = "model_zoo/Dex-Net-3-fcgqcnn.pt"
    inference_benchmark(gqcnn_path, fcgqcnn_path)