from dexnet.grasp_model import fakeFCGQCNN, HighResFCGQCNN, DexNetBase, EfficientNet
import torch
import time

def inference_benchmark(gqcnn_path, fcgqcnn_path, efficientnet_path, device):
    gqcnn_w = torch.load(gqcnn_path)
    base_gqcnn = DexNetBase()
    base_gqcnn.load_state_dict(gqcnn_w)

    fake_fcgqcnn = fakeFCGQCNN(base_gqcnn)
    high_res_fcgqcnn = HighResFCGQCNN()

    high_res_fcgqcnn.load_state_dict(torch.load(fcgqcnn_path))

    fake_fcgqcnn, high_res_fcgqcnn = fake_fcgqcnn.to(device), high_res_fcgqcnn.to(device)
    fake_fcgqcnn.eval()
    high_res_fcgqcnn.eval()

    efficientnet_gqcnn = EfficientNet()
    efficientnet_gqcnn.load_state_dict(torch.load(efficientnet_path))

    loops = 1
    batch = 64
    img_dim = 70

    print(f"computing FC-GQ-CNN metrics for {loops} loops through batches of {batch} {img_dim}x{img_dim} images")
    with torch.no_grad():
        start = time.time()
        for i in range(loops):
            data = torch.rand((batch, 1, img_dim, img_dim)).to(device)
            fake_fcgqcnn(data)
        print("fake-fcgqcnn")
        print("time:", time.time() - start)

        start = time.time()
        for i in range(loops):
            data = torch.rand((batch, 1, img_dim, img_dim)).to(device)
            high_res_fcgqcnn(data)
        print("fcgqcnn")
        print("time:", time.time() - start)

    effientnet_gqcnn, base_gqcnn = efficientnet_gqcnn.to(device), base_gqcnn.to(device)
    loops = 1
    batch = 64

    print(f"computing GQ-CNN metrics for {loops} loops through batches of {batch} 32x32 images")
    with torch.no_grad():
        start = time.time()
        for i in range(loops):
            data = torch.rand((batch, 1, 32, 32)).to(device)
            efficientnet_gqcnn(data)
        print("efficientnet_gqcnn")
        print("time:", time.time() - start)

        start = time.time()
        for i in range(loops):
            data = torch.rand((batch, 1, 32, 32)).to(device)
            base_gqcnn(data)
        print("base_gqcnn")
        print("time:", time.time() - start)


if __name__ == "__main__":
    base_gqcnn_path = "model_zoo/DexNetBaseSuction.pth"
    base_fcgqcnn_path = "model_zoo/SuctionFCGQCNN.pt"
    efficientnet_gqcnn_path = "model_zoo/EfficientNetSuction.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: cuda not available, running inference on CPU")
    inference_benchmark(base_gqcnn_path, base_fcgqcnn_path, efficientnet_gqcnn_path, device)
