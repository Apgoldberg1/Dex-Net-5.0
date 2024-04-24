from dexnet.grasp_model import fakeFCGQCNN, HighResFCGQCNN, DexNetBase
import torch
import time

def inference_benchmark(gqcnn_path, fcgqcnn_path, device):
    gqcnn_w = torch.load(gqcnn_path)
    gqcnn = DexNetBase()
    gqcnn.load_state_dict(gqcnn_w)

    model1 = fakeFCGQCNN(gqcnn)
    model2 = HighResFCGQCNN()

    model2.load_state_dict(torch.load(fcgqcnn_path))

    model1, model2 = model1.to(device), model2.to(device)
    model1.eval()
    model2.eval()

    loops = 10
    batch = 64
    img_dim = 70

    with torch.no_grad():
        print(f"computing metrics for {loops} loops through batches of {batch} {img_dim}x{img_dim} images")
        start = time.time()
        for i in range(loops):
            data = torch.rand((batch, 1, img_dim, img_dim)).to(device)
            model1(data)
        print("fake-fcgqcnn")
        print("time:", time.time() - start)

        start = time.time()
        for i in range(loops):
            data = torch.rand((batch, 1, img_dim, img_dim)).to(device)
            model2(data)
        print("fcgqcnn")
        print("time:", time.time() - start)

        


if __name__ == "__main__":
    gqcnn_path = "model_zoo/Dex-Net-3-gqcnn.pth"
    fcgqcnn_path = "model_zoo/Dex-Net-3-fcgqcnn.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: cuda not available, running inference on CPU")
    inference_benchmark(gqcnn_path, fcgqcnn_path, device)
