from dexnet.grasp_model import fakeFCGQCNN, HighResFCGQCNN, DexNetBase, EfficientNet
import torch
import time
import matplotlib.pyplot as plt
import numpy as np

def img_dim_scale_plot(fcgqcnn, fake_fcgqcnn):
    loops = 20
    batch = 50

    time_log1, time_log2 = [[],[],[]], []

    with torch.no_grad():
        # warm-up inference
        fcgqcnn(torch.zeros(1, 1, 40, 40).to("cuda"))
        start_dim, end_dim, step = 40, 80, 10

        # print(f"fcgqcnn {loops} loops of batch size {batch} of varying image sizes")
        # for i in range(3):
        #     for img_dim in range(start_dim, end_dim, step):
        #         start = time.time()
        #         for _ in range(loops):
        #             data = torch.rand((batch, 1, img_dim, img_dim)).to(device)
        #             fcgqcnn(data)
        #         print("time:", time.time() - start)
        #         time_log1[i].append(time.time() - start)
        # time_log1 = np.array(time_log1)
        # time_log1 = np.mean(time_log1, axis=0)

        # warm-up inference
        fake_fcgqcnn(torch.zeros(1, 1, 40, 40).to("cuda"))

        print(f"fake-fcgqcnn {loops} loops of batch size {batch} of varying image sizes")
        
        # We do one less step because it takes too long!
        for img_dim in range(start_dim, end_dim, step):
            start = time.time()
            for _ in range(loops):
                data = torch.rand((batch, 1, img_dim, img_dim)).to(device)
                fake_fcgqcnn(data)
            print("time:", time.time() - start)
            time_log2.append(time.time() - start)


    # plt.plot(list(range(start_dim, end_dim, step)), time_log1, label = "FC-GQ-CNN")
    # plt.title("FC-GQ-CNN Scaling")
    plt.plot(list(range(start_dim, end_dim, step)), time_log2, label = "Naive Approach", color = "orange")
    plt.title("Naive Approach Scaling")
    plt.xlabel("image dimension (px)")
    plt.ylabel("inference time per 1000 images (sec)")
    plt.xticks(list(range(start_dim, end_dim, step)))
    plt.legend()
    # plt.savefig("outputs/fcgqcnn_time_scale.jpg")
    plt.savefig("outputs/naive_fcgqcnn_time_scale.jpg")
    plt.show()
    plt.close()

    # plt.plot(img_size2, time_log2, label = "Naive Approach")

    

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
    base_gqcnn.eval()

    efficientnet_gqcnn = EfficientNet()
    efficientnet_gqcnn.load_state_dict(torch.load(efficientnet_path))
    efficientnet_gqcnn.eval()

    loops = 0
    batch = 128
    img_dim = 70

    img_dim_scale_plot(high_res_fcgqcnn, fake_fcgqcnn)

    print(f"computing FC-GQ-CNN metrics for {loops} loops through batches of {batch} {img_dim}x{img_dim} images")
    with torch.no_grad():
        start = time.time()
        for _ in range(loops):
            data = torch.rand((batch, 1, img_dim, img_dim)).to(device)
            fake_fcgqcnn(data)
        print("fake-fcgqcnn")
        print("time:", time.time() - start)

        start = time.time()
        for _ in range(loops):
            data = torch.rand((batch, 1, img_dim, img_dim)).to(device)
            high_res_fcgqcnn(data)
        print("fcgqcnn")
        print("time:", time.time() - start)

    efficientnet_gqcnn, base_gqcnn = efficientnet_gqcnn.to(device), base_gqcnn.to(device)
    loops = 0
    batch = 512

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
    base_fcgqcnn_path = "model_zoo/SuctionFCGQCNN.pth"
    efficientnet_gqcnn_path = "model_zoo/EfficientNetSuction.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: cuda not available, running inference on CPU")
    inference_benchmark(base_gqcnn_path, base_fcgqcnn_path, efficientnet_gqcnn_path, device)
