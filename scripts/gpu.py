import torch

if __name__ == "__main__":
    x = torch.cuda.is_available()
    print(x)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    name = torch.cuda.get_device_name(0)
    print(name)

    print(torch.version.cuda)

    print(torch.cuda.get_arch_list())