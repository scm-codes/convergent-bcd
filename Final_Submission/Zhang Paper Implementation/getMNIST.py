def getMNIST():
    import torch.nn.functional as F
    from torchvision import datasets
    from torchvision.transforms import ToTensor, Compose, Normalize

    # load train and test
    train_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=Compose([ToTensor(), Normalize((0,),(1,))]),
    )
    
    test_data= datasets.MNIST( root="data",
        train=False,
        download=True,
        transform = Compose([ToTensor(), Normalize((0,),(1,))]),
    )
    
    # load train and test 
    N_train = len(train_data)
    N_test = len(test_data)
    channels,height,width = train_data[0][0].size()
    
    x_train = train_data.data.numpy().reshape(N_train,height*width)
    x_test = test_data.data.numpy().reshape(N_test,height*width)
    
    y_train = F.one_hot(train_data.targets, num_classes=-1).numpy()
    y_test = F.one_hot(test_data.targets, num_classes=-1).numpy()
    
    return x_train,y_train,x_test,y_test