def compute_map(model, dataset):
    if 'paris' in dataset:
        datapath = 'paris5k'
    elif 'oxford' in dataset:
        datapath = 'oxford5k'
    else:
        print('validation dataset {} is not supported'.format(dataset))

    

def compute_ap():
