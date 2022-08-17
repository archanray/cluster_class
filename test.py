from glob import glob
import pickle as pkl

def load_cifar_results(model=20, type_=True):
    y_preds = []
    all_y_preds = []
    directory = "./saved_outputs/"
    if type_ == True:
        all_files = directory+"c*"
    else:
        all_files = directory+"nc*"
    all_files = glob(all_files)

    check_str = "resnet"+str(model)+"_"
    filtered_files = []
    for files in all_files:
        if check_str in files:
            filtered_files.append(files)
    if filtered_files == []:
        print("error in model number, available choices [20,32,56]")
        exit(1)
    print(filtered_files)

    for filenames in filtered_files:
        with open(filenames, "rb") as f:
            A = pkl.load(f)
            print(A)

    return y_preds, all_y_preds

load_cifar_results(model=56, type_=False)