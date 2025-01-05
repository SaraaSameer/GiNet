import torch
from main_informer import *
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# test_file = 'Panasonic_Testing.csv'
test_file = 'test_data.csv'
model_para = 'traindata_epochs=20_seq=100_lebel=50_pred=25'
model_name = 'checkpoint.pth'


args.test_data = test_file


model_path = f'./checkpoints/{model_para}/{model_name}'

if __name__ == '__main__':
    print(args)

    model = Exp(args)
    model.Model_load(model_path)
    model.test(args, test_file)
    torch.cuda.empty_cache()
