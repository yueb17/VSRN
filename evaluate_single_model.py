from vocab import Vocabulary
import evaluation
import argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_model_path', type=str, default='',
                        help='path to single pretrained model')
    parser.add_argument('--data_path', type=str, default='../data/',
    					help='path to original data')
    opt = parser.parse_args()
    print(opt)

    evaluation.evalrank(opt.pretrain_model_path, data_path=opt.data_path, split="test", fold5=False)

if __name__ == '__main__':
    main()