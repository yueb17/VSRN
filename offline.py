import argparse
import numpy as np
import numpy
from pdb import set_trace as st
import random
import torch
from model import VSRN
import torch.utils.data as data
import os
import pickle
from vocab import Vocabulary  # NOQA
import nltk
import time

# define unlabeled training dataloader
def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, img_ids, caption_labels, caption_masks = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    caption_labels_ = torch.stack(caption_labels, 0)
    caption_masks_ = torch.stack(caption_masks, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]


    return images, targets, lengths, ids, caption_labels_, caption_masks_


class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f8k, f30k, coco, 10crop
    """

    def __init__(self, data_path, data_split, vocab, opt):
        self.vocab = vocab
        loc = data_path + '/'

        # Captions
        self.captions = []
        token_caption = []
        with open(loc+'%s_caps.txt' % data_split, 'rb') as f:
            for line in f:
                self.captions.append(line.strip())
                tokens = nltk.tokenize.word_tokenize(str(line.strip()).lower().decode('utf-8'))
                token_caption.append(tokens)        

        each_cap_lengths = [len(cap) for cap in token_caption]
        calculate_max_len = max(each_cap_lengths) + 2
        print(calculate_max_len)

        # Image features
        self.images = np.load(loc+'%s_ims.npy' % data_split)
        self.length = len(self.captions)
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000

        self.max_len = opt.max_len


    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index/self.im_div
        image = torch.Tensor(self.images[img_id])
        caption = self.captions[index]
        vocab = self.vocab

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower().decode('utf-8'))
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)


        ##### deal with caption model data
        # label = np.zeros(self.max_len)
        mask = np.zeros(self.max_len + 1)
        gts = np.zeros((self.max_len + 1))

        # print(tokens)
        cap_caption = ['<start>'] + tokens + ['<end>']
        # print(cap_caption)
        if len(cap_caption) > self.max_len - 1:
            cap_caption = cap_caption[:self.max_len]
            cap_caption[-1] = '<end>'
            
        for j, w in enumerate(cap_caption):
            gts[j] = vocab(w)

        non_zero = (gts == 0).nonzero()


        mask[:int(non_zero[0][0]) + 1] = 1




        caption_label = torch.from_numpy(gts).type(torch.LongTensor)
        caption_mask = torch.from_numpy(mask).type(torch.FloatTensor)

        return image, target, index, img_id, caption_label, caption_mask

    def __len__(self):
        return self.length

def get_precomp_loader(data_path, data_split, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=2):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset(data_path, data_split, vocab, opt)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn)
    return data_loader

def get_loaders(data_name, vocab, crop_size, batch_size, workers, opt):
    dpath = os.path.join(opt.original_data_path, data_name)
    if opt.data_name.endswith('_precomp'):
        train_loader = get_precomp_loader(dpath, 'train', vocab, opt,
                                          batch_size, False, workers)    

    return train_loader

def encode_data(model, data_loader, log_step=10):#, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    # switch to evaluate mode
    model.val_start()

    # numpy array to keep all the embeddings
    img_embs = None
    cap_embs = None
    for i, (images, captions, lengths, ids, caption_labels, caption_masks) in enumerate(data_loader):
        # compute the embeddings
        img_emb, cap_emb, GCN_img_emd = model.forward_emb(images, captions, lengths,
                                             volatile=True)

        # initialize the numpy arrays given the size of the embeddings
        if img_embs is None:
            img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
            cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1)))

        # preserve the embeddings by copying from gpu and converting to numpy
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids] = cap_emb.data.cpu().numpy().copy()


        del images, captions

    return img_embs, cap_embs

def i2t(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = images.shape[0] / 5
    index_list = []

    ranks = numpy.zeros(npts)
    top1 = numpy.zeros(npts)
    top5 = []
    for index in range(npts):
        if index % 1000 == 0:
            print(index)

        # Get query image
        im = images[5 * index].reshape(1, images.shape[1])

        # Compute scores
        if measure == 'order':
            bs = 100
            if index % bs == 0:
                mx = min(images.shape[0], 5 * (index + bs))
                im2 = images[5 * index:mx:5]
                d2 = order_sim(torch.Tensor(im2).cuda(),
                               torch.Tensor(captions).cuda())
                d2 = d2.cpu().numpy()
            d = d2[index % bs]
        else:
            d = numpy.dot(im, captions.T).flatten()
        inds = numpy.argsort(d)[::-1]
        index_list.append(inds[0])

        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = numpy.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]
        top5.append(inds[:5])

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1, top5)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = images.shape[0] / 5
    ims = numpy.array([images[i] for i in range(0, len(images), 5)])

    ranks = numpy.zeros(5 * npts)
    top1 = numpy.zeros(5 * npts)
    for index in range(npts):

        # Get query captions
        queries = captions[5 * index:5 * index + 5]

        # Compute scores
        if measure == 'order':
            bs = 100
            if 5 * index % bs == 0:
                mx = min(captions.shape[0], 5 * index + bs)
                q2 = captions[5 * index:mx]
                d2 = order_sim(torch.Tensor(ims).cuda(),
                               torch.Tensor(q2).cuda())
                d2 = d2.cpu().numpy()

            d = d2[:, (5 * index) % bs:(5 * index) % bs + 5].T
        else:
            d = numpy.dot(queries, ims.T)
        inds = numpy.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = numpy.argsort(d[i])[::-1]
            ranks[5 * index + i] = numpy.where(inds[i] == index)[0][0]
            top1[5 * index + i] = inds[i][0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def evaluate(img_embs, cap_embs, opt):
    r, rt = i2t(img_embs, cap_embs, measure=opt.measure, return_ranks=True)
    ri, rti = t2i(img_embs, cap_embs,
                      measure=opt.measure, return_ranks=True)

    i2t_top5 = rt[2]
    t2i_top1 = rti[1]
    
    ar = (r[0] + r[1] + r[2]) / 3
    ari = (ri[0] + ri[1] + ri[2]) / 3
    rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
    print("rsum: %.1f" % rsum)
    print("Average i2t Recall: %.1f" % ar)
    print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
    print("Average t2i Recall: %.1f" % ari)
    print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)

    return i2t_top5, t2i_top1


def check_order(img_ids, cap_ids, div=5):
    cap_ids_check = []
    for img_id in img_ids:
        for i in range(div):
            cap_ids_check.append(img_id*div + i)
    if cap_ids_check == cap_ids:
        print("ALL GOOD!!!")
    else:
        print("MIS MATCHED!!! GO FOR CHECK!!!")

# def maxRank_alignment():
#     # 
#     # from image to retrieve captions: 1 img to 5 caps
#     new_img_id2full = PL_text_index
#     new_cap_id2PL = []
#     for i in range(len(i2t_top5)):
#         for j in i2t_top5[i]:
#             new_cap_id2PL.append(j)
#     new_cap_id2full = list(np.array(PL_text_index)[new_cap_id2PL])

#     i2tPL_img = img_embs[new_img_id2full]
#     i2tPL_cap = cap_embs[new_cap_id2full]

#     # from caption to retrieve images: 1 cap to 1 img
#     new_cap_id2full = PL_text_index
#     # st()
#     new_img_id2PL = [int(i*opt.div) for i in t2i_top1]
#     new_img_id2full = list(np.array(PL_text_index)[new_img_id2PL])

#     t2iPL_img = img_embs[new_img_id2full]
#     t2iPL_cap = cap_embs[new_cap_id2full]

#     # test PL set
#     i2t_top5_, t2i_top1_ = evaluate(i2tPL_img, i2tPL_cap, opt)
#     # st()
#     i2t_top5_, t2i_top1_ = evaluate(t2iPL_img, t2iPL_cap, opt)
# #     # st()

def noMiss_alignment(imgs, caps, opt): # 5N * 5N
    imgs_inds = [i for i in range(0, len(imgs), opt.div)]
    imgs_ = imgs[imgs_inds] # N

    i2t_mat = np.dot(imgs_, caps.T) # N * 5N
    div_list = []
    for i in range(opt.div):
        each_div_list = [j+i for j in range(0, len(imgs), opt.div)]
        div_list.append(each_div_list)
    # st()

    div_times_ = 0
    for each_list in div_list:
        div_times_ = div_times_ + i2t_mat[:,each_list]

    i2t_mat_ = div_times_/opt.div # N * N

    lowest = np.min(i2t_mat_)
    removal_value = lowest - 1

    # st()
    pairs = []
    for itr in range( len(i2t_mat_) ): # iterate N times
        if itr % 100 == 0:
            print(itr)
        
        each_max_pos = np.argmax(i2t_mat_)
        row, col = divmod(each_max_pos, i2t_mat_.shape[1])

        i2t_mat_[row,:] = removal_value
        i2t_mat_[:,col] = removal_value

        each_pair = np.array([row, col])

        pairs.append(each_pair)
        

    pairs = np.array(pairs)

    if len(set(pairs[:,0])) == len(pairs) and len(set(pairs[:,1])) == len(pairs) and len((set(pairs[:,0]) - set(pairs[:,1]))) == 0:
        print("ALL GOOD FOR PAIRS!!!")

    acc = np.count_nonzero(pairs[:,0] - pairs[:,1]) / float(len(pairs))
    print('Div times avg PL acc:', 1-acc)
    # st()
    
    # recover to 5*N times
    alg_imgs_index = []
    alg_caps_index = []
    for i in range(len(pairs)): # N 
        for j in range(opt.div): # 5
            alg_imgs_index.append(pairs[i,0]*opt.div+j)
            alg_caps_index.append(pairs[i,1]*opt.div+j)

    return alg_imgs_index, alg_caps_index




    



def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline_model_path', type=str, default='',
                        help='path to pretraine teacher network with labeled data index')
    parser.add_argument('--original_data_path', type=str, default='',
    					help='path to load original dataset')
    parser.add_argument('--data_name', type=str, default='',
    					help='name of datasets (flickr or coco)')
    parser.add_argument('--data_split', type=str, default='train',
    					help='data split, always should be train')
    parser.add_argument('--vocab_path', default='./vocab/',
                        help='Path to saved vocabulary pickle files.')
    parser.add_argument('--crop_size', default=224, type=int,
                        help='Size of an image crop as the CNN input.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument("--max_len", type=int, default=60,
                        help='max length of captions(containing <sos>,<eos>)')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--measure', default='cosine',
                        help='Similarity measure used (cosine|order)')
    parser.add_argument('--PL_results', action='store_true',
                        help='plot PL performance')
    parser.add_argument('--PL_ratio', type=float, default=0.1,
                        help='ratio to get fake label data pair, compared with FULL size')
    parser.add_argument('--div', type=int, default=5,
                        help='ratio between image and text (1 image to "div" text)')

    opt = parser.parse_args()
    print(opt)

    # load original data
    captions_file = opt.original_data_path + '/' + opt.data_name + '/' + opt.data_split + '_caps.txt'
    captions = []
    with open(captions_file, 'rb') as f:
    	for line in f:
    		captions.append(line.strip())

    images_file = opt.original_data_path + '/' + opt.data_name + '/' + opt.data_split + '_ims.npy'
    images = np.load(images_file)

    # load semi labeled data index (np.array)
    labeled_image_index = np.load(opt.baseline_model_path + '/random_image_index.npy')
    labeled_text_index = np.load(opt.baseline_model_path + '/random_text_index.npy')

    # shuffle to get unlabeled data index
    # how to shuffle??? 1 v 1 or 1 v 5
    image_index_set = set(np.array(range(len(images))))
    text_index_set = set(np.array(range(len(captions))))

    unlabeled_image_index = list(image_index_set - set(labeled_image_index))
    unlabeled_text_index = list(text_index_set - set(labeled_text_index))
    # sort: set is not ordered
    unlabeled_image_index.sort()
    unlabeled_text_index.sort()

    # check order, if matches with each other
    check_order(list(labeled_image_index), list(labeled_text_index), opt.div)
    check_order(unlabeled_image_index, unlabeled_text_index, opt.div)

    # sample a subset for PL labels
    total_size = len(images)
    current_unlabeled_size = len(unlabeled_image_index)
    PL_size = int(total_size * opt.PL_ratio) # compared with FULL size

    PL_index = random.sample(range(0, current_unlabeled_size), PL_size) # list
    PL_index.sort()

    PL_image_index = list(np.array(unlabeled_image_index)[PL_index])
    PL_text_index = []
    for PL_img_id in PL_image_index:
        for i in range(opt.div):
            PL_text_index.append(PL_img_id*opt.div + i)
    check_order(PL_image_index, PL_text_index, opt.div)

    # 1 v 1 shuffle NO NEED TO SHUFFLE HERE!!!
    # random.shuffle(unlabeled_image_index)
    # random.shuffle(unlabeled_text_index)

    # load pretrained teacher network
    checkpoint = torch.load(opt.baseline_model_path + '/model_best.pth.tar')
    ck_opt = checkpoint['opt']
    model = VSRN(ck_opt)
    model.load_state_dict(checkpoint['model'])
    
    # get embeddings for unlabeled training data
    # load vocabulary used by the model
    with open(os.path.join(opt.vocab_path,
                           '%s_vocab.pkl' % opt.data_name), 'rb') as f:
        vocab = pickle.load(f)

    data_loader = get_loaders(data_name=opt.data_name, vocab=vocab, 
                              crop_size=opt.crop_size, batch_size=opt.batch_size, workers=opt.workers, opt=opt)

    # get embedding for full training set (both labeled and unlabeled), and save accordingly
    # img_embs, cap_embs = encode_data(model, data_loader)
    # np.save(opt.baseline_model_path + '/img_embs.npy', img_embs) # save data
    # np.save(opt.baseline_model_path + '/cap_embs.npy', cap_embs)
    img_embs = np.load(opt.baseline_model_path + '/img_embs.npy') # load saved data
    cap_embs = np.load(opt.baseline_model_path + '/cap_embs.npy')

    # split into labeled and unlabeled, further PL
    L_imgs = img_embs[labeled_text_index]
    L_caps = cap_embs[labeled_text_index]

    U_imgs = img_embs[unlabeled_text_index]
    U_caps = cap_embs[unlabeled_text_index]

    PL_imgs = img_embs[PL_text_index]
    PL_caps = cap_embs[PL_text_index]
    # st()
    # compute 
    alg_imgs_index, alg_caps_index = noMiss_alignment(PL_imgs, PL_caps, opt)
    # load
    # alg_imgs_index = np.load(opt.baseline_model_path + '/alg_imgs_index' + str(opt.PL_ratio) + '.npy')
    # alg_caps_index = np.load(opt.baseline_model_path + '/alg_caps_index' + str(opt.PL_ratio) + '.npy')
    alg_imgs = img_embs[alg_imgs_index]
    alg_caps = cap_embs[alg_caps_index]
    i2t_top5, t2i_top1 = evaluate(alg_imgs, alg_caps, opt)

    # save alg index
    # np.save(opt.baseline_model_path + '/alg_imgs_index' + str(opt.PL_ratio) + '.npy', alg_imgs_index)
    # np.save(opt.baseline_model_path + '/alg_caps_index' + str(opt.PL_ratio) + '.npy', alg_caps_index)


    # get P labels. NOTE: from both sides
    # i2t_top5, t2i_top1 = evaluate(PL_imgs, PL_caps, opt) # get rank scores and print performances
    
    


if __name__ == '__main__':
    main()