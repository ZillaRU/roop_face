import numpy as np


def calWeight(d,k):
    '''
    :param d: 融合重叠部分直径
    :param k: 融合计算权重参数
    :return:
    '''
    x = np.arange(-d/2,d/2)
    y = 1/(1+np.exp(-k*x))
    return y

 
def imgFusion2(img1,img2, overlap,left_right=True):
    '''
    图像加权融合
    :param img1:
    :param img2:
    :param overlap: 重合长度
    :param left_right: 是否是左右融合
    :return:
    '''
    # 这里先暂时考虑平行向融合
    wei = calWeight(overlap,0.05)    # k=5 这里是超参
    h1, w1, c1 = img1.shape
    h2, w2, c2 = img2.shape
    
    if left_right:  # 左右融合
        assert h1 == h2 and c1 == c2
        img_new = np.zeros((h1, w1+w2-overlap, c1))
        # if img1.shape[1] >= img_new.shape[1]:
        #     return img1[:, :img1.shape[1], :]
        img_new[:,:w1,:] = img1
        wei_expand = np.tile(wei,(h1,1))  # 权重扩增
        wei_expand = np.expand_dims(wei_expand,2).repeat(3, axis=2)
        img_new[:, w1-overlap:w1, :] = (1-wei_expand)*img1[:,w1-overlap:w1, :] + wei_expand*img2[:,:overlap, :]
        img_new[:, w1:, :]=img2[:,overlap:, :]
    else:   # 上下融合
        assert w1 == w2 and c1 == c2
        img_new = np.zeros((h1+h2-overlap, w1, c1))
        # if img1.shape[0] >= img_new.shape[0]:
        #     return img1[:img1.shape[0], :, :]
        img_new[:h1, :, :] = img1
        wei = np.reshape(wei,(overlap,1))
        wei_expand = np.tile(wei,(1, w1))
        wei_expand = np.expand_dims(wei_expand,2).repeat(3, axis=2)
        img_new[h1-overlap:h1, :, :] = (1-wei_expand)*img1[h1-overlap:h1,:, :]+wei_expand*img2[:overlap,:, :]
        img_new[h1:, :, :] = img2[overlap:,:, :]
    return img_new


def imgFusion(img_list, overlap, res_w, res_h):
    print(res_w, res_h)
    pre_v_img = None
    for vi in range(len(img_list)):
        h_img = np.transpose(img_list[vi][0], (1,2,0))
        for hi in range(1, len(img_list[vi])):
            new_img = np.transpose(img_list[vi][hi], (1,2,0))
            h_img = imgFusion2(h_img, new_img, (h_img.shape[1]+new_img.shape[1]-res_w) if (hi == len(img_list[vi])-1) else overlap, True)
        pre_v_img = h_img if pre_v_img is None else imgFusion2(pre_v_img, h_img, (pre_v_img.shape[0]+h_img.shape[0]-res_h) if vi == len(img_list)-1 else overlap, False)
    return np.transpose(pre_v_img, (2,0,1))
