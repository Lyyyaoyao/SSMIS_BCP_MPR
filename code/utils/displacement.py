import torch
from einops import rearrange
import random
import numpy as np
from sklearn.neighbors import KDTree
# import faiss



def ABD_R_KDTree(outputs1_max, outputs2_max, volume_batch, volume_batch_strong,
                outputs1_unlabel, outputs2_unlabel, args):
    """
    使用 KD-Tree 改进 ABD-R 的 patch 匹配过程
    """
    # 拆分 patch
    patches_1 = rearrange(outputs1_max[args.labeled_bs:], 'b (h p1) (w p2) -> b (h w) (p1 p2)',
                          p1=args.patch_size, p2=args.patch_size)
    patches_2 = rearrange(outputs2_max[args.labeled_bs:], 'b (h p1) (w p2) -> b (h w) (p1 p2)',
                          p1=args.patch_size, p2=args.patch_size)

    image_patch_1 = rearrange(volume_batch.squeeze(1)[args.labeled_bs:],
                              'b (h p1) (w p2) -> b (h w) (p1 p2)',
                              p1=args.patch_size, p2=args.patch_size)
    image_patch_2 = rearrange(volume_batch_strong.squeeze(1)[args.labeled_bs:],
                              'b (h p1) (w p2) -> b (h w) (p1 p2)',
                              p1=args.patch_size, p2=args.patch_size)

    # 输出分布的 patch 表示 (用于语义相似性度量)
    patches_outputs_1 = rearrange(outputs1_unlabel, 'b c (h p1) (w p2) -> b (h w) c (p1 p2)',
                                  p1=args.patch_size, p2=args.patch_size)
    patches_outputs_2 = rearrange(outputs2_unlabel, 'b c (h p1) (w p2) -> b (h w) c (p1 p2)',
                                  p1=args.patch_size, p2=args.patch_size)

    # 取每个 patch 的平均概率分布作为语义特征向量
    feat_1 = torch.mean(patches_outputs_1.detach(), dim=3)  # [B, N, C]
    feat_2 = torch.mean(patches_outputs_2.detach(), dim=3)

    batch_size = feat_1.shape[0]

    # 复制图像 patch 用于修改
    new_image_patch_1 = image_patch_1.clone()
    new_image_patch_2 = image_patch_2.clone()

    for i in range(batch_size):
        # 当前样本的特征
        f1 = feat_1[i].cpu().numpy()  # [N, C]
        f2 = feat_2[i].cpu().numpy()

        # 查找最不确定的 patch（平均概率熵最大 or 置信度最低）
        conf_1 = patches_1[i].mean(dim=1).detach().cpu().numpy()  # [N]
        conf_2 = patches_2[i].mean(dim=1).detach().cpu().numpy()

        b = np.argmin(conf_1)  # model1 最不确定的 patch index
        d = np.argmin(conf_2)  # model2 最不确定的 patch index

        # 构建 KD-Tree：用 model2 的特征查找与 model1 最不确定 patch 最相似的 patch
        tree2 = KDTree(f2)
        _, idx1 = tree2.query(f1[b].reshape(1, -1), k=1)  # 找 model2 中最相似的 patch
        best_match_from_2 = idx1[0][0]

        # 同理，model1 中找与 model2 不确定 patch 最相似的
        tree1 = KDTree(f1)
        _, idx2 = tree1.query(f2[d].reshape(1, -1), k=1)
        best_match_from_1 = idx2[0][0]

        # 替换图像 patch（增强一致性）
        new_image_patch_1[i, b] = image_patch_2[i, best_match_from_2]
        new_image_patch_2[i, d] = image_patch_1[i, best_match_from_1]

    # 重新组合图像
    image_patch = torch.cat([new_image_patch_1, new_image_patch_2], dim=0)
    image_patch_last = rearrange(image_patch, 'b (h w) (p1 p2) -> b (h p1) (w p2)',
                                 h=args.h_size, w=args.w_size,
                                 p1=args.patch_size, p2=args.patch_size)
    return image_patch_last



def ABD_R_BCPkdtree(out_max_1, out_max_2, net_input_1, net_input_2, out_1, out_2, args):
    # 拆分 patches
    patches_1 = rearrange(out_max_1, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_2 = rearrange(out_max_2, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    image_patch_1 = rearrange(net_input_1.squeeze(1), 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)
    image_patch_2 = rearrange(net_input_2.squeeze(1), 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)

    patches_mean_1 = torch.mean(patches_1.detach(), dim=2).cpu().numpy()
    patches_mean_2 = torch.mean(patches_2.detach(), dim=2).cpu().numpy()

    patches_outputs_1 = rearrange(out_1, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_outputs_2 = rearrange(out_2, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_mean_outputs_1 = torch.mean(patches_outputs_1.detach(), dim=3).permute(0, 2, 1).cpu().numpy()  # [B, N, C]
    patches_mean_outputs_2 = torch.mean(patches_outputs_2.detach(), dim=3).permute(0, 2, 1).cpu().numpy()

    new_image_patch_1 = image_patch_1.clone()
    new_image_patch_2 = image_patch_2.clone()

    for i in range(args.labeled_bs):
        if random.random() < 0.5:
            # 使用 KDTree 查找最相似的 patch
            tree_2 = KDTree(patches_mean_outputs_2[i])
            b = np.argmin(patches_mean_1[i], axis=0)
            d = np.argmin(patches_mean_2[i], axis=0)

            _, idx_1 = tree_2.query(patches_mean_outputs_1[i][b].reshape(1, -1), k=1)
            best_match_from_2 = idx_1[0][0]

            tree_1 = KDTree(patches_mean_outputs_1[i])
            _, idx_2 = tree_1.query(patches_mean_outputs_2[i][d].reshape(1, -1), k=1)
            best_match_from_1 = idx_2[0][0]

            max_patch_1 = image_patch_2[i][best_match_from_2]
            new_image_patch_1[i][b] = max_patch_1
            max_patch_2 = image_patch_1[i][best_match_from_1]
            new_image_patch_2[i][d] = max_patch_2
        else:
            a = np.argmax(patches_mean_1[i], axis=0)
            b = np.argmin(patches_mean_1[i], axis=0)
            c = np.argmax(patches_mean_2[i], axis=0)
            d = np.argmin(patches_mean_2[i], axis=0)

            max_patch_1 = image_patch_2[i][c]
            new_image_patch_1[i][b] = max_patch_1
            max_patch_2 = image_patch_1[i][a]
            new_image_patch_2[i][d] = max_patch_2

    image_patch = torch.cat([new_image_patch_1, new_image_patch_2], dim=0)
    image_patch_last = rearrange(image_patch, 'b (h w)(p1 p2) -> b (h p1) (w p2)', h=args.h_size, w=args.w_size, p1=args.patch_size, p2=args.patch_size)
    return image_patch_last

def ABD_Ikdtree(outputs1_max, outputs2_max, volume_batch, volume_batch_strong,
                label_batch, label_batch_strong, args):
    # 拆分监督 patches
    patches_supervised_1 = rearrange(
        outputs1_max[:args.labeled_bs],
        'b (h p1) (w p2) -> b (h w) (p1 p2)',
        p1=args.patch_size, p2=args.patch_size
    )
    patches_supervised_2 = rearrange(
        outputs2_max[:args.labeled_bs],
        'b (h p1) (w p2) -> b (h w) (p1 p2)',
        p1=args.patch_size, p2=args.patch_size
    )

    image_patch_supervised_1 = rearrange(
        volume_batch.squeeze(1)[:args.labeled_bs],
        'b (h p1) (w p2) -> b (h w) (p1 p2)',
        p1=args.patch_size, p2=args.patch_size
    )
    image_patch_supervised_2 = rearrange(
        volume_batch_strong.squeeze(1)[:args.labeled_bs],
        'b (h p1) (w p2) -> b (h w) (p1 p2)',
        p1=args.patch_size, p2=args.patch_size
    )

    label_patch_supervised_1 = rearrange(
        label_batch[:args.labeled_bs],
        'b (h p1) (w p2) -> b (h w) (p1 p2)',
        p1=args.patch_size, p2=args.patch_size
    )
    label_patch_supervised_2 = rearrange(
        label_batch_strong[:args.labeled_bs],
        'b (h p1) (w p2) -> b (h w) (p1 p2)',
        p1=args.patch_size, p2=args.patch_size
    )

    # 计算每个 patch 的平均置信度作为特征 [B, N]
    patches_mean_supervised_1 = torch.mean(patches_supervised_1.detach(), dim=2).cpu().numpy()  # [B, N]
    patches_mean_supervised_2 = torch.mean(patches_supervised_2.detach(), dim=2).cpu().numpy()  # [B, N]

    # 克隆用于修改
    new_image_patch_supervised_1 = image_patch_supervised_1.clone()
    new_image_patch_supervised_2 = image_patch_supervised_2.clone()
    new_label_patch_supervised_1 = label_patch_supervised_1.clone()
    new_label_patch_supervised_2 = label_patch_supervised_2.clone()

    for i in range(args.labeled_bs):
        # ✅ 确保只使用第 i 个样本的数据构建 KDTree
        feat_1 = patches_mean_supervised_1[i]  # [N,]
        feat_2 = patches_mean_supervised_2[i]  # [N,]

        # reshape 成 (n_samples, n_features)，这里 n_features = 1（标量）
        feat_1 = feat_1.reshape(-1, 1)  # [N, 1]
        feat_2 = feat_2.reshape(-1, 1)

        if random.random() < 0.5:
            # 构建 tree_2：用 model2 的 patch 特征
            tree_2 = KDTree(feat_2)

            # 找 model1 中最不确定的 patch（置信度最低）
            f = np.argmin(patches_mean_supervised_1[i])  # scalar index
            query_vec = patches_mean_supervised_1[i][f].reshape(1, -1)  # [1, 1]

            _, idx_1 = tree_2.query(query_vec, k=1)
            best_match_from_2 = idx_1[0][0]  # 取最相似 patch 的索引

            # 构建 tree_1：用 model1 的 patch 特征
            tree_1 = KDTree(feat_1)
            _, idx_2 = tree_1.query(feat_2[best_match_from_2].reshape(1, -1), k=1)
            best_match_from_1 = idx_2[0][0]

            # 替换图像 patch
            new_image_patch_supervised_1[i, np.argmax(patches_mean_supervised_1[i])] = \
                image_patch_supervised_2[i, best_match_from_2]

            new_image_patch_supervised_2[i, np.argmin(patches_mean_supervised_1[i])] = \
                image_patch_supervised_1[i, best_match_from_1]

            # 替换 label patch
            new_label_patch_supervised_1[i, np.argmax(patches_mean_supervised_1[i])] = \
                label_patch_supervised_2[i, best_match_from_2]

            new_label_patch_supervised_2[i, np.argmin(patches_mean_supervised_1[i])] = \
                label_patch_supervised_1[i, best_match_from_1]

        else:
            # 随机替换策略（无需 KDTree）
            e = np.argmax(patches_mean_supervised_1[i])
            f = np.argmin(patches_mean_supervised_1[i])
            g = np.argmax(patches_mean_supervised_2[i])
            h = np.argmin(patches_mean_supervised_2[i])

            new_image_patch_supervised_1[i, e] = image_patch_supervised_2[i, h]
            new_image_patch_supervised_2[i, g] = image_patch_supervised_1[i, f]

            new_label_patch_supervised_1[i, e] = label_patch_supervised_2[i, h]
            new_label_patch_supervised_2[i, g] = label_patch_supervised_1[i, f]

    # 重新组合图像和标签
    image_patch_supervised = torch.cat([new_image_patch_supervised_1, new_image_patch_supervised_2], dim=0)
    image_patch_supervised_last = rearrange(
        image_patch_supervised,
        'b (h w) (p1 p2) -> b (h p1) (w p2)',
        h=args.h_size, w=args.w_size, p1=args.patch_size, p2=args.patch_size
    )

    label_patch_supervised = torch.cat([new_label_patch_supervised_1, new_label_patch_supervised_2], dim=0)
    label_patch_supervised_last = rearrange(
        label_patch_supervised,
        'b (h w) (p1 p2) -> b (h p1) (w p2)',
        h=args.h_size, w=args.w_size, p1=args.patch_size, p2=args.patch_size
    )

    return image_patch_supervised_last, label_patch_supervised_last
def ABD_R(outputs1_max, outputs2_max, volume_batch, volume_batch_strong, outputs1_unlabel, outputs2_unlabel, args):
    # ABD-R Bidirectional Displacement Patch
    patches_1 = rearrange(outputs1_max[args.labeled_bs:], 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_2 = rearrange(outputs2_max[args.labeled_bs:], 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    image_patch_1 = rearrange(volume_batch.squeeze(1)[args.labeled_bs:], 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)  # torch.Size([8, 224, 224])
    image_patch_2 = rearrange(volume_batch_strong.squeeze(1)[args.labeled_bs:], 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)
    patches_mean_1 = torch.mean(patches_1.detach(), dim=2)  # torch.Size([8, 16])
    patches_mean_2 = torch.mean(patches_2.detach(), dim=2)

    patches_outputs_1 = rearrange(outputs1_unlabel, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_outputs_2 = rearrange(outputs2_unlabel, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_mean_outputs_1 = torch.mean(patches_outputs_1.detach(), dim=3).permute(0, 2, 1)  # torch.Size([8, 16, 4])
    patches_mean_outputs_2 = torch.mean(patches_outputs_2.detach(), dim=3).permute(0, 2, 1)  # torch.Size([8, 16, 4])

    patches_mean_1_top4_values, patches_mean_1_top4_indices = patches_mean_1.topk(args.top_num, dim=1)  # torch.Size([8, 4])
    patches_mean_2_top4_values, patches_mean_2_top4_indices = patches_mean_2.topk(args.top_num, dim=1)  # torch.Size([8, 4])
    # for i in range(args.labeled_bs):
    for i in range(outputs1_max[args.labeled_bs:].shape[0]):  # 即 range(B_unlabel)
        kl_similarities_1 = torch.empty(args.top_num)
        kl_similarities_2 = torch.empty(args.top_num)
        b = torch.argmin(patches_mean_1[i].detach(), dim=0)
        d = torch.argmin(patches_mean_2[i].detach(), dim=0)
        patches_mean_outputs_min_1 = patches_mean_outputs_1[i, b, :]  # torch.Size([4])
        patches_mean_outputs_min_2 = patches_mean_outputs_2[i, d, :]  # torch.Size([4])
        patches_mean_outputs_top4_1 = patches_mean_outputs_1[i, patches_mean_1_top4_indices[i, :], :]  # torch.Size([4, 4])
        patches_mean_outputs_top4_2 = patches_mean_outputs_2[i, patches_mean_2_top4_indices[i, :], :]  # torch.Size([4, 4])

        for j in range(args.top_num):
            kl_similarities_1[j] = torch.nn.functional.kl_div(patches_mean_outputs_top4_1[j].softmax(dim=-1).log(), patches_mean_outputs_min_2.softmax(dim=-1), reduction='sum')
            kl_similarities_2[j] = torch.nn.functional.kl_div(patches_mean_outputs_top4_2[j].softmax(dim=-1).log(), patches_mean_outputs_min_1.softmax(dim=-1), reduction='sum')

        a = torch.argmin(kl_similarities_1.detach(), dim=0, keepdim=False)
        c = torch.argmin(kl_similarities_2.detach(), dim=0, keepdim=False)
        a_ori = patches_mean_1_top4_indices[i, a]
        c_ori = patches_mean_2_top4_indices[i, c]

        max_patch_1 = image_patch_2[i][c_ori]  
        image_patch_1[i][b] = max_patch_1  
        max_patch_2 = image_patch_1[i][a_ori]
        image_patch_2[i][d] = max_patch_2 

    image_patch = torch.cat([image_patch_1, image_patch_2], dim=0)
    image_patch_last = rearrange(image_patch, 'b (h w)(p1 p2) -> b  (h p1) (w p2)', h=args.h_size, w=args.w_size,p1=args.patch_size, p2=args.patch_size) 
    return image_patch_last

def ABD_R_BCP(out_max_1, out_max_2, net_input_1, net_input_2, out_1, out_2, args):
    patches_1 = rearrange(out_max_1, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_2 = rearrange(out_max_2, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    image_patch_1 = rearrange(net_input_1.squeeze(1), 'b  (h p1) (w p2) -> b (h w)(p1 p2) ',p1=args.patch_size, p2=args.patch_size)  # torch.Size([12, 224, 224])
    image_patch_2 = rearrange(net_input_2.squeeze(1),'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)

    patches_mean_1 = torch.mean(patches_1.detach(), dim=2)
    patches_mean_2 = torch.mean(patches_2.detach(), dim=2)

    patches_outputs_1 = rearrange(out_1, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_outputs_2 = rearrange(out_2, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_mean_outputs_1 = torch.mean(patches_outputs_1.detach(), dim=3).permute(0, 2, 1)  # torch.Size([8, 16, 4])
    patches_mean_outputs_2 = torch.mean(patches_outputs_2.detach(), dim=3).permute(0, 2, 1)  # torch.Size([8, 16, 4])

    patches_mean_1_top4_values, patches_mean_1_top4_indices = patches_mean_1.topk(args.top_num, dim=1)  # torch.Size([8, 4])
    patches_mean_2_top4_values, patches_mean_2_top4_indices = patches_mean_2.topk(args.top_num, dim=1)  # torch.Size([8, 4])

    for i in range(args.labeled_bs):
        if random.random() < 0.5:
            kl_similarities_1 = torch.empty(args.top_num)
            kl_similarities_2 = torch.empty(args.top_num)
            b = torch.argmin(patches_mean_1[i].detach(), dim=0)
            d = torch.argmin(patches_mean_2[i].detach(), dim=0)
            patches_mean_outputs_min_1 = patches_mean_outputs_1[i, b, :]  # torch.Size([4])
            patches_mean_outputs_min_2 = patches_mean_outputs_2[i, d, :]  # torch.Size([4])

            patches_mean_outputs_top4_1 = patches_mean_outputs_1[i, patches_mean_1_top4_indices[i, :], :]  # torch.Size([4, 4])
            patches_mean_outputs_top4_2 = patches_mean_outputs_2[i, patches_mean_2_top4_indices[i, :], :]  # torch.Size([4, 4])

            for j in range(args.top_num):
                kl_similarities_1[j] = torch.nn.functional.kl_div(patches_mean_outputs_top4_1[j].softmax(dim=-1).log(), patches_mean_outputs_min_2.softmax(dim=-1), reduction='sum')
                kl_similarities_2[j] = torch.nn.functional.kl_div(patches_mean_outputs_top4_2[j].softmax(dim=-1).log(), patches_mean_outputs_min_1.softmax(dim=-1), reduction='sum')

            a = torch.argmin(kl_similarities_1.detach(), dim=0, keepdim=False)
            c = torch.argmin(kl_similarities_2.detach(), dim=0, keepdim=False)

            a_ori = patches_mean_1_top4_indices[i, a]
            c_ori = patches_mean_2_top4_indices[i, c]

            max_patch_1 = image_patch_2[i][c_ori]
            image_patch_1[i][b] = max_patch_1
            max_patch_2 = image_patch_1[i][a_ori]
            image_patch_2[i][d] = max_patch_2
        else:
            a = torch.argmax(patches_mean_1[i].detach(), dim=0)
            b = torch.argmin(patches_mean_1[i].detach(), dim=0)
            c = torch.argmax(patches_mean_2[i].detach(), dim=0)
            d = torch.argmin(patches_mean_2[i].detach(), dim=0)

            max_patch_1 = image_patch_2[i][c]
            image_patch_1[i][b] = max_patch_1
            max_patch_2 = image_patch_1[i][a]
            image_patch_2[i][d] = max_patch_2
    image_patch = torch.cat([image_patch_1, image_patch_2], dim=0)
    image_patch_last = rearrange(image_patch, 'b (h w)(p1 p2) -> b  (h p1) (w p2)', h=args.h_size, w=args.w_size,p1=args.patch_size, p2=args.patch_size)  # torch.Size([24, 224, 224])
    return image_patch_last

def ABD_R_BCPkd(out_max_1, out_max_2, net_input_1, net_input_2, out_1, out_2, args):
    patches_1 = rearrange(out_max_1, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_2 = rearrange(out_max_2, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    image_patch_1 = rearrange(net_input_1.squeeze(1), 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size,
                              p2=args.patch_size)
    image_patch_2 = rearrange(net_input_2.squeeze(1), 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size,
                              p2=args.patch_size)

    patches_mean_1 = torch.mean(patches_1.detach(), dim=2)
    patches_mean_2 = torch.mean(patches_2.detach(), dim=2)

    patches_outputs_1 = rearrange(out_1, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_outputs_2 = rearrange(out_2, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_mean_outputs_1 = torch.mean(patches_outputs_1.detach(), dim=3).permute(0, 2, 1)
    patches_mean_outputs_2 = torch.mean(patches_outputs_2.detach(), dim=3).permute(0, 2, 1)

    patches_mean_1_top4_values, patches_mean_1_top4_indices = patches_mean_1.topk(args.top_num, dim=1)
    patches_mean_2_top4_values, patches_mean_2_top4_indices = patches_mean_2.topk(args.top_num, dim=1)

    for i in range(args.labeled_bs):
        rand_val = random.random()

        if rand_val < 0.5:  # KDTree方法
            # 构建两个图像的patch特征KDTree
            features_1 = patches_mean_outputs_1[i].detach().cpu().numpy()
            features_2 = patches_mean_outputs_2[i].detach().cpu().numpy()

            # 为图像2构建KDTree
            tree_2 = KDTree(features_2)

            # 找到图像1中最弱的patch（均值最小）
            b = torch.argmin(patches_mean_1[i].detach(), dim=0).item()
            weakest_feature_1 = features_1[b].reshape(1, -1)

            # 在图像2中寻找与图像1最弱patch最相似的patch
            dist_2, idx_2 = tree_2.query(weakest_feature_1, k=1)
            c_ori = idx_2[0][0]  # 图像2中最相似的patch索引

            # 为图像1构建KDTree
            tree_1 = KDTree(features_1)

            # 找到图像2中最弱的patch（均值最小）
            d = torch.argmin(patches_mean_2[i].detach(), dim=0).item()
            weakest_feature_2 = features_2[d].reshape(1, -1)

            # 在图像1中寻找与图像2最弱patch最相似的patch
            dist_1, idx_1 = tree_1.query(weakest_feature_2, k=1)
            a_ori = idx_1[0][0]  # 图像1中最相似的patch索引

            # 交换patch
            max_patch_1 = image_patch_2[i][c_ori].clone()
            image_patch_1[i][b] = max_patch_1
            max_patch_2 = image_patch_1[i][a_ori].clone()
            image_patch_2[i][d] = max_patch_2

        # elif rand_val < 0.3:  # KL散度方法
        else:
            kl_similarities_1 = torch.empty(args.top_num)
            kl_similarities_2 = torch.empty(args.top_num)
            b = torch.argmin(patches_mean_1[i].detach(), dim=0)
            d = torch.argmin(patches_mean_2[i].detach(), dim=0)
            patches_mean_outputs_min_1 = patches_mean_outputs_1[i, b, :]
            patches_mean_outputs_min_2 = patches_mean_outputs_2[i, d, :]

            patches_mean_outputs_top4_1 = patches_mean_outputs_1[i, patches_mean_1_top4_indices[i, :], :]
            patches_mean_outputs_top4_2 = patches_mean_outputs_2[i, patches_mean_2_top4_indices[i, :], :]

            for j in range(args.top_num):
                kl_similarities_1[j] = torch.nn.functional.kl_div(
                    patches_mean_outputs_top4_1[j].softmax(dim=-1).log(),
                    patches_mean_outputs_min_2.softmax(dim=-1),
                    reduction='sum'
                )
                kl_similarities_2[j] = torch.nn.functional.kl_div(
                    patches_mean_outputs_top4_2[j].softmax(dim=-1).log(),
                    patches_mean_outputs_min_1.softmax(dim=-1),
                    reduction='sum'
                )

            a = torch.argmin(kl_similarities_1.detach(), dim=0, keepdim=False)
            c = torch.argmin(kl_similarities_2.detach(), dim=0, keepdim=False)

            a_ori = patches_mean_1_top4_indices[i, a]
            c_ori = patches_mean_2_top4_indices[i, c]

            max_patch_1 = image_patch_2[i][c_ori].clone()
            image_patch_1[i][b] = max_patch_1
            max_patch_2 = image_patch_1[i][a_ori].clone()
            image_patch_2[i][d] = max_patch_2
        #
        # else:  # 极值交换方法
        # a = torch.argmax(patches_mean_1[i].detach(), dim=0)
        # b = torch.argmin(patches_mean_1[i].detach(), dim=0)
        # c = torch.argmax(patches_mean_2[i].detach(), dim=0)
        # d = torch.argmin(patches_mean_2[i].detach(), dim=0)
        #
        # max_patch_1 = image_patch_2[i][c].clone()
        # image_patch_1[i][b] = max_patch_1
        # max_patch_2 = image_patch_1[i][a].clone()
        # image_patch_2[i][d] = max_patch_2

    image_patch = torch.cat([image_patch_1, image_patch_2], dim=0)
    image_patch_last = rearrange(image_patch, 'b (h w)(p1 p2) -> b  (h p1) (w p2)',
                                 h=args.h_size, w=args.w_size, p1=args.patch_size, p2=args.patch_size)
    return image_patch_last

def ABD_R_BCP3(out_max_1, out_max_2, net_input_1, net_input_2, out_1, out_2, args):
    patches_1 = rearrange(out_max_1, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_2 = rearrange(out_max_2, 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    image_patch_1 = rearrange(net_input_1.squeeze(1), 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size,
                              p2=args.patch_size)
    image_patch_2 = rearrange(net_input_2.squeeze(1), 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size,
                              p2=args.patch_size)

    patches_mean_1 = torch.mean(patches_1.detach(), dim=2)
    patches_mean_2 = torch.mean(patches_2.detach(), dim=2)

    patches_outputs_1 = rearrange(out_1, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_outputs_2 = rearrange(out_2, 'b c (h p1) (w p2)->b c (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_mean_outputs_1 = torch.mean(patches_outputs_1.detach(), dim=3).permute(0, 2, 1)
    patches_mean_outputs_2 = torch.mean(patches_outputs_2.detach(), dim=3).permute(0, 2, 1)

    patches_mean_1_top4_values, patches_mean_1_top4_indices = patches_mean_1.topk(args.top_num, dim=1)
    patches_mean_2_top4_values, patches_mean_2_top4_indices = patches_mean_2.topk(args.top_num, dim=1)

    for i in range(args.labeled_bs):
        rand_val = random.random()
# 0.2
        if rand_val < 0.2:  # KDTree方法
            # 构建两个图像的patch特征KDTree
            features_1 = patches_mean_outputs_1[i].detach().cpu().numpy()
            features_2 = patches_mean_outputs_2[i].detach().cpu().numpy()

            # 为图像2构建KDTree
            tree_2 = KDTree(features_2)

            # 找到图像1中最弱的patch（均值最小）
            b = torch.argmin(patches_mean_1[i].detach(), dim=0).item()
            weakest_feature_1 = features_1[b].reshape(1, -1)

            # 在图像2中寻找与图像1最弱patch最相似的patch
            dist_2, idx_2 = tree_2.query(weakest_feature_1, k=1)
            c_ori = idx_2[0][0]  # 图像2中最相似的patch索引

            # 为图像1构建KDTree
            tree_1 = KDTree(features_1)

            # 找到图像2中最弱的patch（均值最小）
            d = torch.argmin(patches_mean_2[i].detach(), dim=0).item()
            weakest_feature_2 = features_2[d].reshape(1, -1)

            # 在图像1中寻找与图像2最弱patch最相似的patch
            dist_1, idx_1 = tree_1.query(weakest_feature_2, k=1)
            a_ori = idx_1[0][0]  # 图像1中最相似的patch索引

            # 交换patch
            max_patch_1 = image_patch_2[i][c_ori].clone()
            image_patch_1[i][b] = max_patch_1
            max_patch_2 = image_patch_1[i][a_ori].clone()
            image_patch_2[i][d] = max_patch_2
        #
        elif rand_val < 0.3:  # KL散度方法
            kl_similarities_1 = torch.empty(args.top_num)
            kl_similarities_2 = torch.empty(args.top_num)
            b = torch.argmin(patches_mean_1[i].detach(), dim=0)
            d = torch.argmin(patches_mean_2[i].detach(), dim=0)
            patches_mean_outputs_min_1 = patches_mean_outputs_1[i, b, :]
            patches_mean_outputs_min_2 = patches_mean_outputs_2[i, d, :]

            patches_mean_outputs_top4_1 = patches_mean_outputs_1[i, patches_mean_1_top4_indices[i, :], :]
            patches_mean_outputs_top4_2 = patches_mean_outputs_2[i, patches_mean_2_top4_indices[i, :], :]

            for j in range(args.top_num):
                kl_similarities_1[j] = torch.nn.functional.kl_div(
                    patches_mean_outputs_top4_1[j].softmax(dim=-1).log(),
                    patches_mean_outputs_min_2.softmax(dim=-1),
                    reduction='sum'
                )
                kl_similarities_2[j] = torch.nn.functional.kl_div(
                    patches_mean_outputs_top4_2[j].softmax(dim=-1).log(),
                    patches_mean_outputs_min_1.softmax(dim=-1),
                    reduction='sum'
                )

            a = torch.argmin(kl_similarities_1.detach(), dim=0, keepdim=False)
            c = torch.argmin(kl_similarities_2.detach(), dim=0, keepdim=False)

            a_ori = patches_mean_1_top4_indices[i, a]
            c_ori = patches_mean_2_top4_indices[i, c]

            max_patch_1 = image_patch_2[i][c_ori].clone()
            image_patch_1[i][b] = max_patch_1
            max_patch_2 = image_patch_1[i][a_ori].clone()
            image_patch_2[i][d] = max_patch_2
        #
        else:  # 极值交换方法
            a = torch.argmax(patches_mean_1[i].detach(), dim=0)
            b = torch.argmin(patches_mean_1[i].detach(), dim=0)
            c = torch.argmax(patches_mean_2[i].detach(), dim=0)
            d = torch.argmin(patches_mean_2[i].detach(), dim=0)

            max_patch_1 = image_patch_2[i][c].clone()
            image_patch_1[i][b] = max_patch_1
            max_patch_2 = image_patch_1[i][a].clone()
            image_patch_2[i][d] = max_patch_2

    image_patch = torch.cat([image_patch_1, image_patch_2], dim=0)
    image_patch_last = rearrange(image_patch, 'b (h w)(p1 p2) -> b  (h p1) (w p2)',
                                 h=args.h_size, w=args.w_size, p1=args.patch_size, p2=args.patch_size)
    return image_patch_last

def ABD_I(outputs1_max, outputs2_max, volume_batch, volume_batch_strong, label_batch, label_batch_strong, args):
    # ABD-I Bidirectional Displacement Patch
    patches_supervised_1 = rearrange(outputs1_max[:args.labeled_bs], 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    patches_supervised_2 = rearrange(outputs2_max[:args.labeled_bs], 'b (h p1) (w p2)->b (h w) (p1 p2)', p1=args.patch_size, p2=args.patch_size)
    image_patch_supervised_1 = rearrange(volume_batch.squeeze(1)[:args.labeled_bs], 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)  # torch.Size([8, 224, 224])
    image_patch_supervised_2 = rearrange(volume_batch_strong.squeeze(1)[:args.labeled_bs], 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)
    label_patch_supervised_1 = rearrange(label_batch[:args.labeled_bs], 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)
    label_patch_supervised_2 = rearrange(label_batch_strong[:args.labeled_bs], 'b  (h p1) (w p2) -> b (h w)(p1 p2) ', p1=args.patch_size, p2=args.patch_size)
    patches_mean_supervised_1 = torch.mean(patches_supervised_1.detach(), dim=2)
    patches_mean_supervised_2 = torch.mean(patches_supervised_2.detach(), dim=2)
    e = torch.argmax(patches_mean_supervised_1.detach(), dim=1)
    f = torch.argmin(patches_mean_supervised_1.detach(), dim=1)
    g = torch.argmax(patches_mean_supervised_2.detach(), dim=1)
    h = torch.argmin(patches_mean_supervised_2.detach(), dim=1)
    for i in range(args.labeled_bs): 
        if random.random() < 0.5:
            min_patch_supervised_1 = image_patch_supervised_2[i][h[i]]  
            image_patch_supervised_1[i][e[i]] = min_patch_supervised_1
            min_patch_supervised_2 = image_patch_supervised_1[i][f[i]]
            image_patch_supervised_2[i][g[i]] = min_patch_supervised_2

            min_label_supervised_1 = label_patch_supervised_2[i][h[i]]
            label_patch_supervised_1[i][e[i]] = min_label_supervised_1
            min_label_supervised_2 = label_patch_supervised_1[i][f[i]]
            label_patch_supervised_2[i][g[i]] = min_label_supervised_2
    image_patch_supervised = torch.cat([image_patch_supervised_1, image_patch_supervised_2], dim=0)
    image_patch_supervised_last = rearrange(image_patch_supervised, 'b (h w)(p1 p2) -> b  (h p1) (w p2)', h=args.h_size, w=args.w_size,p1=args.patch_size, p2=args.patch_size)  # torch.Size([16, 224, 224])
    label_patch_supervised = torch.cat([label_patch_supervised_1, label_patch_supervised_2], dim=0)
    label_patch_supervised_last = rearrange(label_patch_supervised, 'b (h w)(p1 p2) -> b  (h p1) (w p2)', h=args.h_size, w=args.w_size,p1=args.patch_size, p2=args.patch_size)  # torch.Size([16, 224, 224])
    return image_patch_supervised_last, label_patch_supervised_last


def get_low_freq_component_single(image_tensor, threshold):
    """
    获取单张图像的低频区域并转换回空间域
    :param image_tensor: 输入的图像张量，形状为 (1, H, W)
    :param threshold: 阈值，用于生成低频掩码
    :return: 低频区域的空间域表示，形状为 (1, H, W)
    """
    # 获取图像的形状
    _, _, rows, cols = image_tensor.shape

    # 转换到频域
    f_transform = torch.fft.fft2(image_tensor, dim=(-2, -1))
    f_shift = torch.fft.fftshift(f_transform, dim=(-2, -1))

    # 生成低频掩码
    crow, ccol = rows // 2, cols // 2
    mask = torch.zeros((rows, cols), dtype=torch.float32, device=image_tensor.device)
    mask[crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = 1

    # 应用掩码并进行逆傅里叶变换，得到低频部分的图像
    f_low_freq = f_shift * mask
    f_ishift = torch.fft.ifftshift(f_low_freq, dim=(-2, -1))
    img_back = torch.abs(torch.fft.ifft2(f_ishift, dim=(-2, -1)))

    return img_back.unsqueeze(0)


def process_image_batches(image_batch_with_labels_tensor, image_batch_without_labels_tensor, threshold):
    """
    处理多个图像批次，提取并交换每个批次第 i 张图像的低频区域
    :param image_batch_with_labels_tensor: 带标签的图像张量（批量），形状为 (N, 1, H, W)
    :param image_batch_without_labels_tensor: 不带标签的图像张量（批量），形状为 (N, 1, H, W)
    :param threshold: 阈值，决定低频区域的大小
    :return: 互换低频区域后的两个图像张量（批量）
    """
    # 确保两个批次具有相同的大小
    assert image_batch_with_labels_tensor.shape[0] == image_batch_without_labels_tensor.shape[0], \
        "两个批次的大小必须相同"

    batch_size = image_batch_with_labels_tensor.shape[0]

    new_batch_with_labels = []
    new_batch_without_labels = []

    # 对每张图像进行处理
    for i in range(batch_size):
        # 提取第 i 张图像
        img_with_labels = image_batch_with_labels_tensor[i:i + 1]
        img_without_labels = image_batch_without_labels_tensor[i:i + 1]

        # print("img_with_labels大小",img_with_labels.shape)

        # 计算低频区域
        low_freq_with_labels = get_low_freq_component_single(img_with_labels, threshold)
        low_freq_without_labels = get_low_freq_component_single(img_without_labels, threshold)

        # 交换低频区域
        swapped_img_with_labels = img_with_labels - low_freq_with_labels + low_freq_without_labels
        swapped_img_without_labels = img_without_labels - low_freq_without_labels + low_freq_with_labels

        # 将处理后的图像加入新的批次中
        new_batch_with_labels.append(swapped_img_with_labels)
        new_batch_without_labels.append(swapped_img_without_labels)

    # 将列表转换为张量
    new_batch_with_labels = torch.cat(new_batch_with_labels, dim=0)
    new_batch_without_labels = torch.cat(new_batch_without_labels, dim=0)
    last_image = torch.cat([new_batch_with_labels, new_batch_without_labels], dim=0)
    return last_image.view(-1, 256, 256)

