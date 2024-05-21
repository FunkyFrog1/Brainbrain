import torch
import torch.nn.functional as F
# 设置随机种子以确保结果的可重复性

# 创建一个16x16的随机矩阵，模拟一批次的数据
data = torch.randn(4, 768).half()
data2 = torch.randn(4, 768).half()

# 计算相似矩阵
similarity_matrix = torch.matmul(data, data2.t()).half()
print(similarity_matrix)

weight_i2t = F.softmax(similarity_matrix, dim = 1)
weight_t2i = F.softmax(similarity_matrix.t(), dim = 1)

weight_i2t.fill_diagonal_(0)
weight_t2i.fill_diagonal_(0)

print(weight_t2i)

image_embed_neg = []

for b in range(4):
    neg_idx = torch.multinomial(weight_t2i[b], 1).item()
    image_embed_neg.append(data2[neg_idx])

    print(neg_idx)

image_embed_neg = torch.stack(image_embed_neg, dim=0)


print(image_embed_neg.shape)

for i in range(4):
    F.cross_entropy(image_embed_neg[i], data[i])
