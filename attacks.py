import torch
import torch.nn.functional as F
import pdb

def flag(model_forward, perturb_shape, y, args, optimizer, device, criterion) :
    model, forward = model_forward
    model.train()
    optimizer.zero_grad()

    perturb = torch.FloatTensor(*perturb_shape).uniform_(-args.step_size, args.step_size).to(device) # 在括号里的均匀分布中取值并重新赋值
    # pert初始化为和X形状相同、服从(-alpha, alpha)均匀分布的矩阵
    perturb.requires_grad_()    # pert带有梯度
    out = forward(perturb)  # 为输入数据增加对抗性扰动pert
    loss = criterion(out, y)
    loss /= args.m  # 因为loss的梯度一直是累加的，所以每个step贡献1/M的grad值
    # 每个epoch分为M个step，M个loss的grad进行累加，得到最终的loss
    for _ in range(args.m-1):
        loss.backward()
        perturb_data = perturb.detach() + args.step_size * torch.sign(perturb.grad.detach())
        perturb.data = perturb_data.data
        perturb.grad[:] = 0  # pert梯度grad清零
        # 重复对抗性扰动的训练过程
        out = forward(perturb)
        loss = criterion(out, y)
        loss /= args.m
    # 通过M个step累加的grad，更新model的参数
    loss.backward()
    optimizer.step()

    return loss, out

def flag_biased(model_forward, perturb_shape, y, args, optimizer, device, criterion, training_idx) :
    unlabel_idx = list(set(range(perturb_shape[0])) - set(training_idx))

    model, forward = model_forward
    model.train()
    optimizer.zero_grad()

    perturb = torch.FloatTensor(*perturb_shape).uniform_(-args.step_size, args.step_size).to(device)
    perturb.data[unlabel_idx] *= args.amp
    perturb.requires_grad_()
    out = forward(perturb)
    loss = criterion(out, y)
    loss /= args.m

    for _ in range(args.m-1):
        loss.backward()

        perturb_data_training = perturb[training_idx].detach() + args.step_size * torch.sign(perturb.grad[training_idx].detach())
        perturb.data[training_idx] = perturb_data_training.data

        perturb_data_unlabel = perturb[unlabel_idx].detach() + args.amp*args.step_size * torch.sign(perturb.grad[unlabel_idx].detach())
        perturb.data[unlabel_idx] = perturb_data_unlabel.data

        perturb.grad[:] = 0
        out = forward(perturb)
        loss = criterion(out, y)
        loss /= args.m

    loss.backward()
    optimizer.step()

    return loss, out
