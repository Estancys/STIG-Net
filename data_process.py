# -*- coding: utf-8 -*-
import argparse
import os
import platform

import torch
from torch_geometric.data import Data, Dataset
from tqdm import tqdm

from alphapose.models import builder
from utils.config import update_config
from utils.detector import DetectionLoader
from utils.pPose_nms import pose_nms
from utils.transforms import get_func_heatmap_to_coord
from apis import get_detector

parser = argparse.ArgumentParser(description='AlphaPose Demo')
parser.add_argument('--cfg', type=str, required=False,
                    default='G:/gcn/GNNcv/configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml',
                    help='experiment configure file name')
parser.add_argument('--checkpoint', type=str, required=False,
                    default='G:/gcn/GNNcv/pretrained_models/fast_res50_256x192.pth',
                    help='checkpoint file name')
parser.add_argument('--sp', default=False, action='store_true',
                    help='Use single process for pytorch')
parser.add_argument('--detector', dest='detector',
                    help='detector name', default="yolo")

parser.add_argument('--outdir', dest='outputpath',
                    help='output-directory', default="G:/gcn/GNNcv/ouput")
parser.add_argument('--save_img', default=False, action='store_true',
                    help='save result as image')
parser.add_argument('--vis', default=False, action='store_true',
                    help='visualize image')
parser.add_argument('--showbox', default=False, action='store_true',
                    help='visualize human bbox')
parser.add_argument('--profile', default=False, action='store_true',
                    help='add speed profiling at screen output')
parser.add_argument('--format', type=str,
                    help='save in the format of cmu or coco or openpose, option: coco/cmu/open')
parser.add_argument('--min_box_area', type=int, default=0,
                    help='min box area to filter out')
parser.add_argument('--detbatch', type=int, default=5,
                    help='detection batch size PER GPU')
parser.add_argument('--posebatch', type=int, default=64,
                    help='pose estimation maximum batch size PER GPU')
parser.add_argument('--eval', dest='eval', default=False, action='store_true',
                    help='save the result json as coco format, using image index(int) instead of image name(str)')
parser.add_argument('--gpus', type=str, dest='gpus', default="0",
                    help='choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)')
parser.add_argument('--qsize', type=int, dest='qsize', default=1024,
                    help='the length of result buffer, where reducing it will lower requirement of cpu memory')

"""----------------------------- Video options -----------------------------"""
parser.add_argument('--video', dest='video',
                    help='video-name', default="")
parser.add_argument('--webcam', dest='webcam', type=int,
                    help='webcam number', default=-1)
parser.add_argument('--save_video', dest='save_video',
                    help='whether to save rendered video', default=False, action='store_true')
parser.add_argument('--vis_fast', dest='vis_fast',
                    help='use fast rendering', action='store_true', default=False)
parser.add_argument('--pose_flow', dest='pose_flow',
                    help='track humans in video with PoseFlow', action='store_true', default=False)
parser.add_argument('--pose_track', dest='pose_track',
                    help='track humans in video with reid', action='store_true', default=False)
args = parser.parse_args()
cfg = update_config(args.cfg)

if platform.system() == 'Windows':
    args.sp = True

args.gpus = [int(i) for i in args.gpus.split(',')] if torch.cuda.device_count() >= 1 else [-1]
args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")

args.detbatch = args.detbatch * len(args.gpus)
args.posebatch = args.posebatch * len(args.gpus)
args.tracking = args.pose_track or args.pose_flow or args.detector == 'tracker'

if not args.sp:
    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')


def check_input():
    # for video
    if len(args.video):
        if os.path.isfile(args.video):
            videofile = args.video
            return 'video', videofile
        else:
            raise IOError('Error: --video must refer to a video file, not directory.')


def print_finish_info():
    print('===========================> Finish Model Running.')
    if (args.save_img or args.save_video) and not args.vis_fast:
        print('===========================> Rendering remaining images in the queue...')
        print('===========================> If this step takes too long, you can enable the --vis_fast flag to use '
              'fast rendering (real-time).')


def loop():
    n = 0
    while True:
        yield n
        n += 1


def mix(a, b, repeat, totensor):
    num_repeat = a[0].shape[0]
    leng = len(a)
    if repeat:
        combined = [torch.cat([a[i], b[i].repeat(num_repeat).unsqueeze(1)], dim=1) for i in range(leng)]  # b是高维的
    else:
        combined = [torch.cat([a[i], b[i]], dim=1) for i in range(leng)]
    # Convert the combined list to a tensor
    if totensor:
        result = torch.stack(combined)
    else:
        result = combined
    # print(result)
    # Print the shape of the resulting tensor
    return result


def build_edge_index(features):
    n = int(features.shape[0] / 18)
    a = 4  # 可攻击的肢体点 4
    b = 6  # 人体可受击打的部位 6
    for i in range(n):  # 人数
        # 人体可受击打的部位，6个点
        col_hand_foot_i = torch.tensor([18 * i + 0, 18 * i + 5, 18 * i + 6, 18 * i + 17, 18 * i + 11, 18 * i + 12]).to(
            torch.long)
        col_hand_foot_i = torch.unsqueeze(col_hand_foot_i, 0)
        if i == 0:
            col_hand_foot = col_hand_foot_i
        else:
            col_hand_foot = torch.cat((col_hand_foot, col_hand_foot_i), dim=1)
    col_hand_foot = col_hand_foot.repeat(1, a * n)  # a代表可以攻击的手足部点数
    for i in range(n):  # 个体自连接。共15条边
        row_person_link = [18 * i + 0, 18 * i + 0, 18 * i + 1, 18 * i + 2, 18 * i + 5, 18 * i + 5, 18 * i + 7,
                           18 * i + 6, 18 * i + 8, 18 * i + 17, 18 * i + 17, 18 * i + 11, 18 * i + 12, 18 * i + 13,
                           18 * i + 14]
        col_person_link = [18 * i + 1, 18 * i + 2, 18 * i + 3, 18 * i + 4, 18 * i + 6, 18 * i + 7, 18 * i + 9,
                           18 * i + 8, 18 * i + 10, 18 * i + 11, 18 * i + 12, 18 * i + 13, 18 * i + 14, 18 * i + 15,
                           18 * i + 16]
        row_i = torch.tensor(row_person_link).to(torch.long)  # 每个人体的自然连接
        col_i = torch.tensor(col_person_link).to(torch.long)
        row_i = torch.unsqueeze(row_i, 0)
        col_i = torch.unsqueeze(col_i, 0)

        # 可攻击的肢体点 4
        row_hand_foot_i = torch.tensor([18 * i + 9, 18 * i + 10, 18 * i + 15, 18 * i + 16]).to(torch.long)
        row_hand_foot_i = torch.unsqueeze(row_hand_foot_i, 0)

        if i == 0:
            row = row_i
            col = col_i
            row_hand_foot = row_hand_foot_i
        else:
            row = torch.cat((row, row_i), dim=1)
            col = torch.cat((col, col_i), dim=1)
            row_hand_foot = torch.cat((row_hand_foot, row_hand_foot_i), dim=1)  # 一帧中的人体关键点排列
    row_hand_foot = row_hand_foot.repeat_interleave(b * n)
    row_hand_foot = torch.unsqueeze(row_hand_foot, dim=0)
    self_link = row.shape[1]
    row = torch.cat((row, row_hand_foot), dim=1)
    col = torch.cat((col, col_hand_foot), dim=1)
    row = row.squeeze()
    col = col.squeeze()
    # same_elements = torch.eq(row, col)
    # indices = torch.nonzero(same_elements)    # 检查是否存在自连接
    index = torch.stack([row, col], dim=0)
    return index, self_link


def build_edge_attr(features, edge_index):
    positions = features[:, -2:]
    p1 = positions[edge_index[0]]
    p2 = positions[edge_index[1]]
    distances = torch.pairwise_distance(p1, p2)
    value = 1 / distances
    # distances = torch.cdist(positions[edge_index[0]], positions[edge_index[1]]) + torch.eye(features.shape[0])
    # value = 1 / distances - torch.eye(features.shape[0])
    # z = F.normalize(value)
    return value


def select_point(keypoints, k):
    # 计算所有人的第17个点的位置距离
    indexes = []
    # 这里记录的index是每次删除元素后重新编排的index列表，
    # 与外面删除分数时的方式保持一致，因此删除列表的序号是含有顺序因素的，不代表实际的序号
    person_num = len(keypoints)
    half_length = person_num // 2 + 1
    if half_length > k:
        half_length = k
    # 循环删除最小的张量，直到列表长度大于一半
    while len(keypoints) > half_length:
        D = []
        key = torch.stack(keypoints, dim=0)
        # 以下操作仅为了筛选列表keypoints中的值
        points = key[:, 17, :]
        for i in range(len(keypoints)):
            point1 = points[i, :]  # 取要检测的元素
            distances = 0
            for j in range(len(keypoints)):
                if j != i:  # 取其他元素
                    point2 = points[j, :]
                    distance = torch.norm(point1 - point2, p=2, dim=0)
                    distances += distance
                else:
                    continue
            distances /= (len(keypoints) - 1)  # 计算每个点到其他点的平均距离
            D.append(distances)  # 将所有点的平均距离加入到列表
        max_tensor = max(D)  # 找到列表中最大的平均距离
        index = D.index(max_tensor)
        indexes.append(index)
        D.remove(max_tensor)
        keypoints.pop(index)  # 删除最远的一个人,pop只能对列表进行操作，因此这里仅删除传入的keypoints列表
    return keypoints, indexes


def zero_feature():
    feature = torch.tensor([[0, 0, 0, 0]])


def data_process(file_path, frame):
    args.video = file_path
    path = file_path.split("/")[-1]
    if path.startswith("fi"):
        y = torch.tensor([1], dtype=torch.long)
    elif path.startswith("no"):
        y = torch.tensor([0], dtype=torch.long)
    else:
        raise IOError('Error: Static label applied poorly or missing')
    mode, input_source = check_input()

    if not os.path.exists(args.outputpath):
        os.makedirs(args.outputpath)

    # Load detection loader

    det_loader = DetectionLoader(input_source, get_detector(args), cfg, args, batchSize=args.detbatch, mode=mode,
                                 queueSize=args.qsize)
    det_worker = det_loader.start()
    # Load pose model
    pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

    print('Loading pose model from %s...' % (args.video,))
    pose_model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    pose_dataset = builder.retrieve_dataset(cfg.DATASET.TRAIN)

    if len(args.gpus) > 1:
        pose_model = torch.nn.DataParallel(pose_model, device_ids=args.gpus).to(args.device)
    else:
        pose_model.to(args.device)
    pose_model.eval()

    runtime_profile = {
        'dt': [],
        'pt': [],
        'pn': []
    }
    # Init data writer
    EVAL_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    queueSize = args.qsize
    graph_data_list = []
    if args.save_video:
        from alphapose.utils.writer import DEFAULT_VIDEO_SAVE_OPT as video_save_opt, DataWriter

        if mode == 'video':
            video_save_opt['savepath'] = os.path.join(args.outputpath, 'AlphaPose_' + os.path.basename(input_source))
        else:
            video_save_opt['savepath'] = os.path.join(args.outputpath, 'AlphaPose_webcam' + str(input_source) + '.mp4')
        video_save_opt.update(det_loader.videoinfo)
        writer = DataWriter(cfg, args, save_video=True, video_save_opt=video_save_opt, queueSize=queueSize).start()

    data_len = det_loader.length
    im_names_desc = tqdm(range(data_len), dynamic_ncols=True)
    batchSize = args.posebatch
    node_features = []
    for i in im_names_desc:
        with torch.no_grad():
            (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = det_loader.read()
            if orig_img is None:
                break
            if boxes is not None and boxes.nelement() != 0:
                # writer.save(None, None, None, None, None, orig_img, im_name)
                # continue
                # Pose Estimation
                inps = inps.to(args.device)
                datalen = inps.size(0)
                leftover = 0
                if (datalen) % batchSize:
                    leftover = 1
                num_batches = datalen // batchSize + leftover
                hm = []
                for j in range(num_batches):
                    inps_j = inps[j * batchSize:min((j + 1) * batchSize, datalen)]
                    hm_j = pose_model(inps_j)
                    hm.append(hm_j)
                hm = torch.cat(hm)
                hm = hm.cpu()
                pose_coords = []
                pose_scores = []
                heatmap_to_coord = get_func_heatmap_to_coord(cfg)
                for hm_i in range(hm.shape[0]):
                    bbox = cropped_boxes[hm_i].tolist()
                    pose_coord, pose_score = heatmap_to_coord(hm[hm_i][EVAL_JOINTS], bbox,
                                                              hm_shape=cfg.DATA_PRESET.HEATMAP_SIZE,
                                                              norm_type=cfg.LOSS.get('NORM_TYPE', None))
                    pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
                    pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))
                preds_img = torch.cat(pose_coords)
                preds_scores = torch.cat(pose_scores)
                for j in range(preds_img.shape[0]):
                    p_t = torch.cat((preds_img[j], torch.unsqueeze((preds_img[j][5, :] + preds_img[j][6, :]) / 2, 0)))
                    p_s = torch.cat(
                        (preds_scores[j], torch.unsqueeze((preds_scores[j][5, :] + preds_scores[j][6, :]) / 2, 0)))
                    p_t = torch.unsqueeze(p_t, 0)
                    p_s = torch.unsqueeze(p_s, 0)
                    if j == 0:
                        p_img = p_t
                        p_scores = p_s
                    else:
                        p_img = torch.cat([p_img, p_t], dim=0)
                        p_scores = torch.cat([p_scores, p_s], dim=0)
                preds_img = p_img
                preds_scores = p_scores
                boxes, scores, ids, preds_img, preds_scores, pick_ids = \
                    pose_nms(boxes, scores, ids, preds_img, preds_scores, args.min_box_area,
                             use_heatmap_loss=cfg.DATA_PRESET.get('LOSS_TYPE', 'MSELoss') == 'MSELoss')
                if len(preds_img) == 0:
                    continue  # 跳过空白帧
                preds_img, indexes = select_point(preds_img, 8)
                for p in range(len(preds_img)):
                    preds_img[p] /= torch.tensor([256, 192])
                for n in range(len(indexes)):
                    scores.pop(indexes[n])
                    preds_scores.pop(indexes[n])
                # 将关键点得分和人体得分组合
                mix_score = mix(preds_scores, scores, repeat=True, totensor=False)
                # 将得分和关键点坐标组合
                mix_pos = mix(mix_score, preds_img, repeat=False, totensor=True)
                shape = mix_pos.shape
                graph_features = torch.reshape(mix_pos, (shape[0] * shape[1], shape[2]))
                node_features.append(graph_features)
            if ((i + 1) % frame == 0) or ((i + 1) == data_len):
                if not node_features:
                    node_features = torch.zeros([18, 4])
                    edge_index, self_link = build_edge_index(node_features)  # 构建边
                    edge_attr = torch.zeros(39)
                else:
                    node_features = torch.cat(node_features, dim=0)
                    edge_index, self_link = build_edge_index(node_features)  # 构建边
                    attr = build_edge_attr(node_features, edge_index)  # 构建边权重
                    edge_attr = torch.round(attr * 1000) / 1000
                    index1 = torch.isinf(edge_attr)
                    edge_attr[index1] = 200
                data = Data(node_features, edge_index, edge_attr, y=y, seg=self_link)
                graph_data_list.append(data)
                node_features = []
    # edge_attr = attr[edge_index[0], edge_index[1]]
    # contains_zero = torch.any(edge_attr == 'nan')   # 检查是否存在0权重
    # writer.stop()
    return graph_data_list
