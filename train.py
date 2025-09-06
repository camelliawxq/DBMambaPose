import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch
from torch import optim
from tqdm import tqdm

from loss.pose3d import loss_mpjpe, n_mpjpe, loss_velocity, loss_limb_var, loss_limb_gt, loss_angle, \
    loss_angle_velocity, loss_spatial_rank, loss_temporal_rank, weighted_mpjpe, loss_TCL
from loss.pose3d import jpe as calculate_jpe
from loss.pose3d import p_mpjpe as calculate_p_mpjpe
from loss.pose3d import mpjpe as calculate_mpjpe
from loss.pose3d import acc_error as calculate_acc_err
from loss.pose3d import mean_velocity_error as calculate_mpjve
from data.const import H36M_JOINT_TO_LABEL, H36M_UPPER_BODY_JOINTS, H36M_LOWER_BODY_JOINTS, H36M_1_DF, H36M_2_DF, \
    H36M_3_DF
from data.reader.h36m import DataReaderH36M
from data.reader.motion_dataset import MotionDataset3D
from utils.data import flip_data
from utils.graph_utils import adj_mx_from_skeleton_temporal, adj_mx_from_skeleton
from utils.tools import set_random_seed, get_config, print_args, create_directory_if_not_exists, Logger, \
    print_action_errors, print_action_errors_mpjve
from torch.utils.data import DataLoader

from utils.learning import load_model, AverageMeter, decay_lr_exponentially
from utils.tools import count_param_numbers
from utils.data import Augmenter2D


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/h36m/DBMambaPose-L.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                        help='checkpoint directory')
    parser.add_argument('--new-checkpoint', type=str, metavar='PATH', default='debug',
                        help='new checkpoint directory')
    parser.add_argument('--checkpoint-file', type=str, help="checkpoint file name")
    parser.add_argument('-sd', '--seed', default=0, type=int, help='random seed')
    parser.add_argument('--num-cpus', default=12, type=int, help='Number of CPU cores')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--eval-only', action='store_true')
    opts = parser.parse_args()
    return opts


def train_one_epoch(args, model, train_loader, optimizer, device, losses):
    model.train()
    for x, y in tqdm(train_loader, total=len(train_loader), ncols=80):
        batch_size = x.shape[0]
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            if args.root_rel:
                y = y - y[..., 0:1, :]
            else:
                y[..., 2] = y[..., 2] - y[:, 0:1, 0:1, 2]  # Place the depth of first frame root to be 0

        pred = model(x)  # (N, F, 17, 3)

        optimizer.zero_grad()

        loss_3d_pos = loss_mpjpe(pred, y)
        loss_3d_scale = n_mpjpe(pred, y)
        loss_3d_velocity = loss_velocity(pred, y)
        loss_lv = loss_limb_var(pred)
        loss_lg = loss_limb_gt(pred, y)
        loss_a = loss_angle(pred, y)
        loss_av = loss_angle_velocity(pred, y)
        loss_sr = loss_spatial_rank(pred, y)
        loss_tr = loss_temporal_rank(pred, y)

        w_mpjpe = torch.tensor([1, 1, 2.5, 2.5, 1, 2.5, 2.5, 1, 1, 1, 1.5, 1.5, 4, 4, 1.5, 4, 4]).cuda()
        loss_3d_w = weighted_mpjpe(pred, y, w_mpjpe)  # 3D weighted
        loss_tcl = loss_TCL(pred, w_mpjpe)

        loss_total = args.lambda_3d * loss_3d_pos + \
                     args.lambda_scale * loss_3d_scale + \
                     args.lambda_3d_velocity * loss_3d_velocity + \
                     args.lambda_lv * loss_lv + \
                     args.lambda_lg * loss_lg + \
                     args.lambda_a * loss_a + \
                     args.lambda_av * loss_av + \
                     args.lambda_sr * loss_sr + \
                     args.lambda_tr * loss_tr + \
                     args.lambda_3dw * loss_3d_w + \
                     args.lambda_tcl * loss_tcl

        losses['3d_pose'].update(loss_3d_pos.item(), batch_size)
        losses['3d_scale'].update(loss_3d_scale.item(), batch_size)
        losses['3d_velocity'].update(loss_3d_velocity.item(), batch_size)
        losses['lv'].update(loss_lv.item(), batch_size)
        losses['lg'].update(loss_lg.item(), batch_size)
        losses['angle'].update(loss_a.item(), batch_size)
        losses['angle_velocity'].update(loss_av.item(), batch_size)
        losses['spatial_rank'].update(loss_sr.item(), batch_size)
        losses['temporal_rank'].update(loss_tr.item(), batch_size)
        losses['weighted_loss'].update(loss_3d_w.item(), batch_size)
        losses['tcl'].update(loss_tcl.item(), batch_size)
        losses['total'].update(loss_total.item(), batch_size)

        loss_total.backward()
        optimizer.step()


def evaluate(args, model, test_loader, datareader, device):
    print("[INFO] Evaluation")
    results_all = []
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(test_loader, total=len(test_loader), ncols=80):
            x, y = x.to(device), y.to(device)

            if args.flip:
                batch_input_flip = flip_data(x)
                predicted_3d_pos_1 = model(x)
                predicted_3d_pos_flip = model(batch_input_flip)
                predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip)  # Flip back
                predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2
            else:
                predicted_3d_pos = model(x)
            if args.root_rel:
                predicted_3d_pos[:, :, 0, :] = 0  # [N,T,17,3]
            else:
                y[:, 0, 0, 2] = 0

            results_all.append(predicted_3d_pos.cpu().numpy())

    results_all = np.concatenate(results_all)
    results_all = datareader.denormalize(results_all)
    _, split_id_test = datareader.get_split_id()
    actions = np.array(datareader.dt_dataset['test']['action'])
    factors = np.array(datareader.dt_dataset['test']['2.5d_factor'])
    gts = np.array(datareader.dt_dataset['test']['joints_2.5d_image'])
    sources = np.array(datareader.dt_dataset['test']['source'])

    num_test_frames = len(actions)
    frames = np.array(range(num_test_frames))
    action_clips = actions[split_id_test]
    factor_clips = factors[split_id_test]
    source_clips = sources[split_id_test]
    frame_clips = frames[split_id_test]
    gt_clips = gts[split_id_test]
    if args.add_velocity:
        action_clips = action_clips[:, :-1]
        factor_clips = factor_clips[:, :-1]
        frame_clips = frame_clips[:, :-1]
        gt_clips = gt_clips[:, :-1]

    assert len(results_all) == len(action_clips)

    e1_all = np.zeros(num_test_frames)
    jpe_all = np.zeros((num_test_frames, args.num_joints))
    e2_all = np.zeros(num_test_frames)
    acc_err_all = np.zeros(num_test_frames - 2)
    mpjve_all = np.zeros(num_test_frames)
    oc = np.zeros(num_test_frames)
    results = {}
    results_procrustes = {}
    results_joints = [{} for _ in range(args.num_joints)]
    results_accelaration = {}
    results_mpjve = {}
    action_names = sorted(set(datareader.dt_dataset['test']['action']))
    for action in action_names:
        results[action] = []
        results_procrustes[action] = []
        results_accelaration[action] = []
        results_mpjve[action] = []

        for joint_idx in range(args.num_joints):
            results_joints[joint_idx][action] = []

    block_list = ['s_09_act_05_subact_02',
                  's_09_act_10_subact_02',
                  's_09_act_13_subact_01']
    for idx in range(len(action_clips)):
        source = source_clips[idx][0][:-6]
        if source in block_list:
            continue
        frame_list = frame_clips[idx]
        action = action_clips[idx][0]
        factor = factor_clips[idx][:, None, None]
        gt = gt_clips[idx]
        pred = results_all[idx]
        pred *= factor

        # Root-relative Errors
        pred = pred - pred[:, 0:1, :]
        gt = gt - gt[:, 0:1, :]
        err1 = calculate_mpjpe(pred, gt)
        jpe = calculate_jpe(pred, gt)
        for joint_idx in range(args.num_joints):
            jpe_all[frame_list, joint_idx] += jpe[:, joint_idx]
        acc_err = calculate_acc_err(pred, gt)
        acc_err_all[frame_list[:-2]] += acc_err
        e1_all[frame_list] += err1
        err2 = calculate_p_mpjpe(pred, gt)
        e2_all[frame_list] += err2
        mpjve = calculate_mpjve(pred, gt)
        mpjve_all[frame_list] += mpjve
        oc[frame_list] += 1

    for idx in range(num_test_frames):
        if e1_all[idx] > 0:
            err1 = e1_all[idx] / oc[idx]
            err2 = e2_all[idx] / oc[idx]
            action = actions[idx]
            results_procrustes[action].append(err2)
            acc_err = acc_err_all[idx] / oc[idx]
            results[action].append(err1)
            results_accelaration[action].append(acc_err)
            for joint_idx in range(args.num_joints):
                jpe = jpe_all[idx, joint_idx] / oc[idx]
                results_joints[joint_idx][action].append(jpe)
            mpjve = mpjve_all[idx] / oc[idx]
            results_mpjve[action].append(mpjve)

    final_result_procrustes = []  # P2
    final_result_joints = [[] for _ in range(args.num_joints)]
    final_result_acceleration = []  # acc err
    final_result = []  # P1
    final_result_mpjve = []  # mpjve

    for action in action_names:
        final_result.append(np.mean(results[action]))
        final_result_procrustes.append(np.mean(results_procrustes[action]))
        final_result_acceleration.append(np.mean(results_accelaration[action]))
        for joint_idx in range(args.num_joints):
            final_result_joints[joint_idx].append(np.mean(results_joints[joint_idx][action]))
        final_result_mpjve.append(np.mean(results_mpjve[action]))

    joint_errors = []
    for joint_idx in range(args.num_joints):
        joint_errors.append(
            np.mean(np.array(final_result_joints[joint_idx]))
        )
    joint_errors = np.array(joint_errors)

    p1, p2, acceleration_error = print_action_errors(action_names, final_result, final_result_procrustes,
                                                     final_result_acceleration, joint_errors)
    # p1, p2, acceleration_error, mpjve = print_action_errors_mpjve(action_names, final_result, final_result_procrustes,
    #                                                  final_result_acceleration, joint_errors, final_result_mpjve)
    return p1, p2, joint_errors, acceleration_error


def save_checkpoint(checkpoint_path, epoch, lr, optimizer, model, min_mpjpe):
    torch.save({
        'epoch': epoch + 1,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'min_mpjpe': min_mpjpe,
    }, checkpoint_path)


def train(args, opts):
    print_args(args)
    # create_directory_if_not_exists(opts.new_checkpoint)

    train_dataset = MotionDataset3D(args, args.subset_list, 'train')
    test_dataset = MotionDataset3D(args, args.subset_list, 'test')

    common_loader_params = {
        'batch_size': args.batch_size,
        'num_workers': opts.num_cpus - 1,
        'pin_memory': True,
        'prefetch_factor': (opts.num_cpus - 1) // 3,
        'persistent_workers': True
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **common_loader_params)
    test_loader = DataLoader(test_dataset, shuffle=False, **common_loader_params)

    datareader = DataReaderH36M(n_frames=args.n_frames, sample_stride=1,
                                data_stride_train=args.n_frames // 3, data_stride_test=args.n_frames,
                                dt_root='data/motion3d', dt_file=args.dt_file)  # Used for H36m evaluation

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #####Topology Construction######

    temporal_skeleton = list(range(0, args.n_frames))  # 243

    temporal_skeleton = np.array(temporal_skeleton)

    temporal_skeleton -= 1  # [-1, 0, ... ,241]

    # temporal topology

    adj_temporal = adj_mx_from_skeleton_temporal(args.n_frames, temporal_skeleton)

    # spatial topology

    adj = adj_mx_from_skeleton()

    model = load_model(args, adj=adj, adj_temporal=adj_temporal)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
    model.to(device)

    n_params = count_param_numbers(model)
    print(f"[INFO] Number of parameters: {n_params / 1000000:,}M")

    lr = args.learning_rate
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=lr,
                            weight_decay=args.weight_decay)
    lr_decay = args.lr_decay
    epoch_start = 0
    min_mpjpe = float('inf')  # Used for storing the best model

    if opts.checkpoint:
        checkpoint_path = os.path.join(opts.checkpoint,
                                       opts.checkpoint_file if opts.checkpoint_file else "latest_epoch.pth.tr")

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['model'], strict=True)
            # model.load_state_dict(checkpoint['model_pos'], strict=True)
            print(f'Load checkpoint : {checkpoint_path}')

            if opts.resume:
                lr = checkpoint['lr']
                epoch_start = checkpoint['epoch']
                optimizer.load_state_dict(checkpoint['optimizer'])
                min_mpjpe = checkpoint['min_mpjpe']
        else:
            print("[WARN] Checkpoint path is empty. Starting from the beginning")
            opts.resume = False

    if not opts.eval_only:
        checkpoint_path_latest = os.path.join(opts.new_checkpoint, 'latest_epoch.pth.tr')
        checkpoint_path_best = os.path.join(opts.new_checkpoint, 'best_epoch.pth.tr')

    for epoch in range(epoch_start, args.epochs):
        if opts.eval_only:
            evaluate(args, model, test_loader, datareader, device)
            exit()

        print(f"[INFO] epoch {epoch}")
        loss_names = ['3d_pose', '3d_scale', '2d_proj', 'lg', 'lv', '3d_velocity', 'angle', 'angle_velocity',
                      'spatial_rank', 'temporal_rank', 'weighted_loss', 'tcl', 'total']

        losses = {name: AverageMeter() for name in loss_names}

        train_one_epoch(args, model, train_loader, optimizer, device, losses)

        mpjpe, p_mpjpe, joints_error, acceleration_error = evaluate(args, model, test_loader, datareader, device)

        if mpjpe < min_mpjpe:
            min_mpjpe = mpjpe
            save_checkpoint(checkpoint_path_best, epoch, lr, optimizer, model, min_mpjpe)
            print(f'Best checkpoint save to : {checkpoint_path_best}')

        save_checkpoint(checkpoint_path_latest, epoch, lr, optimizer, model, min_mpjpe)

        joint_label_errors = {}
        for joint_idx in range(args.num_joints):
            joint_label_errors[f"eval_joints/{H36M_JOINT_TO_LABEL[joint_idx]}"] = joints_error[joint_idx]

        lr = decay_lr_exponentially(lr, lr_decay, optimizer)


def main():
    opts = parse_args()
    set_random_seed(opts.seed)
    torch.backends.cudnn.benchmark = False
    args = get_config(opts.config)

    print(opts)

    if opts.eval_only:
        description = "Evaluate!"
        log_path = opts.checkpoint
        TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}_eval".format(datetime.now())
    else:
        description = "Train!"
        if opts.new_checkpoint is None:
            opts.new_checkpoint = opts.checkpoint
        log_path = opts.new_checkpoint
        create_directory_if_not_exists(log_path)
        TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}_train".format(datetime.now())

    print(description)
    print('python ' + ' '.join(sys.argv))
    print("CUDA Device Count: ", torch.cuda.device_count())

    logfile = os.path.join(log_path, TIMESTAMP + '_logging.log')
    sys.stdout = Logger(logfile)

    train(args, opts)


if __name__ == '__main__':
    main()
