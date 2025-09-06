import argparse

from matplotlib import gridspec
from matplotlib.animation import FuncAnimation
import sys
import os
import cv2
import copy
import imageio
import io
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from demo.lib.utils import camera_to_world

sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))
import os
current_dir = os.path.dirname(__file__)
root_path = os.path.dirname(current_dir)
sys.path.append(root_path)
from tools import *
from learning_all import *
from utils.data import read_pkl, flip_data, normalize_screen_coordinates

connections = [
    (10, 9),
    (9, 8),
    (8, 11),
    (8, 14),
    (14, 15),
    (15, 16),
    (11, 12),
    (12, 13),
    (8, 7),
    (7, 0),
    (0, 4),
    (0, 1),
    (1, 2),
    (2, 3),
    (4, 5),
    (5, 6)
]


def show3Dpose(pred, gt, ax):
    ax.view_init(elev=15., azim=70)

    colors = [(43/255, 44/255, 124/255),
            (39/255, 98/255, 53/255),
            (171/255, 37/255, 36/255)]

    I = np.array( [0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9])
    J = np.array( [1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])

    LR = [3, 3, 3, 3, 3, 3, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1]

    RADIUS = 0.72
    RADIUS_Z = 0.7

    for i in np.arange( len(I) ):
        x, y, z = [np.array( [gt[I[i], j], gt[J[i], j]] ) for j in range(3)]
        ax.plot(x, y, z, lw=2, color = '0.5')

    xroot, yroot, zroot = gt[0,0], gt[0,1], gt[0,2]
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])
    ax.set_zlim3d([-RADIUS_Z+zroot, RADIUS_Z+zroot])
    ax.set_aspect('auto') # works fine in matplotlib==2.2.2

    for i in np.arange( len(I) ):
        x, y, z = [np.array( [pred[I[i], j], pred[J[i], j]] ) for j in range(3)]
        ax.plot(x, y, z, lw=2, color = colors[LR[i]-1])

    xroot, yroot, zroot = pred[0,0], pred[0,1], pred[0,2]
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])
    ax.set_zlim3d([-RADIUS_Z+zroot, RADIUS_Z+zroot])
    ax.set_aspect('auto') # works fine in matplotlib==2.2.2

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white)
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params('x', labelbottom = False)
    ax.tick_params('y', labelleft = False)
    ax.tick_params('z', labelleft = False)
def read_h36m(x):
    cam2real = np.array([[1, 0, 0],
                         [0, 0, -1],
                         [0, 1, 0]], dtype=np.float32)
    scale_factor = 0.298
    sample_joint_seq = x
    sample_joint_seq = sample_joint_seq - sample_joint_seq[:,0:1,:]
    sample_joint_seq = sample_joint_seq.transpose(1, 0, 2)
    sample_joint_seq = (sample_joint_seq / scale_factor) @ cam2real
    return sample_joint_seq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_pose_id', type=int, default=300)
    parser.add_argument('--dataset', default='h36m')
    parser.add_argument("--config", type=str, default=root_path + '/configs/h36m/MyModel-base.yaml', help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default=root_path + '/checkpoint/BESTMambaPose_b4r2l14d128_lr001decay99_207', type=str, metavar='PATH', help='checkpoint path')
    parser.add_argument('--checkpoint-file', type=str, help="checkpoint file name", default='best_epoch.pth.tr')
    parser.add_argument('--test_pose', default=root_path + '/data/motion3d/H36M-243/test/%08d.pkl', type=str, metavar='PATH', help='test pose path')
    args = parser.parse_args()

    for i in range(810, 811, 1): # (0, 2228, 150)
        args.test_pose_id = i
        print(f"Visualizing test_pose {args.test_pose_id} of {args.dataset} dataset")
        dir_path = args.checkpoint + f'/test_pose{args.test_pose_id}'

        # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        configs = get_config(args.config)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = load_model(configs)
        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model)
        model.to(device)

        checkpoint_path = os.path.join(args.checkpoint, args.checkpoint_file)
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model'], strict=True)
        print(f'Load checkpoint : {checkpoint_path}')

        model.eval()

        pose_data = read_pkl(args.test_pose % args.test_pose_id)  # dict
        gt_3D = pose_data['data_label']
        input_2D = pose_data['data_input'].reshape(1,243,17,-1)  # (1, T, J, 3)
        input_2D_aug = flip_data(input_2D)
        # confidence_scores = input_2D[..., 2].reshape(1, 243, 17, 1)
        # input_2D = normalize_screen_coordinates(input_2D[..., :2], w=1000, h=1000)
        # input_2D = np.concatenate((input_2D, confidence_scores), axis=-1)
        input_2D = torch.from_numpy(input_2D.astype('float32')).cuda()
        input_2D_aug = torch.from_numpy(input_2D_aug.astype('float32')).cuda()
        with torch.no_grad():
            output_3D_non_flip = model(input_2D)
            output_3D_flip = model(input_2D_aug)
            output_3D_flip = flip_data(output_3D_flip)
            output_3D = (output_3D_non_flip + output_3D_flip) / 2

        # output_3D[:, :, 0, :] = 0
        output_3D = output_3D[0].cpu().detach().numpy()
        output_3D = output_3D - output_3D[:, 0:1, :]
        gt_3D = gt_3D - gt_3D[:, 0:1, :]

        for j, post_out in enumerate(output_3D):
            if j % 27 != 0:
                continue
            gt_pose = gt_3D[j, ...]
            rot = [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
            rot = np.array(rot, dtype='float32')
            post_out = camera_to_world(post_out, R=rot, t=0)
            post_out[:, 2] -= np.min(post_out[:, 2])
            max_value = np.max(post_out)
            post_out /= max_value

            gt_pose = camera_to_world(gt_pose, R=rot, t=0)
            gt_pose[:, 2] -= np.min(gt_pose[:, 2])
            max_value = np.max(gt_pose)
            gt_pose /= max_value

            fig = plt.figure(figsize=(9.6, 5.4))
            gs = gridspec.GridSpec(1, 1)
            gs.update(wspace=-0.00, hspace=0.05)
            ax = plt.subplot(gs[0], projection='3d')
            show3Dpose(post_out, gt_pose, ax)

            output_dir_3D = dir_path + '/pred_gt/'
            os.makedirs(output_dir_3D, exist_ok=True)
            fig_name = output_dir_3D + f'frame_{j}' + '_3D.png'
            plt.savefig(fig_name, dpi=200, format='png',
                        bbox_inches='tight')
            print(fig_name)
            plt.close(fig)





if __name__ == '__main__':
    main()
