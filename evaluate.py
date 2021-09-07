import os 
import argparse

from douzero.evaluation.simulation import evaluate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    'Dou Dizhu Evaluation')
    parser.add_argument('--landlord', type=str,
            default='baselines/douzero_12/landlord_weights_39762328900.ckpt')
    parser.add_argument('--landlord_up', type=str,
            default='baselines/douzero_12/landlord_up_weights_39762328900.ckpt')
    parser.add_argument('--landlord_down', type=str,
            default='baselines/douzero_12/landlord_down_weights_39762328900.ckpt')
    parser.add_argument('--eval_data', type=str,
            default='eval_data_1000.pkl')
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--gpu_device', type=str, default='0')
    parser.add_argument('--output', type=bool, default=True)
    parser.add_argument('--bid', type=bool, default=True)
    parser.add_argument('--title', type=str, default='New')
    args = parser.parse_args()
    args.output = True
    args.bid = False
    if args.output or args.bid:
        args.num_workers = 1
    t = 3
    frame = 3085177900
    adp_frame = 2511184300
    # args.landlord = 'baselines/resnet_landlord_%i.ckpt' % frame
    args.landlord_up = 'baselines/resnet_landlord_up_%i.ckpt' % frame
    args.landlord_down = 'baselines/resnet_landlord_%i.ckpt' % frame
    args.landlord = 'baselines/douzero_ADP/landlord.ckpt'
    # args.landlord_up = 'baselines/douzero_ADP/landlord_up.ckpt'
    # args.landlord_down = 'baselines/douzero_ADP/landlord_down.ckpt'
    if t == 1:
        args.landlord = 'baselines/resnet_landlord_%i.ckpt' % frame
        args.landlord_up = 'baselines/douzero_ADP/landlord_up.ckpt'
        args.landlord_down = 'baselines/douzero_ADP/landlord_down.ckpt'
    elif t == 2:
        args.landlord = 'baselines/douzero_ADP/landlord.ckpt'
        args.landlord_up = 'baselines/resnet_landlord_up_%i.ckpt' % frame
        args.landlord_down = 'baselines/resnet_landlord_down_%i.ckpt' % frame
    elif t == 3:
        args.landlord = 'baselines/resnet_landlord_%i.ckpt' % frame
        args.landlord_up = 'baselines/resnet_landlord_up_%i.ckpt' % frame
        args.landlord_down = 'baselines/resnet_landlord_down_%i.ckpt' % frame
    elif t == 4:
        args.landlord = 'baselines/douzero_ADP/landlord.ckpt'
        args.landlord_up = 'baselines/douzero_ADP/landlord_up.ckpt'
        args.landlord_down = 'baselines/douzero_ADP/landlord_down.ckpt'
    elif t == 5:
        args.landlord = 'baselines/douzero_WP/landlord.ckpt'
        args.landlord_up = 'baselines/douzero_WP/landlord_up.ckpt'
        args.landlord_down = 'baselines/douzero_WP/landlord_down.ckpt'
    elif t == 6:
        args.landlord = 'baselines/resnet_landlord_%i.ckpt' % frame
        args.landlord_up = 'baselines/douzero_ADP/landlord_up_weights_%i.ckpt' % adp_frame
        args.landlord_down = 'baselines/douzero_ADP/landlord_down_weights_%i.ckpt' % adp_frame
    elif t == 7:
        args.landlord = 'baselines/douzero_ADP/landlord_weights_%i.ckpt' % adp_frame
        args.landlord_up = 'baselines/resnet_landlord_up_%i.ckpt' % frame
        args.landlord_down = 'baselines/resnet_landlord_down_%i.ckpt' % frame
    elif t == 8:
        args.landlord = 'baselines/douzero_ADP/landlord_weights_%i.ckpt' % adp_frame
        args.landlord_up = 'baselines/douzero_ADP/landlord_up_weights_%i.ckpt' % adp_frame
        args.landlord_down = 'baselines/douzero_ADP/landlord_down_weights_%i.ckpt' % adp_frame
    elif t == 9:
        args.landlord = 'baselines/resnet_landlord_%i.ckpt' % frame
        args.landlord_up = 'baselines/resnet_landlord_up_%i.ckpt' % adp_frame
        args.landlord_down = 'baselines/resnet_landlord_down_%i.ckpt' % adp_frame
    elif t == 10:
        # landlord_down_weights_10777798400
        args.landlord = 'baselines/douzero_ADP/landlord.ckpt'
        args.landlord_up = 'baselines/douzero_ADP/landlord_up_weights_%i.ckpt' % adp_frame
        args.landlord_down = 'baselines/douzero_ADP/landlord_down_weights_%i.ckpt' % adp_frame
    elif t == 11:
        args.landlord = 'baselines/douzero_ADP/landlord_weights_%i.ckpt' % adp_frame
        args.landlord_up = 'baselines/douzero_ADP/landlord_up.ckpt'
        args.landlord_down = 'baselines/douzero_ADP/landlord_down.ckpt'
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

    evaluate(args.landlord,
             args.landlord_up,
             args.landlord_down,
             args.eval_data,
             args.num_workers,
             args.output,
             args.bid,
             args.title)
