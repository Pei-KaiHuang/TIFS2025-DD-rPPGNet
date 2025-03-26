# parameter setting
import argparse


def get_args():
    
    parser = argparse.ArgumentParser()
    
    # ----------------- General ------------------
    parser.add_argument('--train_dataset', default="", type=str,
                        help="""
                        Options => C: COHFACE, P: PURE, U: UBFC, M: MR-NIRP, V: VIPL-HR,
                        e.g. --dataset="C"  for intra-training/testing on COHFACE
                             --dataset="UP" for cross-training/testing on PURE and UBFC        
                        """)
    parser.add_argument('--test_dataset', default="", type=str,
                        help="Same as above")
    
    parser.add_argument('--in_ch', default=3, type=float, 
                        help="input channel, you may change to 1 if dataset type is NIR")

    parser.add_argument('--model_S', default=2, type=int,
                        help="spatial dimension of model")
    parser.add_argument('--conv', default="conv3d", type=str,
                        help="Convolution type for 3DCNN")
        
    parser.add_argument('--bs', default=6, type=int,
                        help="batch size")
    parser.add_argument('--epoch', default=100, type=int,
                        help="training/testing epoch")
    parser.add_argument('--fps', default=30, type=int,
                        help="fps for dataset")
    parser.add_argument('--lr', default=1e-5, type=float,
                        help="learning rate")
    
    parser.add_argument('--high_pass', default=40, type=int)
    parser.add_argument('--low_pass', default=250, type=int)
    
    # ----------------- Training -----------------
    parser.add_argument('--train_T', default=10, type=int,
                        help="training clip length(seconds))")
    
    # ----------------- Testing -----------------
    parser.add_argument('--test_T', default=30, type=int,
                        help="testing clip length(seconds))")
    
    parser.add_argument('--delta_T', default=5, type=int,
                        help="sampling clip length(seconds))")
    
    
    parser.add_argument('--numSample', default=4, type=int,
                        help="")
    
    parser.add_argument('--inject_noise', action='store_true')

    # ----------------- Others -----------------
    #parser.add_argument('--preload_path', default="", type=str,
                       # help="preloaded dataset path")

    return parser.parse_args()


def get_name(args, train=True):
    
    trainName = f"{args.train_dataset}_{args.conv}_train_T{args.train_T}_k{args.model_S}"
    testName  = f"{args.train_dataset}_to_{args.test_dataset}_{args.conv}_test_T{args.test_T}_k{args.model_S}"
    
    if args.inject_noise:
        testName += "_injectNoise"
        
    return trainName, testName