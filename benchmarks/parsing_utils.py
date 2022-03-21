from configargparse import ArgParser
import os

def make_parser(default_config_files=[".my_settings"]):
    parser = ArgParser(description="Driver", default_config_files=default_config_files) #???? 

    parser.add_argument(
        '--weighted_samples', 
        action='store_true',
        help='choose to sample training data in a weighted manner',
    )

    parser.add_argument(
        '--roi', 
        action='store_true',
        help='crop global image input to an ROI',
    )
    parser.add_argument(
        '--train', 
        action='store_true',
        help='run train function',
    )
    parser.add_argument(
        '--test', 
        action='store_true',
        help='run evaluation function ',
    )

    parser.add_argument(
        "--cuda",
        type=int,
        default=-1,
        help="enables CUDA training on specified number of gpus (0 for cpu). Default to use all",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        metavar="b",
        help="number of elements per minibatch",
    )

    parser.add_argument(
        "--lr", type=float, default=1e-3, metavar="lr", help="learning rate"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )

    parser.add_argument(
        "--classes",
        type=list,
        default=['control', 'mdivi', 'llo'],
        metavar="cl",
        help="list of possible class types",
    )


    parser.add_argument(
        "--loss_fn",
        type=str,
        default="",
        metavar="lf",
        help="loss function",
    )

    parser.add_argument(
        "--root_dir",
        type=str,
        default="./",
        metavar="d",
        help="sets the root directory for saving operations",
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        default="./",
        metavar="i",
        help="absolute path to the directory for accessing data",
    ) 

    #Todo save model
    parser.add_argument(
            "--save",
            type=str,
            default="",
            metavar="s",
            help="saves the weights to a given filepath",
        )

    # args.root_dir = args.root_path if args.root_path[-1] == "/" else f"{args.root_path}/" 
    args = parser.parse_known_args()[0]
    if args.save:
        args.save = os.path.join(args.root_dir, args.save) # TODO - do u need to create folder?  

    return args, parser
