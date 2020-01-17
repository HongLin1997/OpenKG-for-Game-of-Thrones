
import argparse
import sys
import numpy as np

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--notransfer_flag", default=False,
                        type=bool, help="notransfer_flag")
    parser.add_argument("--model_name", default='bert', 
                        type=str, help="model_name")
    parser.add_argument("--target_domain", default='GOT', 
                        type=str, help="target_domain")
    parser.add_argument("--discriminator_objective", default=1, #1
                        type=int, help="discriminator_objective")
    parser.add_argument("--tgt_encoder_shared", default=False,
                        type=bool, help="tgt_encoder_shared")
    parser.add_argument("--share_encoder_init", default=False,
                        type=bool, help="share_encoder_init")
    
    
    parser.add_argument("--dataset", default='CCKS',
                        type=str, help="dataset")
    parser.add_argument("--dataset_att", default='CCKS',
                        type=str, help="dataset")
    parser.add_argument("--pad_size", default=90,
                        type=int, help="pad_size")
    
    parser.add_argument("--optimizer", default='adamw',
                        type=str, help="optimizer")
    parser.add_argument("--batch_size", default=64,
                        type=int, help="batch_size")
    parser.add_argument("--num_workers", default=0,
                        type=int, help="num_workers")
    
    
    parser.add_argument("--num_gpu", default=0,
                        type=int, help="num_gpu")
    parser.add_argument("--num_epochs_pre", default=15,#100
                        type=int, help="num_epochs_pre")
    parser.add_argument("--log_step_pre", default=1,
                        type=int, help="log_step_pre")
    parser.add_argument("--eval_step_pre", default=1,
                        type=int, help="eval_step_pre")
    parser.add_argument("--save_step_pre", default=1,
                        type=int, help="save_step_pre")
    parser.add_argument("--num_epochs", default=30, #300
                        type=int, help="num_epochs")
    parser.add_argument("--num_epochs_shared", default=50, #30
                        type=int, help="num_epochs_shared")
    parser.add_argument("--log_step", default=1,
                        type=int, help="log_step")
    parser.add_argument("--save_step", default=1,
                        type=int, help="save_step")
    parser.add_argument("--manual_seed", default=42,
                        type=int, help="manual_seed")
    
    # params for optimizing models
    parser.add_argument("--num_epochs_critic", default=20,#100
                        type=int, help="num_epochs_critic before adversarial training")
    parser.add_argument("--warmup_step", default=300,#100
                        type=int, help="warmup_step")
    parser.add_argument("--early_stop_patient", default=30,#100
                        type=int, help="early_stop_patient")
    parser.add_argument("--lambda_", default=0.1,
                        type=float, help="lambda_")
    parser.add_argument("--gamma_", default=10,
                        type=float, help="gamma_")
    
    parser.add_argument("--dropout_linear", default=0.5,
                        type=float, help="dropout_linear")
    parser.add_argument("--dropout_other", default=0.1,
                        type=float, help="dropout_other")
    parser.add_argument("--weight_decay", default=1e-5,
                        type=float, help="d_learning_rate")
    parser.add_argument("--update_critic_threshold", default=0.7,
                        type=float, help="update_critic_threshold")
    parser.add_argument("--d_learning_rate", default=1e-3,
                        type=float, help="d_learning_rate")
    parser.add_argument("--c_learning_rate", default=1e-3,
                        type=float, help="c_learning_rate")
    parser.add_argument("--adv_learning_rate", default=1e-3,
                        type=float, help="adv_learning_rate")
    parser.add_argument("--bert_learning_rate", default=3e-5,
                        type=float, help="bert_learning_rate")
    parser.add_argument("--beta1", default=0.9,
                        type=float, help="beta1")
    parser.add_argument("--beta2", default=0.999,
                        type=float, help="beta2")
    
    # adversarial parameters
    parser.add_argument('--alpha', type=float, default=1.0,
                        help="Specify adversarial weight for cls_loss")

    parser.add_argument('--beta', type=float, default=1.0,
                        help="Specify KD loss weight for kd_loss")

    parser.add_argument('--gamma', type=float, default=0.0,
                        help="Specify regularizer weight for mmd loss")

    parser.add_argument('--temperature', type=int, default=20,
                        help="Specify temperature for kd_loss")

    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--clip_value", type=float, default=np.inf, #0.01
                        help="lower and upper clip value for discriminator. weights")
    
    parser.add_argument('--iterations', type=int, default=500,
                        help="iterations")
    parser.add_argument('--k_clf', type=int, default=10,
                        help="k_disc")
    parser.add_argument('--k_disc', type=int, default=1,
                        help="k_disc")

    if len(sys.argv) == 0:
        parser.print_help()
        sys.exit(1)

    print("")
    args = parser.parse_args()
    for arg in vars(args):
        print("{}={}".format(arg.upper(), getattr(args, arg)))
    print("")

    return args


FLAGS = parse_args()


notransfer_flag = FLAGS.notransfer_flag    
model_name = FLAGS.model_name       
target_domain = FLAGS.target_domain
tgt_encoder_shared = FLAGS.tgt_encoder_shared
share_encoder_init = FLAGS.share_encoder_init
discriminator_objective = FLAGS.discriminator_objective

# params for dataset and data loader

dataset_att = {'CCKS':{'data_root':'datasets/ccks_IPRE/',
                       'num_classes':35},
               'GOT':{'data_root':'datasets/GOT/',
                       'num_classes':24},
                }
               
num_classes= dataset_att[FLAGS.dataset_att]['num_classes']
dataset = FLAGS.dataset
pad_size = FLAGS.pad_size
data_root = dataset_att[FLAGS.dataset_att]['data_root']
tgt_num_classes = dataset_att[FLAGS.target_domain]['num_classes']
batch_size = FLAGS.batch_size
optimizer = FLAGS.optimizer
num_workers = FLAGS.num_workers

source_train = 'train.json'
source_validate = 'dev.json'
source_test = 'test.json'

target_dataroot = dataset_att[FLAGS.target_domain]['data_root']
target_train = 'corpus_train.jsonl'
target_validate = 'corpus_dev.jsonl'
target_test = 'corpus_eval.jsonl'

# params for adversarial training
alpha = FLAGS.alpha
beta = FLAGS.beta
gamma = FLAGS.gamma
temperature = FLAGS.temperature
max_grad_norm = FLAGS.max_grad_norm
clip_value = FLAGS.clip_value

# params for source encoder
model_root = "snapshot"

checkpoints_pretrain = model_root +"/checkpoint_pretrain.pkl"
checkpoints_adapt = model_root +"/checkpoint_adapt.pkl"
checkpoints_shared = model_root +"/checkpoint.pkl"

shared_encoder_restore = model_root +"/%s-shared-encoder-final.pt"%(model_name)
shared_classifier_restore = model_root +"/%s-shared-classifier-final.pt"%(model_name)
shared_model_trained = True

src_encoder_restore = model_root +"/%s-source-encoder-final.pt"%(model_name)
src_classifier_restore = model_root +"/%s-source-classifier-final.pt"%(model_name)
src_model_trained = True

# params for target encoder
tgt_encoder_restore = model_root +"/%s-target-encoder-final.pt"%(model_name)
tgt_model_trained = True

tgt_classifier_restore = model_root +"/%s-target-classifier-final.pt"%(model_name)
tgt_classifier_trained = True

# params for setting up domain discriminator 
d_model_restore = model_root +"/%s-critic-final.pt"%(model_name)

# params for training network
num_gpu = FLAGS.num_gpu
num_epochs_pre = FLAGS.num_epochs_pre#100
num_epochs_shared = FLAGS.num_epochs_shared
log_step_pre = FLAGS.log_step_pre
eval_step_pre = FLAGS.eval_step_pre
save_step_pre = FLAGS.save_step_pre
num_epochs = FLAGS.num_epochs
log_step = FLAGS.log_step
save_step = FLAGS.save_step
manual_seed = FLAGS.manual_seed

# params for optimizing models
num_epochs_critic = FLAGS.num_epochs_critic
update_critic_threshold = FLAGS.update_critic_threshold
warmup_step = FLAGS.warmup_step
early_stop_patient = FLAGS.early_stop_patient
lambda_ = FLAGS.lambda_
gamma_ = FLAGS.gamma_
dropout_linear = FLAGS.dropout_linear
dropout_other = FLAGS.dropout_other
weight_decay = FLAGS.weight_decay
d_learning_rate = FLAGS.d_learning_rate
c_learning_rate = FLAGS.c_learning_rate
adv_learning_rate = FLAGS.adv_learning_rate
bert_learning_rate = FLAGS.bert_learning_rate
beta1 = FLAGS.beta1
beta2 = FLAGS.beta2

iterations = FLAGS.iterations
k_clf = FLAGS.k_clf
k_disc = FLAGS.k_disc