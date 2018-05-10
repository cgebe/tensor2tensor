import os

PROBLEMS = [
    "court_classification",
    "verdict_classification"
]


def main():
    for problem in PROBLEMS:
        os.system("mkdir -p $TRAIN_DIR/classification/"+problem);
        successful = False
        while not successful:
            cmd = "python ./t2t-trainer --data_dir=$DATA_DIR/classification --output_dir=$TRAIN_DIR/classification/"+problem+" --worker_gpu=4 --train_steps=50000 --model=multi_model --hparams_set=multimodel_legal --local_eval_frequency=500 --save_checkpoints_secs=90 --problems="+problem
            if os.system(cmd) == 0:
                successful = True
    
    joint = ""
    for problem in PROBLEMS:
        joint += problem + "-"

    joint = joint[:-1]

    os.system("mkdir -p $TRAIN_DIR/classification/joint");

    print(joint)
    successful = False
    while not successful:
        cmd = "python ./t2t-trainer --data_dir=$DATA_DIR/classification --output_dir=$TRAIN_DIR/classification/joint --worker_gpu=4 --train_steps=50000 --model=multi_model --hparams_set=multimodel_legal --local_eval_frequency=500 --save_checkpoints_secs=180 --problems=" + joint
        if os.system(cmd) == 0:
            successful = True


main()
