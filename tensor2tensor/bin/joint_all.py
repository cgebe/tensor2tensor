import os

PROBLEMS = [
    "translate_deen_legal32k",
    "summarize_de_legal32k",
    "multi_labeling_de_legal32k",
]


def main():
    joint = ""
    for problem in PROBLEMS:
        joint += problem+"-"

    joint = joint[:-1]
    print(joint)
    cmd = "python ./t2t-trainer --data_dir=$DATA_DIR/joint --output_dir=$TRAIN_DIR/joint/all-de --worker_gpu=4 --train_steps=500000 --model=multi_model --hparams_set=multimodel_legal --save_checkpoints_secs=600 --problems="+joint
    os.system(cmd)


main()
