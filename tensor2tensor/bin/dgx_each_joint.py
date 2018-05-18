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
    cmd = "python ./t2t-trainer --data_dir=$DATA_DIR --output_dir=$TRAIN_DIR/multimodel/translate/joint-each --worker_gpu=8 --train_steps=250000 --model=multi_model --hparams_set=multimodel_base --save_checkpoints_secs=1200 --problems="+joint
    os.system(cmd)


main()
