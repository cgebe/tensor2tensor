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
    cmd = "python ./t2t-trainer --data_dir=$DATA_DIR/translate --output_dir=$TRAIN_DIR/multimodel/translate/joint-chain --worker_gpu=8 --train_steps=250000 --model=multi_model --hparams_set=multimodel_base --problems="+joint
    if os.system(cmd) == 0:
        continue


main()
