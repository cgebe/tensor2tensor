import os

PROBLEMS = [
    "translate_deen_legal32k",
    "translate_dees_legal32k",
    "translate_defr_legal32k",
    "translate_deit_legal32k",
    "translate_desv_legal32k",
    "summarize_cs_legal32k",
    "summarize_de_legal32k",
    "summarize_en_legal32k",
    "summarize_es_legal32k",
    "summarize_fr_legal32k",
    "summarize_it_legal32k",
    "summarize_sv_legal32k",
    "multi_labeling_cs_legal32k",
    "multi_labeling_de_legal32k",
    "multi_labeling_en_legal32k",
    "multi_labeling_es_legal32k",
    "multi_labeling_fr_legal32k",
    "multi_labeling_it_legal32k",
    "multi_labeling_sv_legal32k"
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
