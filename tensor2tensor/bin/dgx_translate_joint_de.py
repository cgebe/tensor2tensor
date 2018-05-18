import os

TRANSLATE_PROBLEMS = [
    #"translate_csde_legal32k",
    #"translate_csen_legal32k",
    #"translate_cses_legal32k",
    #"translate_csfr_legal32k",
    #"translate_csit_legal32k",
    #"translate_cssv_legal32k",
    "translate_deen_legal32k",
    "translate_dees_legal32k",
    "translate_defr_legal32k",
    "translate_deit_legal32k",
    "translate_desv_legal32k",
    #"translate_enes_legal32k",
    #"translate_enfr_legal32k",
    #"translate_enit_legal32k",
    #"translate_ensv_legal32k",
    #"translate_esfr_legal32k",
    #"translate_esit_legal32k",
    #"translate_essv_legal32k",
    #"translate_frit_legal32k",
    #"translate_frsv_legal32k",
    #"translate_itsv_legal32k"
]


def main():
    joint = ""
    for problem in TRANSLATE_PROBLEMS:
        joint += problem+"-"

    joint = joint[:-1]
    print(joint)
    cmd = "python ./t2t-trainer --data_dir=$DATA_DIR/translate --output_dir=$TRAIN_DIR/multimodel/translate/joint-5-de --worker_gpu=8 --train_steps=250000 --model=multi_model --hparams_set=multimodel_base --save_checkpoints_secs=1200 --problems="+joint
    if os.system(cmd) == 0:
        continue


main()
