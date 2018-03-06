import os

TRANSLATE_PROBLEMS = [
    "translate_csde_legal8k"
    "translate_csen_legal8k"
    "translate_cses_legal8k"
    "translate_csfr_legal8k"
    "translate_csit_legal8k"
    "translate_cssv_legal8k"
    "translate_deen_legal8k"
    "translate_dees_legal8k"
    "translate_defr_legal8k"
    "translate_deit_legal8k"
    "translate_desv_legal8k"
    "translate_enes_legal8k"
    "translate_enfr_legal8k"
    "translate_enit_legal8k"
    "translate_ensv_legal8k"
    "translate_esfr_legal8k"
    "translate_esit_legal8k"
    "translate_essv_legal8k"
    "translate_frit_legal8k"
    "translate_frsv_legal8k"
    "translate_itsv_legal8k"
]


def main():
    for problem in TRANSLATE_PROBLEMS:
        if os.system("python ./t2t-trainer --data_dir=$DATA_DIR --output_dir=$TRAIN_DIR --model=multi_model --worker_gpu 4 --hparams_set=multimodel_base --problems="+problem) == 0:
            continue
        else:
            print "ERROR " + problem
            break

main()
