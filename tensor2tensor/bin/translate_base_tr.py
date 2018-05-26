import os

TRANSLATE_PROBLEMS = [
    #"translate_csde_legal32k",
    #"translate_csen_legal32k",
    #"translate_cses_legal32k",
    #"translate_csfr_legal32k",
    #"translate_csit_legal32k",
    #"translate_cssv_legal32k",
    "translate_deen_legal32k",
    #"translate_dees_legal32k",
    #"translate_defr_legal32k",
    #"translate_deit_legal32k",
    #"translate_desv_legal32k",
    #"translate_enes_legal32k",
    #"translate_enfr_legal32k",
    #"translate_enit_legal32k",
    #"translate_ensv_legal32k",
    #"translate_esfr_legal32k",
    #"translate_esit_legal32k",
    #"translate_essv_legal32k",
    #"translate_frit_legal32k",
    #"translate_frsv_legal32k",
    #"translate_itsv_legal32k",
    #"translate_deit_legal32k",
    #"translate_desv_legal32k"
]


def main():
    for problem in TRANSLATE_PROBLEMS:
        cmd = "python ./t2t-trainer --data_dir=$DATA_DIR --output_dir=$TRAIN_DIR/transformer/translate/"+problem+" --schedule=train --worker_gpu=4 --training_steps=500000 --model=transformer --hparams_set=transformer_base_v3 --problems="+problem
        os.system(cmd)

main()
