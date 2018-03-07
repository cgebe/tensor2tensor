import os

TRANSLATE_PROBLEMS = [
    "translate_csde_legal32k",
    "translate_csen_legal32k",
    "translate_cses_legal32k",
    "translate_csfr_legal32k",
    "translate_csit_legal32k",
    "translate_cssv_legal32k",
    "translate_deen_legal32k",
    "translate_dees_legal32k",
    "translate_defr_legal32k",
    "translate_deit_legal32k",
    "translate_desv_legal32k",
    "translate_enes_legal32k",
    "translate_enfr_legal32k",
    "translate_enit_legal32k",
    "translate_ensv_legal32k",
    "translate_esfr_legal32k",
    "translate_esit_legal32k",
    "translate_essv_legal32k",
    "translate_frit_legal32k",
    "translate_frsv_legal32k",
    "translate_itsv_legal32k"
]


def main():
    for problem in TRANSLATE_PROBLEMS:
        if os.system("python ./t2t-datagen --data_dir=$DATA_DIR/translate/"+problem+" --tmp_dir=$TMP_DIR --problem=" + problem) == 0:
            continue
        else:
            print "ERROR " + problem
            break


main()
