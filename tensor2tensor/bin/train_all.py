import os

TRANSLATE_PROBLEMS = [
    #"translate_csde_legal32k",
    #"translate_csen_legal32k",
    #"translate_cses_legal32k",
    #"translate_csfr_legal32k",
    #"translate_csit_legal32k",
    #"translate_cssv_legal32k",
    #"translate_deen_legal32k",
    #"translate_dees_legal32k",
    #"translate_defr_legal32k",
    #"translate_deit_legal32k",
    #"translate_desv_legal32k",
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
        os.system("mkdir -p $TRAIN_DIR/translate/"+problem);
        successful = False
        while not successful:
            numbers = []
            for f in os.listdir(os.environ['TRAIN_DIR']+"/translate/"+problem):
                if f.endswith(".index"):
                    numbers.append(int(f.split("-")[1].split(".")[0]))
            if (len(numbers)) > 0 :
                steps = 250000 - max(numbers)
            else:
                steps = 250000

            print(steps)
            cmd = "python ./t2t-trainer --data_dir=$DATA_DIR/translate/"+problem+" --output_dir=$TRAIN_DIR/translate/"+problem+" --worker_gpu=4 --training_steps="+str(steps)+" --model=multi_model --hparams_set=multimodel_legal --problems="+problem
            if os.system(cmd) == 0:
                successful = True

main()
