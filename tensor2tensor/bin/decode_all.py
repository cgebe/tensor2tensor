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

CORPORA = [
    "jrc",
    "dcep",
    "europarl"
]


def main():
    for problem in TRANSLATE_PROBLEMS:
        for corpus in CORPORA:
            os.system("mkdir -p $TRAIN_DIR/translate/"+problem);
            if os.system("python ./t2t-decoder --data_dir=$DATA_DIR/translate/"+problem+" --output_dir=$TRAIN_DIR/translate/"+problem+" --model=multi_model --hparams_set=multimodel_legal --problems="+problem+" --decode_hparams='beam_size=10,alpha=0.6' --decode_from_file=$DECODE_DIR/"+getTestFile(corpus, problem)+" --decode_to_file=$DECODE_DIR/"+getDecodeFile(corpus, problem)) == 0:
                continue
            else:
                print "ERROR " + problem
                break

def getTestFile(corpus, problem):
    pair = problem.split("_")[1]
    langs = pair[:2] + "-" + pair[2:]
    if corpus == "jrc":
        return "jrc/jrc_acquis."+langs+"-test."+pair[:2]
    if corpus == "dcep":
        return "dcep/dcep."+langs+"-test."+pair[:2]
    if corpus == "europarl":
        return "europarl/europarl-v7."+langs+"-test."+pair[:2]

def getDecodeFile(corpus, problem):
    pair = problem.split("_")[1]
    langs = pair[:2] + "-" + pair[2:]
    if corpus == "jrc":
        return "jrc/jrc_acquis."+langs+"-decode."+pair[2:]
    if corpus == "dcep":
        return "dcep/dcep."+langs+"-decode."+pair[2:]
    if corpus == "europarl":
        return "europarl/europarl-v7."+langs+"-decode."+pair[2:]

main()
