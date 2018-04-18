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

METRICS = [
    #"bleu"
    "chrf"
]

CORPORA = [
    "dcep",
    "europarl",
    "jrc"
]

def main():
    model = "multi_model"
    hparams = "multimodel_legal"
    decode_params = "beam4.alpha0.6"
    chrf_order = 6
    chrf_beta = 3

    for metric in METRICS:
        print(metric)
        for corpus in CORPORA:
            print(corpus)
            for problem in TRANSLATE_PROBLEMS:
                print(problem)
                decoded = os.environ['DECODE_DIR']+"/"+getDecodePrefix(corpus, problem)+"."+model+"."+hparams+"."+problem+"."+decode_params+".decodes"
                test = os.environ['DECODE_DIR']+"/"+getRefFile(corpus, problem)
                langpair = getLangpair(problem)

                if not (os.path.isfile(decoded) and os.path.isfile(test)):
                    print("{} {} not existing".format(decoded, test))
                    continue

                num_lines1 = sum(1 for line in open(decoded))
                num_lines2 = sum(1 for line in open(test))
                printcmd = "cat"
                if num_lines1 > num_lines2:
                    printcmd = "head -n -"+str(num_lines1-num_lines2)

                cmd = printcmd+" " + decoded + " | sacrebleu --metrics " + metric + " --chrf-order 6 --chrf-beta 3 --short --force --score-only --tokenize intl " + test
                #print(cmd)
                if os.system(cmd) == 0:
                    continue

def getDecodePrefix(corpus, problem):
    pair = problem.split("_")[1]
    langs = pair[:2] + "-" + pair[2:]
    if corpus == "jrc":
        return "jrc/jrc_acquis"
    if corpus == "dcep":
        return "dcep/dcep"
    if corpus == "europarl":
        return "europarl/europarl-v7"

def getRefFile(corpus, problem):
    pair = problem.split("_")[1]
    langs = pair[:2] + "-" + pair[2:]
    if corpus == "jrc":
        return "jrc/jrc_acquis."+langs+"-test."+pair[2:]
    if corpus == "dcep":
        return "dcep/dcep."+langs+"-test."+pair[2:]
    if corpus == "europarl":
        return "europarl/europarl-v7."+langs+"-test."+pair[2:]

def getLangpair(problem):
    pair = problem.split("_")[1]
    return pair[:2] + "-" + pair[2:]

main()
