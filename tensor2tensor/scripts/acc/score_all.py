import os

LABEL_PROBLEMS = [
    "multi_labeling_cs_legal32k",
    "multi_labeling_de_legal32k",
    "multi_labeling_en_legal32k",
    "multi_labeling_es_legal32k",
    "multi_labeling_fr_legal32k",
    "multi_labeling_it_legal32k",
    "multi_labeling_sv_legal32k"
]

TYPES = [
    "single",
    "joint"
]

def main():
    for problem in LABEL_PROBLEMS:
        for type in TYPES:
            lang = problem.split("_")[2]
            decode_file = "./data/"+type+".multi_model.multimodel_legal."+problem+".beam4.alpha0.6.decodes"
            reference_file = "./data/jrc_acquis."+lang+"-test.labels"
            print("problem: {} type: {}".format(problem, type))
            cmd = "python score.py -d " + decode_file + " -r " + reference_file
            os.system(cmd)
            print("============================================")

main()
