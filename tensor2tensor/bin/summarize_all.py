import os

TRANSLATE_PROBLEMS = [
    "summarize_cs_legal32k",
    "summarize_de_legal32k",
    "summarize_en_legal32k",
    "summarize_es_legal32k",
    "summarize_fr_legal32k",
    "summarize_it_legal32k",
    "summarize_sv_legal32k",
]


def main():
    for problem in TRANSLATE_PROBLEMS:
        os.system("mkdir -p $TRAIN_DIR/summarize/"+problem);
        successful = False
        while not successful:
            numbers = []
            for f in os.listdir(os.environ['TRAIN_DIR']+"/summarize/"+problem):
                if f.endswith(".index"):
                    numbers.append(int(f.split("-")[1].split(".")[0]))
            if (len(numbers)) > 0 :
                steps = 250000 - max(numbers)
            else:
                steps = 250000

            print(steps)
            cmd = "python ./t2t-trainer --data_dir=$DATA_DIR/summarize/"+problem+" --output_dir=$TRAIN_DIR/summarize/"+problem+" --worker_gpu=4 --training_steps="+str(steps)+" --model=multi_model --hparams_set=multimodel_legal --problems="+problem
            if os.system(cmd) == 0:
                successful = True

main()
