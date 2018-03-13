import os
import sys
import json
import shutil
import pprint

from experiments.hmmvsrnn_reco.a_data import tmpdir

jobfile = sys.argv[-1]
while True:
    todo = None

    with open(jobfile) as f:
        joblist = json.load(f)

    for job in joblist:
        if not os.path.exists(os.path.join(tmpdir, job['name'] + ".dat")):
            todo = job
            break
        if not os.path.exists(os.path.join(tmpdir, job['name'] + ".ipynb")):
            todo = job
            break

    if todo is None:
        sys.exit(0)

    pprint.pprint(todo)

    os.putenv('EXPERIMENT_NAME', todo['name'])
    os.putenv('PYTHONPATH', ":".join(sys.path + [os.getcwd()]))
    python = sys.executable

    script = "experiments/hmmvsrnn_reco/d_train_" + todo['model'] + ".py"
    args = [
        script,
        "--name='" + todo['name'] + "'",
        "--modality='" + todo['modality'] + "'",
        "--variant='" + todo['variant'] + "'",
        "--notes='" + todo['notes'] + "'",
        "--batch_size=" + str(todo['batch_size']),
        "--max_time=" + str(todo['max_time']),
        "--encoder_kwargs=" + "'" + json.dumps(todo['encoder_kwargs']) + "'"] + (
        ["--date", todo['date']] if 'date' in todo else [])
    ret = os.system(python + " " + " ".join(args))
    if ret != 0:
        sys.exit(-1)

    args = ['-m', 'jupyter', 'nbconvert',
            "experiments/hmmvsrnn_reco/e_analyse_" + todo['model'] + ".ipynb",
            "--to", "notebook", "--execute", "--ExecutePreprocessor.timeout=360",
            "--output", todo['name'] + ".ipynb"]
    ret = os.system(python + " " + " ".join(args))
    if ret != 0:
        sys.exit(-1)

    shutil.move(os.path.join("experiments/hmmvsrnn_reco/", todo['name'] + ".ipynb"),
                os.path.join(tmpdir, todo['name'] + ".ipynb"))
