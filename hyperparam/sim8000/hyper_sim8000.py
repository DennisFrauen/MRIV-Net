import misc
import yaml
from hyperparam.main import run_hyper_tuning

if __name__ == "__main__":
    # Select configuration file here
    config_name = "hyper_sim8000_config.yaml"
    stream = open(misc.get_project_path() + "/hyperparam/sim8000/" + config_name, 'r')
    run_config = yaml.safe_load(stream)
    run_hyper_tuning(run_config)