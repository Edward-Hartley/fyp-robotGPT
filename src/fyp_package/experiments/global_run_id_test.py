import fyp_package.config as config
import fyp_package.agent_logging as agent_logging

def test_global_run_id():
    print(config.simulation)

def test_print_run_id():
    agent_logging.print_run_id()