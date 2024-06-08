from fyp_package import config
from fyp_package.agent_logging import read_logged_results
import os
import re
from datetime import datetime
from collections import defaultdict

CONFIG_LOG = "configuration.pkl"
LOG_FILE = "logs.txt"
LANGSAM_LOG = "langsam.pkl"
OBJECT_CUBE_LOG = "object_cube.pkl"
ROBOT_AGENT_GPT_MODEL_LOG = "robot_agent_gpt_model.pkl"
ROBOT_AGENT_LOG = "robot_agent.pkl"
VISION_ASSISTANT_GPT_MODEL_LOG = "vision_assistant_gpt_model.pkl"
VISION_ASSISTANT_LOG = "vision_assistant.pkl"

def get_log_file(run_id, filename) -> str:
    return os.path.join(config.log_directory_path.format(run_id=run_id), filename)

def parse_log_time(log_time):
    return datetime.strptime(log_time, '%Y-%m-%d %H:%M:%S.%f')

def sum_dicts(dicts):
    # dictionaries may have nested dictionaries
    # all values are summable
    # dictionaries will be same structure
    # return a dictionary with the same structure
    # and the sum of the values
    summed_dict = {}
    for key in dicts[0].keys():
        if isinstance(dicts[0][key], dict):
            summed_dict[key] = sum_dicts([d[key] for d in dicts])
        else:
            summed_dict[key] = sum([d[key] for d in dicts])
    return summed_dict

def average_dicts(dicts):
    # dictionaries may have nested dictionaries
    # all values are summable
    # dictionaries will be same structure
    # return a dictionary with the same structure
    # and the average of the values
    num_dicts = len(dicts)
    averaged_dict = {}
    for key in dicts[0].keys():
        if isinstance(dicts[0][key], dict):
            averaged_dict[key] = average_dicts([d[key] for d in dicts])
        else:
            averaged_dict[key] = sum([d[key] for d in dicts]) / num_dicts
    return averaged_dict

def successful_run(run_id) -> bool:
    log_file = get_log_file(run_id, LOG_FILE)
    
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"Log file for run ID {run_id} does not exist.")
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # Regex to find the "Test complete" event with success status
    pattern = re.compile(r'Event Type: Test complete, Details: Success: (?P<success>True|False), Notes: (?P<notes>.+)')
    
    for line in lines:
        match = pattern.search(line)
        if match:
            success = match.group('success')
            return success == 'True'
    
    # If the "Test complete" event is not found
    raise ValueError(f"No 'Test complete' event found in the log for run ID {run_id}")

def successful_runs(run_ids) -> list[bool]:
    return [successful_run(run_id) for run_id in run_ids]

def success_rate(run_ids) -> float:
    successes = successful_runs(run_ids)
    return sum(successes) / len(successes)

def user_query(run_id) -> str:
    cfg_file = get_log_file(run_id, CONFIG_LOG)
    cfg = read_logged_results(cfg_file)[0]
    return (cfg['response']['configuration']['user_query'])

def test_name(run_id) -> str:
    cfg_file = get_log_file(run_id, CONFIG_LOG)
    cfg = read_logged_results(cfg_file)[0]
    return (cfg['response']['configuration']['test_name'])

def chat_generation_time_metrics(run_id) -> dict:
    log_file = get_log_file(run_id, LOG_FILE)
    
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"Log file for run ID {run_id} does not exist.")
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    agent_metrics = defaultdict(lambda: {
        'total_time_generating': 0.0,
        'total_completion_tokens': 0,
        'total_prompt_tokens': 0,
        'max_prompt_tokens': 0,
        'max_completion_tokens': 0
    })
    #initiate robot_agent and vision_assistant metrics
    agent_metrics['robot_agent'] = agent_metrics['robot_agent']
    agent_metrics['vision_assistant'] = agent_metrics['vision_assistant']

    current_starts = {}

    start_pattern = re.compile(r'(?P<time>.+?) - INFO - Event Type: Chat Completion Start, Details: Agent: (?P<agent>.+)')
    end_pattern = re.compile(r'(?P<time>.+?) - INFO - Event Type: Chat Completion End, Details: Agent: (?P<agent>.+?), Usage: CompletionUsage\(completion_tokens=(?P<completion_tokens>\d+), prompt_tokens=(?P<prompt_tokens>\d+), total_tokens=\d+\)')

    for line in lines:
        start_match = start_pattern.match(line)
        end_match = end_pattern.match(line)

        if start_match:
            log_time = start_match.group('time')
            agent = start_match.group('agent')
            current_starts[agent] = parse_log_time(log_time)
        
        if end_match:
            log_time = end_match.group('time')
            agent = end_match.group('agent')
            completion_tokens = int(end_match.group('completion_tokens'))
            prompt_tokens = int(end_match.group('prompt_tokens'))

            start_time = current_starts.pop(agent, None)
            if start_time:
                end_time = parse_log_time(log_time)
                time_generating = (end_time - start_time).total_seconds()
                agent_metrics[agent]['total_time_generating'] += time_generating

            agent_metrics[agent]['total_completion_tokens'] += completion_tokens
            agent_metrics[agent]['total_prompt_tokens'] += prompt_tokens
            agent_metrics[agent]['max_prompt_tokens'] = max(agent_metrics[agent]['max_prompt_tokens'], prompt_tokens)
            agent_metrics[agent]['max_completion_tokens'] = max(agent_metrics[agent]['max_completion_tokens'], completion_tokens)

    return agent_metrics

def chat_generation_time_metrics_all(run_ids) -> dict:
    all_metrics = [chat_generation_time_metrics(run_id) for run_id in run_ids]
    averaged_metrics = average_dicts(all_metrics)
    return averaged_metrics
    
def models_usage(run_id) -> dict:
    models_detect_strings = {
        "langsam": {
            "start": re.compile(
            r'(?P<time>.+?) - INFO - Event Type: Model Use Start, Details: Model: langsam, prompt: (?P<prompt>.+?), model_hot: (?P<hot>True|False)'
            ),
            "end": re.compile(
            r'(?P<time>.+?) - INFO - Event Type: Model Use End, Details: Model: langsam')
        },
        "graspnet": {
            "start": re.compile(
                r'(?P<time>.+?) - INFO - Event Type: Model Use Start, Details: Model: graspnet, model_hot: (?P<hot>True|False)'
                ),
            "end": re.compile(
            r'(?P<time>.+?) - INFO - Event Type: Model Use End, Details: Model: graspnet')
        },
    }

    models_usage = {
        "langsam": {
            'hot_time': 0.0,
            'cold_time': 0.0,
            'hot_calls': 0,
            'cold_calls': 0,
            'total_time': 0.0
        },
        "graspnet": {
            'hot_time': 0.0,
            'cold_time': 0.0,
            'hot_calls': 0,
            'cold_calls': 0,
            'total_time': 0.0
        },
    }

    log_file = get_log_file(run_id, LOG_FILE)
    
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"Log file for run ID {run_id} does not exist.")
    
    with open(log_file, 'r') as f:
        lines = f.readlines()

    current_starts = {}
    for line in lines:
        for model, detect_patterns in models_detect_strings.items():
            start_pattern = detect_patterns["start"]
            end_pattern = detect_patterns["end"]

            start_match = start_pattern.match(line)
            end_match = end_pattern.match(line)

            if start_match:
                log_time = start_match.group('time')
                hot = start_match.group('hot') == 'True'
                current_starts[model] = {
                    "time": parse_log_time(log_time),
                    "hot": hot
                }

            if end_match:
                log_time = end_match.group('time')
                start_info = current_starts.pop(model, None)
                if start_info:
                    end_time = parse_log_time(log_time)
                    time_generating = (end_time - start_info["time"]).total_seconds()
                    models_usage[model]['total_time'] += time_generating
                    if start_info["hot"]:
                        models_usage[model]['hot_time'] += time_generating
                        models_usage[model]['hot_calls'] += 1
                    else:
                        models_usage[model]['cold_time'] += time_generating
                        models_usage[model]['cold_calls'] += 1

    return models_usage

def models_usage_all(run_ids) -> dict:
    all_usage = [models_usage(run_id) for run_id in run_ids]
    average_dict = average_dicts(all_usage)
    return average_dict

def modules_usage(run_id) -> dict:
    control_detect_strings = {
        "object_cube": {
            "start": "Object Detection Start",
            "end": "Object Detection End"
        },
        "fine_control": {
            "start": "Move Robot Start",
            "end": "Move Robot End"
        },
        "gripper": {
            "start": "Gripper Action Start",
            "end": "Gripper Action End"
        },
        "put_first_on_second": {
            "start": "Put First on Second Start",
            "end": "Put First on Second End"
        }
    }

    modules_usage = {
        "object_cube": 0.0,
        "fine_control": 0.0,
        "gripper": 0.0,
        "put_first_on_second": 0.0
    }

    log_file = get_log_file(run_id, LOG_FILE)
    
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"Log file for run ID {run_id} does not exist.")
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    current_starts = {}
    for line in lines:
        for model, detect_strings in control_detect_strings.items():
            start_pattern = re.compile(r'(?P<time>.+?) - INFO - Event Type: ' + re.escape(detect_strings["start"]) + r'.*')
            end_pattern = re.compile(r'(?P<time>.+?) - INFO - Event Type: ' + re.escape(detect_strings["end"]) + r'.*')
            start_match = start_pattern.match(line)
            end_match = end_pattern.match(line)

            if start_match:
                log_time = start_match.group('time')
                current_starts[model] = parse_log_time(log_time)
            
            if end_match:
                log_time = end_match.group('time')
                model_time = parse_log_time(log_time)
                start_time = current_starts.pop(model, None)
                if start_time:
                    time_generating = (model_time - start_time).total_seconds()
                    modules_usage[model] += time_generating

    return modules_usage

def modules_usage_all(run_ids) -> dict:
    all_usage = [modules_usage(run_id) for run_id in run_ids]
    average_dict = average_dicts(all_usage)
    return average_dict

def total_time_for_run(run_id) -> float:
    log_file = get_log_file(run_id, LOG_FILE)
    
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"Log file for run ID {run_id} does not exist.")
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    start_pattern = re.compile(r'(?P<time>.+?) - INFO - Event Type: Configuration Logged')
    end_pattern = re.compile(r'(?P<time>.+?) - INFO - Event Type: Test complete')
    
    start_time = None
    end_time = None
    
    for line in lines:
        start_match = start_pattern.match(line)
        end_match = end_pattern.match(line)
        
        if start_match:
            log_time = start_match.group('time')
            start_time = parse_log_time(log_time)
        
        if end_match:
            log_time = end_match.group('time')
            end_time = parse_log_time(log_time)
    
    if start_time and end_time:
        return (end_time - start_time).total_seconds()
    else:
        raise ValueError(f"Run ID {run_id} did not have both a start and end event.")
    
def vision_test_success_mapping(run_id) -> float:
    # vision test includes multiple user queries and test complete logs
    # we want a mapping from query to whether it was successful
    # we can use the test complete logs to determine success
    log_file = get_log_file(run_id, LOG_FILE)

    if not os.path.exists(log_file):
        raise FileNotFoundError(f"Log file for run ID {run_id} does not exist.")

    with open(log_file, 'r') as f:
        lines = f.readlines()

    # Regex to find the "Test complete" event with success status
    pattern = re.compile(r'Event Type: Test complete, Details: Success: (?P<success>True|False), Notes: (?P<notes>.+)')
    query_pattern = re.compile(r'Event Type: User query, Details: (?P<query>.+)')

    query_success_mapping = {}
    current_query = None

    for line in lines:
        query_match = query_pattern.search(line)
        match = pattern.search(line)
        if query_match:
            current_query = query_match.group('query')
        if match and current_query != 'exit':
            success = match.group('success')
            query_success_mapping[current_query] = query_success_mapping.get(current_query, 0) + (success == 'True')

    return query_success_mapping

def print_runs_metrics(run_ids):
    print(test_name(run_ids[0]))
    print(user_query(run_ids[0]))
    print("Success rate:", success_rate(run_ids))
    chat_metrics = chat_generation_time_metrics_all(run_ids)
    print("Average chat generation time:", chat_metrics['robot_agent']['total_time_generating'] + chat_metrics['vision_assistant']['total_time_generating'])
    print("\tRobot agent:", chat_metrics['robot_agent'])
    print("\tVision assistant:", chat_metrics['vision_assistant'])
    models_usage = models_usage_all(run_ids)
    print("Average models usage:", models_usage['langsam']['total_time'] + models_usage['graspnet']['total_time'])
    print("\tLangsam:\n\t\tHot time:", models_usage['langsam']['hot_time'], "\n\t\tHot calls:", models_usage['langsam']['hot_calls'], "\n\t\tCold time:", models_usage['langsam']['cold_time'], "\n\t\tCold calls:", models_usage['langsam']['cold_calls'])
    print("\tGraspnet:\n\t\tHot time:", models_usage['graspnet']['hot_time'], "\n\t\tHot calls:", models_usage['graspnet']['hot_calls'], "\n\t\tCold time:", models_usage['graspnet']['cold_time'], "\n\t\tCold calls:", models_usage['graspnet']['cold_calls'])
    modules_usage = modules_usage_all(run_ids)
    print("Average modules usage:", sum(modules_usage.values()))
    print("\tObject cube:", modules_usage['object_cube'])
    print("\tFine control:", modules_usage['fine_control'])
    print("\tGripper:", modules_usage['gripper'])
    print("\tPut first on second:", modules_usage['put_first_on_second'])
    print()

def print_vision_test_metrics(vision_test_run_id):
    print("Run ID", vision_test_run_id)
    vision_gpt_usage = chat_generation_time_metrics(vision_test_run_id)['vision_assistant']
    print("total prompt tokens:", vision_gpt_usage['total_prompt_tokens'], "total completion tokens:", vision_gpt_usage['total_completion_tokens'])
    for i, success in vision_test_success_mapping(vision_test_run_id).items():
        print(f"\tquery {i}: {success}")
    print("Total success:", sum(vision_test_success_mapping(vision_test_run_id).values()))


def main():
    # full
    for i in range(8):
        run_ids = list(range(i*6+1, (i+1)*6+1))
        print_runs_metrics(run_ids)
        # print(total_time_for_run(run_ids[0])
    run_ids = list(range(91, 97))
    print_runs_metrics(run_ids)
        
    # remove modules
    for i in range(4):
        run_ids = list(range(i*3+49, (i+1)*3+49))
        # print("Run IDs", run_ids, ": ", user_query(run_ids[0]))
        print_runs_metrics(run_ids)

    #vision tests
    for i in [60, 61, 90]:
        vision_test_run_id = i
        # print vision assistant usage
        print_vision_test_metrics(vision_test_run_id)

    # prompt experiments
    for i in range(9):
        run_ids = list(range(i*3+63, (i+1)*3+63))
        print_runs_metrics(run_ids)



    for i in range(8):
        run_ids = list(range(i*6+1, (i+1)*6+1))
        print("Run IDs", run_ids, ": ", user_query(run_ids[0]))
        print("success rate:", success_rate(run_ids))

    # for i in [37, 38, 39, 40, 41, 42] :
    #     print("Run", i, "success:", successful_run(i))
    #     print("query:", user_query(i))

    # print_runs_metrics([59])

if __name__ == "__main__":
    main()
