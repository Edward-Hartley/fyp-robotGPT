from fyp_package import config
from fyp_package.agent_logging import read_logged_results
import os
import re
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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
    original_query = (cfg['response']['configuration']['user_query'])
    return task_name_merger(original_query)

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
            # dont want to count gripper actions within put first on second
            if model == "gripper" and "put_first_on_second" in current_starts:
                continue

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
    
def total_time_for_runs(run_ids) -> float:
    return sum([total_time_for_run(run_id) for run_id in run_ids])

def average_time_for_runs(run_ids) -> float:
    return total_time_for_runs(run_ids) / len(run_ids)

def display_image_used(run_id) -> bool:
    chat_file = get_log_file(run_id, VISION_ASSISTANT_LOG)
    if not os.path.exists(chat_file):
        return False
    
    logs = read_logged_results(chat_file)
    for log in logs:
        message = log['response']['message']
        if isinstance(message['content'], list):
            return True
    return False
    
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
    print("Average time:", average_time_for_runs(run_ids))
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
    print("\tDisplay used image:", any(display_image_used(run_id) for run_id in run_ids))
    print()


    return {
        "test_name": test_name(run_ids[0]),
        "user_query": user_query(run_ids[0]),
        "success_rate": success_rate(run_ids),
        "average_time": average_time_for_runs(run_ids),
        "average_chat_generation_time": chat_metrics['robot_agent']['total_time_generating'] + chat_metrics['vision_assistant']['total_time_generating'],
        "robot_agent": chat_metrics['robot_agent'],
        "vision_assistant": chat_metrics['vision_assistant'],
        "average_models_usage": models_usage['langsam']['total_time'] + models_usage['graspnet']['total_time'],
        "langsam": {
            "hot_time": models_usage['langsam']['hot_time'],
            "hot_calls": models_usage['langsam']['hot_calls'],
            "cold_time": models_usage['langsam']['cold_time'],
            "cold_calls": models_usage['langsam']['cold_calls'],
            "total_time": models_usage['langsam']['total_time']
        },
        "graspnet": {
            "hot_time": models_usage['graspnet']['hot_time'],
            "hot_calls": models_usage['graspnet']['hot_calls'],
            "cold_time": models_usage['graspnet']['cold_time'],
            "cold_calls": models_usage['graspnet']['cold_calls'],
            "total_time": models_usage['graspnet']['total_time']
        },
        "average_modules_usage": sum(modules_usage.values()),
        "object_cube": modules_usage['object_cube'],
        "fine_control": modules_usage['fine_control'],
        "gripper": modules_usage['gripper'],
        "put_first_on_second": modules_usage['put_first_on_second'],
        "display_used_image": sum(display_image_used(run_id) for run_id in run_ids) > 3
    }

def time_graph(tasks_data):
    tasks = [short_task_name(task['user_query']) for task in tasks_data]
    data = {
        'total_time': [task['average_time'] for task in tasks_data],
        'generation_time_top_level': [task['robot_agent']['total_time_generating'] for task in tasks_data],
        'generation_time_vision_assistant': [task['vision_assistant']['total_time_generating'] for task in tasks_data],
        'langsam_time': [task['langsam']['total_time'] for task in tasks_data],
        'graspnet_time': [task['graspnet']['total_time'] for task in tasks_data],
        'object_cube_calculation': [task['object_cube'] for task in tasks_data],
        'put_first_on_second_time': [task['put_first_on_second'] for task in tasks_data],
        'fine_control_time': [task['fine_control'] + task['gripper'] for task in tasks_data]
    }


    # Define colors for different categories
    colors = {
        'generation': ['#66c2a5', '#1b9e77'],
        'perception': ['#decbe4', '#c994c7', '#984ea3'],
        'control': ['#80b1d3', '#377eb8']
    }

    patterns = {
        'generation': ['-', '|'],
        'perception': ['-', '|', '/'],
        'control': ['-', '|']
    }

    # List of all components
    components = [
        'generation_time_top_level',
        'generation_time_vision_assistant',
        'langsam_time',
        'graspnet_time',
        'object_cube_calculation',
        'put_first_on_second_time',
        'fine_control_time'
    ]

    # Labels for the components
    component_labels = {
        'generation_time_top_level': 'Top-Level Generation',
        'generation_time_vision_assistant': 'Vision Assistant Generation',
        'langsam_time': 'LangSAM',
        'graspnet_time': 'GraspNet',
        'object_cube_calculation': 'Object Cube Calculation',
        'put_first_on_second_time': 'Put First on Second',
        'fine_control_time': 'Fine Control'
    }

    # Mapping components to colors
    component_colors = {
        'generation_time_top_level': colors['generation'][0],
        'generation_time_vision_assistant': colors['generation'][1],
        'langsam_time': colors['perception'][0],
        'graspnet_time': colors['perception'][1],
        'object_cube_calculation': colors['perception'][2],
        'put_first_on_second_time': colors['control'][0],
        'fine_control_time': colors['control'][1]
    }

    component_patterns = {
        'generation_time_top_level': patterns['generation'][0],
        'generation_time_vision_assistant': patterns['generation'][1],
        'langsam_time': patterns['perception'][0],
        'graspnet_time': patterns['perception'][1],
        'object_cube_calculation': patterns['perception'][2],
        'put_first_on_second_time': patterns['control'][0],
        'fine_control_time': patterns['control'][1]
    }

    # Calculate the bottom positions for each component
    bottom = np.zeros(len(tasks))

    # Create the plot
    plt.figure(figsize=(10, 4.3))
    plt.grid(axis='y', linestyle='--', color='gray', alpha=0.7, zorder=1)
    # Create handles for the legend
    handles = []
    labels = []

    # Plot each component
    for component in components:
        values = data[component]
        bar = plt.bar(tasks, values, bottom=bottom, color=component_colors[component], label=component_labels[component], zorder=2)
        bottom += np.array(values)
        # Collect handles and labels for reversing later
        handles.append(bar)
        labels.append(component_labels[component])

    # Reverse the order of handles and labels for the legend
    handles = handles[::-1]
    labels = labels[::-1]

    # Add communal legend with reversed order
    plt.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.35, 1), title="Components")

    plt.xticks(rotation=24, ha='right')

    # Add titles and labels
    plt.title('Task Time Breakdown by Component')
    plt.ylabel('Time (s)')

    # Show the plot
    plt.tight_layout()
    plt.show()

    print("proportion of time spent on generating and models: " + str((sum(data['generation_time_top_level']) + sum(data['generation_time_vision_assistant']) + sum(data['langsam_time']) + sum(data['graspnet_time']) + sum(data['object_cube_calculation'])) / (sum(data['fine_control_time'] + data['put_first_on_second_time']))))

def results_table(tasks):
    # List of all modules
    modules_list = ["Vision", "Detect Object", "Fine Control", "Put First on Second", "Graspnet", "Display Image"]

    # Define a function to generate ticks for LaTeX
    def get_tick(module_used):
        return "\\checkmark" if module_used else ""

    # Start the LaTeX table
    latex_table = """
    \\begin{table}[h\centering]
    \\centering
    \\begin{tabular}{|l|c|""" + "c|" * len(modules_list) + """}
    \hline
    \textbf{Task} & \textbf{Success Rate} & \multicolumn{6}{|c|}{\textbf{Modules Used}} \\
    \cline{3-8}
    &  & \textbf{Vision} & \textbf{LangSAM} & \textbf{Fine Control} & \textbf{Pick and Place} & \textbf{Graspnet} & \textbf{Display Image} \\
    \hline
    """

    # Add each task and its modules to the table
    for task in tasks:
        modules_used = "Vision"
        if task['object_cube'] > 0:
            modules_used += ", Detect Object"
        if task['fine_control'] > 0:
            modules_used += ", Fine Control"
        if task['put_first_on_second'] > 0:
            modules_used += ", Put First on Second"
        if task['graspnet']['cold_calls'] > 0:
            modules_used += ", Graspnet"
        if task['display_used_image']:
            modules_used += ", Display Image"

        latex_table += f"{task['user_query']} & {task['success_rate']:.2f} "
        for module in modules_list:
            latex_table += f"& {get_tick(module in modules_used)} "
        latex_table += "\\\\ \\hline\n"

    # End the LaTeX table
    latex_table += """
    \\end{tabular}
    \\caption{Module usage per task with success rates. A tick indicates module usage.}
    \\label{tab:module_usage}
    \\end{table}
    """

    print(latex_table)

def failure_modes_chart():

    # Sample data for the two tasks
    hard_task_failures = {
        'Perception Errors': 1,
        'Movement Errors': 1,
        'Generation Errors': 0,
        'Planning Errors': 8,
    }

    successful_task_failures = {
        'Perception Errors': 2,
        'Movement Errors': 2,
        'Generation Errors': 1,
        'Planning Errors': 1,
    }

    # Prepare the data
    labels = list(successful_task_failures.keys())
    task1_counts = list(hard_task_failures.values())
    task2_counts = list(successful_task_failures.values())

    # colour-blind friendly palette
    colors = sns.color_palette("colorblind", len(labels))

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # Plot the first pie chart
    wedges, texts, autotexts = ax1.pie(task1_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax1.set_title('Failure Modes in Largely Unsuccessful Tasks')

    # Plot the second pie chart
    wedges, texts2, autotexts2 = ax2.pie(task2_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax2.set_title('Failure Modes in Mostly Successful Tasks')

    # Add a communal legend
    fig.legend(wedges, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, 0.05))

    # Neat configuration for text and layout
    for text in texts + autotexts + texts2 + autotexts2:
        text.set_fontsize(10)
    for autotext in autotexts + autotexts2:
        autotext.set_color('white')
        autotext.set_weight('bold')

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    # Show the plot
    plt.show()

def prompt_ablations_table_rows(original_tasks, prompt_tasks, example_tasks):
    tasks = []
    original_success_rate = []
    prompt_success_rate = []
    example_success_rate = []
    original_gpt_total_input = []
    prompt_gpt_total_input = []
    example_gpt_total_input = []
    original_gpt_total_output = []
    prompt_gpt_total_output = []
    example_gpt_total_output = []

    for task_name in prompt_tasks.keys():
        original_task = original_tasks[task_name]
        prompt_task = prompt_tasks[task_name]
        example_task = example_tasks[task_name]

        tasks.append(task_name)
        original_success_rate.append(original_task['success_rate'])
        prompt_success_rate.append(prompt_task['success_rate'])
        example_success_rate.append(example_task['success_rate'])
        original_gpt_total_input.append(original_task['robot_agent']['total_prompt_tokens'] + original_task['vision_assistant']['total_prompt_tokens'])
        prompt_gpt_total_input.append(prompt_task['robot_agent']['total_prompt_tokens'] + prompt_task['vision_assistant']['total_prompt_tokens'])
        example_gpt_total_input.append(example_task['robot_agent']['total_prompt_tokens'] + example_task['vision_assistant']['total_prompt_tokens'])
        original_gpt_total_output.append(original_task['robot_agent']['total_completion_tokens'] + original_task['vision_assistant']['total_completion_tokens'])
        prompt_gpt_total_output.append(prompt_task['robot_agent']['total_completion_tokens'] + prompt_task['vision_assistant']['total_completion_tokens'])
        example_gpt_total_output.append(example_task['robot_agent']['total_completion_tokens'] + example_task['vision_assistant']['total_completion_tokens'])

    # Define a color-blind friendly palette
    colors = sns.color_palette("colorblind", 3)

    # Sample Data

    summary_values = [6858, 5250, 2286]

    # Number of groups
    n_groups = len(tasks)

    # Bar Width
    bar_width = 0.25

    # Index for groups
    index = np.arange(n_groups)

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot Summary on top left
    axes[0, 0].bar(['Original', 'Minimal Prompt', 'Remove Examples'], summary_values, color=colors)
    axes[0, 0].set_title('Initial Prompt Size by Experiment')
    axes[0, 0].set_ylabel('Prompt Size in Tokens')

    # Plot Success Rates on top right
    axes[0, 1].bar(index, original_success_rate, bar_width, label='Original', color=colors[0])
    axes[0, 1].bar(index + bar_width, prompt_success_rate, bar_width, label='Minimal Prompt', color=colors[1])
    axes[0, 1].bar(index + 2 * bar_width, example_success_rate, bar_width, label='Remove Examples', color=colors[2])
    axes[0, 1].set_title('Success Rates by Task')
    axes[0, 1].set_ylabel('Success Rate')
    axes[0, 1].set_xticks(index + bar_width)
    axes[0, 1].set_xticklabels(tasks)
    axes[0, 1].legend(loc='upper left')

    # Plot Total Input on bottom left
    axes[1, 0].bar(index, original_gpt_total_input, bar_width, label='Original', color=colors[0])
    axes[1, 0].bar(index + bar_width, prompt_gpt_total_input, bar_width, label='Minimal Prompt', color=colors[1])
    axes[1, 0].bar(index + 2 * bar_width, example_gpt_total_input, bar_width, label='Remove Examples', color=colors[2])
    axes[1, 0].set_title('Total Input Tokens by Task')
    axes[1, 0].set_ylabel('Total Input Tokens')
    axes[1, 0].set_xticks(index + bar_width)
    axes[1, 0].set_xticklabels(tasks)
    axes[1, 0].legend()

    # Plot Total Output on bottom right
    axes[1, 1].bar(index, original_gpt_total_output, bar_width, label='Original', color=colors[0])
    axes[1, 1].bar(index + bar_width, prompt_gpt_total_output, bar_width, label='Minimal Prompt', color=colors[1])
    axes[1, 1].bar(index + 2 * bar_width, example_gpt_total_output, bar_width, label='Remove Examples', color=colors[2])
    axes[1, 1].set_title('Total Output Tokens by Task')
    axes[1, 1].set_ylabel('Total Output Tokens')
    axes[1, 1].set_xticks(index + bar_width)
    axes[1, 1].set_xticklabels(tasks)
    axes[1, 1].legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()

def print_vision_test_metrics(vision_test_run_id):
    print("Run ID", vision_test_run_id)
    vision_gpt_usage = chat_generation_time_metrics(vision_test_run_id)['vision_assistant']
    print("total prompt tokens:", vision_gpt_usage['total_prompt_tokens'], "total completion tokens:", vision_gpt_usage['total_completion_tokens'])
    for i, success in vision_test_success_mapping(vision_test_run_id).items():
        print(f"\t {i}: {success}")
    print("Total success:", sum(vision_test_success_mapping(vision_test_run_id).values()))

def task_name_merger(task_name):
    mapping = {
        "Arrange the cans in a line extending to the left of the paper cup." : "Move the can which is to the right of the bowl, to in front of the other can.",
        "Put the can which is furthest from the bowl into the bowl" : "Move the can which is to the right of the bowl, to in front of the other can.",
        "Move the can which is to the right of the bowl, to in front of the other can." : "Move the can which is to the right of the bowl, to in front of the other can.",
        "Knock over the bottle" : "Knock over the bottle.",
        "Knock over the bottle." : "Knock over the bottle.",
        "Knock over the bottle, then put a sponge on the \"spill\" you made." : "Knock over the bottle, then put a sponge on the \"spill\" you made.",
        "Knock over the bottle, then put a sponge on the \"spill\" you make." : "Knock over the bottle, then put a sponge on the \"spill\" you made.",
        "Move the lonely object to the others" : "Move the lonely object to the others.",
        "Move the lonely object to the others." : "Move the lonely object to the others.",
        "My child is hungry, please pour him a bowl of chips." : "My child is hungry, please pour him a bowl of chips.",
        "My child is hungry, please pour him a bowl of chips. (The chips themselves are fake and made of cardboard, this is okay)" : "My child is hungry, please pour him a bowl of chips.",
        "My child is hungry, please pour him a plate of chips. (The chips themselves are fake and made of cardboard, this is okay)" : "My child is hungry, please pour him a bowl of chips.",
        "Put an empty paper cup on the plate." : "Put the empty frying pan on the plate.",
        "Put the empty cup on the plate." : "Put the empty frying pan on the plate.",
        "Put the empty frying pan on the plate." : "Put the empty frying pan on the plate.",
        "Shake the can" : "Shake the coke can",
        "Shake the coke can" : "Shake the coke can",
        "Stack the three objects steadily." : "Stack the three objects steadily.",
        "Wipe the table with the sponge, while avoiding the bowl on the table." : "Wipe the table with the sponge, while avoiding the bowl on the table.",
        "Wipe the table with the sponge, while avoiding the coke can on the table." : "Wipe the table with the sponge, while avoiding the bowl on the table.",
        "Varied user query to test vision agent capabilities." : "Varied user query to test vision agent capabilities.",
    }
    return mapping[task_name]

def short_task_name(task_name):
    mapping = {
        "Knock over the bottle, then put a sponge on the \"spill\" you made." :  "Clean spill",
        "Knock over the bottle." : "Knock over",
        "Move the can which is to the right of the bowl, to in front of the other can." : "Arrange Cans",
        "Move the lonely object to the others." : "Lonely object",
        "My child is hungry, please pour him a bowl of chips." : "Pour chips",
        "Put the empty frying pan on the plate." : "Empty Frying Pan",
        "Shake the coke can" : "Shake can",
        "Stack the three objects steadily." : "Stack",
        "Wipe the table with the sponge, while avoiding the bowl on the table." : "Wipe table",
        "Varied user query to test vision agent capabilities." : "Vision test"
    }
    return mapping[task_name]

def main():
    # full
    tasks = []
    for i in range(8):
        run_ids = list(range(i*6+1, (i+1)*6+1))
        tasks.append(print_runs_metrics(run_ids))
        # print(total_time_for_run(run_ids[0])
    run_ids = list(range(91, 97))
    tasks.append(print_runs_metrics(run_ids))
    # results_table(tasks)
    # time_graph(tasks)
        
    # remove_modules_tasks = []
    # # remove modules
    # for i in range(4):
    #     run_ids = list(range(i*3+49, (i+1)*3+49))
    #     # print("Run IDs", run_ids, ": ", user_query(run_ids[0]))
    #     remove_modules_tasks.append(print_runs_metrics(run_ids))

    # time_graph(remove_modules_tasks)

    #vision tests
    for i in [61, 62, 90]:
        vision_test_run_id = i
        # print vision assistant usage
        print_vision_test_metrics(vision_test_run_id)

    # prompt_experiments = []
    # # prompt experiments
    # for i in range(9):
    #     run_ids = list(range(i*3+63, (i+1)*3+63))
    #     prompt_experiments.append(print_runs_metrics(run_ids))    
    # for i in range(3):
    #     run_ids = list(range(i*3+97, (i+1)*3+97))
    #     prompt_experiments.append(print_runs_metrics(run_ids))

    # tasks_dict = {}
    # for task in tasks:
    #     tasks_dict[short_task_name(task['user_query'])] = task
    # minimal_prompt_experiments_dict = {}
    # few_prompt_experiments_dict = {}
    # for task in prompt_experiments:
    #     if task['test_name'] == "minimal_top_prompt":
    #         minimal_prompt_experiments_dict[short_task_name(task['user_query'])] = task
    #     elif task['test_name'] == "remove_few_shot_examples_top_and_vision":
    #         few_prompt_experiments_dict[short_task_name(task['user_query'])] = task

    # prompt_ablations_table_rows(tasks_dict, minimal_prompt_experiments_dict, few_prompt_experiments_dict)


    # for i in range(8):
    #     run_ids = list(range(i*6+1, (i+1)*6+1))
    #     print("Run IDs", run_ids, ": ", user_query(run_ids[0]))
    #     print("success rate:", success_rate(run_ids))

    # for i in [37, 38, 39, 40, 41, 42] :
    #     print("Run", i, "success:", successful_run(i))
    #     print("query:", user_query(i))

    # print_runs_metrics([59])

    # failure_modes_chart()

    experiment_set = set()
    configuration_set = set()
    for i in range(1, 106):
        run_id = i
        configuration = test_name(run_id)
        experiment = (test_name(run_id), user_query(run_id))
        configuration_set.add(configuration)
        experiment_set.add(experiment)

    # print("Configurations:", configuration_set)
    # all_features = 'all_features'
    # # filter experiments set
    # all_features_experiments = experiment_set.copy()
    # for experiment in experiment_set:
    #     if all_features not in experiment:
    #         all_features_experiments.remove(experiment)
    experiments_list = list(set([experiment[1] for experiment in experiment_set]))
    experiments_list.sort()
    for experiment in experiments_list:
        print(experiment)


if __name__ == "__main__":
    main()
