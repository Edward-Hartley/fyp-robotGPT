import logging
from datetime import datetime
import os
from fyp_package import config
from functools import wraps
import pickle

# Global run ID
run_id = 0
log_directory = ""

# Function to get next run ID
def get_next_run_id():
    run_id_file = config.run_id_file_path
    
    if not os.path.exists(run_id_file):
        with open(run_id_file, 'w') as f:
            f.write('0')
    
    with open(run_id_file, 'r') as f:
        run_id = int(f.read().strip())
    
    next_run_id = run_id + 1
    
    with open(run_id_file, 'w') as f:
        f.write(str(next_run_id))
    
    return next_run_id

def setup_logging():
    global run_id
    global log_directory
    run_id = get_next_run_id()
    log_directory = config.log_directory_path.format(run_id=run_id)
    os.mkdir(log_directory)
    log_filename = config.log_file_path.format(run_id=run_id)
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def log_event(event_type, details):
    logger = logging.getLogger()
    log_record = logging.LogRecord(
        name=logger.name,
        level=logging.INFO,
        pathname=__file__,
        lineno=0,
        msg=f"Event Type: {event_type}, Details: {details}",
        args=(),
        exc_info=None
    )
    timestamp = datetime.fromtimestamp(log_record.created).strftime('%Y-%m-%d %H:%M:%S.%f')
    logger.handle(log_record)
    return timestamp

def log_object(reponse, source, timestamp):
    # append response to log file
    log_response_obj = {
        'timestamp': timestamp,
        'source': source,
        'response': reponse
    }

    with open(f"{log_directory}/{source}.pkl", 'ab') as f:
        pickle.dump(log_response_obj, f)

def read_logged_results(filename):
    results = []
    with open(filename, 'rb') as f:
        while True:
            try:
                results.append(pickle.load(f))
            except EOFError:
                break
    return results

#### environment ####

def log_move_robot(move_robot):
    @wraps(move_robot)
    def wrapper(self, *args, **kwargs):
        current_pose = self.get_ee_pose()

        log_event('Move Robot Start', f"Current Pose: {current_pose}, Args: {args}, Kwargs: {kwargs}")

        result = move_robot(self, *args, **kwargs)

        log_event('Move Robot End', f"Result: {result}")

        return result
    return wrapper

def log_put_first_on_second(put_first_on_second):
    @wraps(put_first_on_second)
    def wrapper(self, *args, **kwargs):
        current_pose = self.get_ee_pose()

        log_event('Put First on Second Start', f"Current Pose: {current_pose}, Args: {args}, Kwargs: {kwargs}")

        result = put_first_on_second(self, *args, **kwargs)

        log_event('Put First on Second End', f"Result: {result}")

        return result
    return wrapper

def log_gripper_action(gripper_action):
    @wraps(gripper_action)
    def wrapper(self, *args, **kwargs):
        gripper_action_name = gripper_action.__name__

        log_event(f'Gripper Action Start', f"Action: {gripper_action_name}")

        result = gripper_action(self, *args, **kwargs)

        log_event(f'Gripper Action End', f"Result: {result}")

        return result
    return wrapper

#### models ####

def log_model_use(model_function):
    @wraps(model_function)
    def wrapper(self, *args, **kwargs):
        model_name = model_function.__name__.split('_predict')[0]
        model_hot = self.active_model is not None and self.active_model == model_name

        if model_name == 'langsam':
            # Access the 'prompt' argument (args[1] because 'self' is included in *args)
            prompt = args[1] if len(args) > 1 else kwargs.get('prompt', None)
            log_event('Model Use Start', f"Model: {model_name}, prompt: {prompt}, model_hot: {model_hot}")
        else:
            log_event('Model Use Start', f"Model: {model_name}, model_hot: {model_hot}")


        result = model_function(self, *args, **kwargs)
        timestamp = log_event('Model Use End', f"Model: {model_name}")
        log_object(result, model_name, timestamp)

        return result
    return wrapper

#### gpt ####

def log_chat_completion(chat_completion):
    @wraps(chat_completion)
    def wrapper(self, *args, **kwargs):
        name = self.name
        log_event('Chat Completion Start', f"Agent: {name}")

        completion = chat_completion(self, *args, **kwargs)
        usage = self.most_recent_usage

        timestamp = log_event('Chat Completion End', f"Agent: {name}, Usage: {usage}")
        log_object(
            {
                'usage': usage,
                'completion': completion
            },
            name + '_gpt_model',
            timestamp
        )

        return completion
    return wrapper

def log_message(name, message, generated):
    timestamp = log_event('Message Added', f"Agent: {name}, Generated: {generated}")

    log_object(
        {
            'message': message,
            'generated': generated,
        },
        name,
        timestamp
        )
    

#### object_detection_utils ####

def log_object_cube_calculations(detect_objects):
    @wraps(detect_objects)
    def wrapper(*args, **kwargs):
        segmentation_texts = args[1] if len(args) > 1 else kwargs.get('segmentation_texts', None)
        log_event('Object Detection Start', f"Segmentation Texts: {segmentation_texts}")

        result = detect_objects(*args, **kwargs)

        timestamp = log_event('Object Detection End', f"Result: {result}")
        log_object(result, 'object_cube', timestamp)

        return result
    return wrapper
    
#### configuration ####

def log_configuration(configuration, test_name):
    timestamp = log_event('Configuration Logged', f"run_id: {run_id}, test_name: {test_name}")

    log_object(
        {
            'configuration': configuration
        },
        'configuration',
        timestamp
        )


def print_run_id():
    print(f"Current run ID: {run_id}")

# Example usage
def main():
    setup_logging()

if __name__ == "__main__":
    main()
