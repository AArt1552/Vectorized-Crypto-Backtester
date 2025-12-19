import json
import itertools
import sys

def generate_and_save_parameters(config_file_name='config.json'):
    """Reads the config, generates all combinations, and saves to test_params.json."""
    print(f'📂 Loading parameters from {config_file_name}...')
    try:
        with open(config_file_name, 'r') as f:
            parameters_to_test = json.load(f)
        print('✅ Parameters loaded successfully.')
    except FileNotFoundError:
        print(f'❌ Error: File {config_file_name} not found.')
        sys.exit(1)
    except json.JSONDecodeError:
        print(f'❌ Error: File {config_file_name} contains invalid JSON.')
        sys.exit(1)

    sorted_keys = list(parameters_to_test.keys())
    # Exclude candle file paths from parameter combinations
    keys_to_combine = [k for k in sorted_keys if not k.startswith('candles_file')]
    values_for_itertools = [parameters_to_test[key] for key in keys_to_combine]

    generated_combinations = []
    for p in itertools.product(*values_for_itertools):
        combination = {key: [p[i]] for i, key in enumerate(keys_to_combine)}
        generated_combinations.append(combination)

    with open('test_params.json', 'w') as file:
        json.dump(generated_combinations, file, indent=4)
        
    total_combinations = len(generated_combinations)
    print(f'✅ test_params.json generated successfully.')
    
    return total_combinations

if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'config.json'
    total = generate_and_save_parameters(config_file)
    print(f'🔢 Total combinations generated: {total}\n')
