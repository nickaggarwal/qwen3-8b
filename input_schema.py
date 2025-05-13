INPUT_SCHEMA = {
    "prompt": {
        'datatype': 'STRING',
        'required': True,
        'shape': [1],
        'example': ["Write a short story about a robot learning to paint:"]
    },
    "min_tokens": {
        'datatype': 'INT16',
        'required': False,
        'shape': [1],
        'example': [0]
    },
    "max_tokens": {
        'datatype': 'INT16',
        'required': False,
        'shape': [1],
        'example': [128]
    },
    "temperature": {
        'datatype': 'FP32',
        'required': False,
        'shape': [1],
        'example': [1.0]
    },
    "top_p": {
        'datatype': 'FP32',
        'required': False,
        'shape': [1],
        'example': [1.0]
    },
    "top_k": {
        'datatype': 'INT8',
        'required': False,
        'shape': [1],
        'example': [50]
    },
    "repetition_penalty": {
        'datatype': 'FP32',
        'required': False,
        'shape': [1],
        'example': [1.0]
    }
}
