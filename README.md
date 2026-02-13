# MySimAre

Quick start:
```
python -m simulation.simulation.runner --version dynamic-knowledge-state --input_csv D:\MySimAre\Data\competition_math\data\train_fixed_level_4_5.csv --knowledge_level intermediate --dynamic_knowledge_state_init --num_conversations 1
```

## Overview
MySimAre is a modular math‑tutoring simulation pipeline with IU‑graph extraction, dynamic knowledge state updates, and end‑to‑end conversation simulation.

## Requirements
Install dependencies:
```
pip install -r requirements.txt
```

## Quick start (more examples)
Run one conversation with dynamic knowledge state:
```
python -m simulation.simulation.runner --version dynamic-knowledge-state --input_csv D:\MySimAre\Data\competition_math\data\train_fixed_level_4_5.csv --knowledge_level intermediate --dynamic_knowledge_state_init --num_conversations 1
```

Run with a specific IU extraction model:
```
python -m simulation.simulation.runner --version dynamic-knowledge-state --input_csv D:\MySimAre\Data\competition_math\data\train_fixed_level_4_5.csv --knowledge_level intermediate --dynamic_knowledge_state_init --num_conversations 1 --iu_model gpt-4o-mini
```

Run without dynamic knowledge state init:
```
python -m simulation.simulation.runner --version dynamic-knowledge-state --input_csv D:\MySimAre\Data\competition_math\data\train_fixed_level_4_5.csv --num_conversations 1
```

## Tools
### Conversation visualization
```
python D:\MySimAre\simulation\tools\visualize_conversations.py
```
Output:
```
D:\MySimAre\output\competition_math\gpt-5-mini\dynamic-knowledge-state.html
```

### LLM call log visualization
```
python D:\MySimAre\simulation\tools\visualize_llm_calls.py
```
Output:
```
D:\MySimAre\logs\llm_calls.html
```

## Logs
LLM call logging and printing are configured in:
```
simulation/config.json
```

Example:
```json
{
  "log_llm_calls": true,
  "log_llm_path": "logs/llm_calls_{timestamp}.jsonl",
  "print_llm_calls": true
}
```

## Outputs
Conversations are saved to:
```
output\competition_math\<model>\dynamic-knowledge-state_<timestamp>.json
```

## Notes
- Prompts live in `simulation/prompts`.
- IU graph extraction uses the prompt in `simulation/prompts/iu_graph_extraction.txt`.

