# Copyright (c) ModelScope Contributors. All rights reserved.
# Copyright (c) 2026 FlagOS Contributors. All rights reserved.
#
# Vendored from ms-swift (Apache-2.0): swift/loss_scale/utils.py, plus
# ``split_str_parts_by`` / ``_split_str_by_regex`` from swift/template/utils.py.
# The two splitters live here so this package does not import swift.template.

import re
from typing import Dict, List, Optional, Tuple


def _split_str_by_regex(text: str, regex_delimiters: List[str]) -> List[str]:
    combined_pattern = '|'.join(f'({pattern})' for pattern in regex_delimiters)
    parts = re.split(combined_pattern, text, flags=re.DOTALL)
    parts = [part for part in parts if part is not None]
    if parts[0] == '':
        parts.pop(0)
    else:
        parts.insert(0, '')
    assert len(parts) % 2 == 0, f'result: {parts}'
    assert ''.join(parts) == text, f'split_result: {parts}, text: {text}'
    return parts


def split_str_parts_by(text: str, delimiters: List[str], regex_mode: bool = False) -> List[Dict[str, str]]:
    """Split the text field into parts.

    Args:
        text: A text to be split.
        delimiters: The delimiters.

    Returns:
        The split text in list of dicts.
    """
    assert isinstance(text, str), f'text: {text}'
    delimiters_origin = delimiters
    if not regex_mode:
        delimiters = [re.escape(delimiter) for delimiter in delimiters]
    parts = _split_str_by_regex(text, delimiters) if delimiters else ['', text]
    res = []
    if regex_mode:
        parts = [part for part in parts if part]
        for part in parts:
            for delimiter, delimiter_origin in zip(delimiters, delimiters_origin):
                if re.match(delimiter, part, re.DOTALL):
                    break
            else:
                delimiter_origin = ''
            res.append({'key': delimiter_origin, 'content': part})
    else:
        for key, content in zip(parts[::2], parts[1::2]):
            res.append({'key': key, 'content': content})
    return res


def calculate_loss_scale(query: Optional[str],
                         response: str,
                         response_loss_scale_map: Dict[str, list],
                         query_loss_scale_map: Optional[Dict[str, list]] = None) -> Tuple[List[str], List[float]]:
    """Calculate the loss scale by splitting the agent response.

    This algorithm comes from paper: https://arxiv.org/pdf/2309.00986.pdf

    Agent response format:

    ```text
        Thought: you should always think about what to do
        Action: the action to take, should be one of the above tools[fire_recognition,
            fire_alert, call_police, call_fireman]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question
    ```
    Returns:
        A tuple of agent response parts and their weights.
    """
    # query loss scale map
    if query_loss_scale_map is not None and query is not None:
        for key in query_loss_scale_map.keys():
            if key in query:
                if isinstance(query_loss_scale_map[key], (float, int)):
                    query_loss_scale_map[key] = [query_loss_scale_map[key]]
                loss_scale_value = query_loss_scale_map[key][0]
                return [response], [float(loss_scale_value)]
    delimiters = [k for k, v in response_loss_scale_map.items() if len(v) == 2]
    if delimiters:
        agent_parts = split_str_parts_by(response, delimiters)
    else:
        regex_delimiters = [k for k, v in response_loss_scale_map.items() if len(v) == 1]
        agent_parts = split_str_parts_by(response, regex_delimiters, regex_mode=True)
    weights = []
    agent_content = []
    for c in agent_parts:
        if c['key'] in response_loss_scale_map:
            loss_scale = response_loss_scale_map[c['key']]
            assert len(loss_scale) in {1, 2}, f'loss_scale: {loss_scale}'
            if len(loss_scale) == 1:
                weights += loss_scale
                agent_content.append(c['content'])
            else:
                weights += loss_scale
                agent_content += [c['key'], c['content']]
        else:
            weights.append(1.)
            agent_content.append(c['content'])
    return agent_content, weights
