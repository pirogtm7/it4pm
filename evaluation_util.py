"""MIT License

Copyright (c) 2024 a-rebmann

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""


import string
import numpy as np
from pm4py.objects.process_tree.semantics import GenerationTree, ProcessTree, generate_log
import re
from pm4py.objects.process_tree.obj import Operator
from uuid import uuid4

import signal

def camel_to_white(label):
    label = CAMEL_PATTERN_1.sub(r"\1 \2", label)
    return CAMEL_PATTERN_2.sub(r"\1 \2", label)


NON_ALPHANUM = re.compile("[^a-zA-Z]")
CAMEL_PATTERN_1 = re.compile("(.)([A-Z][a-z]+)")
CAMEL_PATTERN_2 = re.compile("([a-z0-9])([A-Z])")


def extract_directly_follows_pairs(sequences):
    directly_follows_pairs = set()  # Use a set to avoid duplicate pairs

    # Iterate over each sequence
    for sequence in sequences:
        # Iterate over consecutive pairs of activities in the sequence
        for i in range(len(sequence) - 1):
            pair = (sequence[i], sequence[i+1])  # Current activity and the next one
            directly_follows_pairs.add(pair)

    return list(directly_follows_pairs)


def compute_footprint_matrix(sequences, activities):
    pairs = extract_directly_follows_pairs(sequences)
    return compute_footprint_matrix_pairs(pairs, activities)


def extract_activity_pairs(output_str, valid_activities):
    """
    Extracts correctly formatted activity pairs from the model output.
    - Searches for the longest activity name first to prevent substring mismatches.
    - Ensures commas within activity names do not interfere with tuple extraction.
    - Maintains correct order of extracted activities.
    """
    extracted_pairs = []
    working_str = output_str.strip("[] ")  # Remove surrounding brackets

    sorted_activities = sorted(valid_activities, key=len, reverse=True)

    while "(" in working_str and ")" in working_str:
        start = working_str.find("(")
        end = working_str.find(")", start)
        
        if start == -1 or end == -1:
            break  # No more valid pairs found

        pair_str = working_str[start + 1:end]  # Extract content inside the first `( ... )`
        working_str = working_str[end + 1:].strip()  # Remove the processed part

        left_activity, right_activity = None, None

        # Find the longest valid activity
        for activity in sorted_activities:
            if pair_str.startswith(activity):
                left_activity = activity
                remaining_str = pair_str[len(activity):].strip(", ")
                break
            elif pair_str.endswith(activity):
                right_activity = activity
                remaining_str = pair_str[:-len(activity)].strip(", ")
                break

        # Find the second valid activity in the remaining string
        if left_activity:
            for activity in sorted_activities:
                if activity in remaining_str:
                    right_activity = activity
                    break
        elif right_activity:
            for activity in sorted_activities:
                if activity in remaining_str:
                    left_activity = activity
                    break

        if left_activity and right_activity:
            extracted_pairs.append((left_activity, right_activity))
        else:
            print(f"Skipped invalid pair: ({pair_str}) - Not in valid activities")

    return extracted_pairs


def compute_footprint_matrix_pairs(pairs, activities):
    activities = sorted(act.strip() for act in activities)  # Sort to maintain order
    n = len(activities)
    
    # Step 2: Create an empty n x n matrix
    footprint_matrix = np.full((n, n), '#', dtype='<U2')  # Initialize with '#'
    
    # Map activities to indices
    activity_idx = {activity: idx for idx, activity in enumerate(activities)}
    
    # Step 3: Fill the matrix based on the pairs
    for a, b in pairs:
        try:
            i, j = activity_idx[a.strip()], activity_idx[b.strip()]
            footprint_matrix[i][j] = '→'  # A can follow B
        except KeyError:
            print(f"Activity not found in the list: skipping '{a}' or '{b}'")
    
    # Step 4: Identify concurrent and opposite flows
    for i in range(n):
        for j in range(n):
            if footprint_matrix[i][j] == '→' and footprint_matrix[j][i] == '→':
                footprint_matrix[i][j] = footprint_matrix[j][i] = '‖'
            elif footprint_matrix[i][j] == '→' and footprint_matrix[j][i] == '#':
                footprint_matrix[j][i] = '←'
    return footprint_matrix


def compute_footprint_fitness(matrix1, matrix2):
    # Ensure both matrices have the same shape
    if matrix1.shape != matrix2.shape:
        raise ValueError("Both matrices must have the same dimensions.")
    
    total_elements = matrix1.size  # Total number of elements in the matrix
    matching_elements = np.sum(matrix1 == matrix2)  # Count matching symbols
    
    # Compute fitness as the fraction of matching symbols
    fitness = matching_elements / total_elements
    
    return fitness


# Define a class to represent each node in the tree
class TreeNode:
    def __init__(self, name, node_type):
        self.id = uuid4()
        self.name = name
        self.node_type = node_type
        self.children: list[TreeNode] = []

    def __repr__(self):
        return f"{self.name} - {self.children}"

def parse_tree_str(tree_str):
    # ensure that "->" is recognized correctly as an operator
    match = re.match(r"([+\-*X>]+)\s*\((.*)\)", tree_str.strip())
    
    if not match:
        return TreeNode(name=tree_str.strip(), node_type="activity")

    root_name, children_str = match.groups()

    if root_name not in ["+", "->", "X", "*"]:
        return TreeNode(name=root_name, node_type="activity")

    root = TreeNode(name=root_name, node_type=root_name)

    children_list = []
    start, depth = 0, 0
    for i, c in enumerate(children_str):
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == "," and depth == 0:
            children_list.append(children_str[start:i].strip())
            start = i + 1

    children_list.append(children_str[start:].strip())
    for child in children_list:
        root.children.append(parse_tree_str(child))

    return root


def convert_to_pm4py(current: TreeNode, parent: ProcessTree) -> ProcessTree:
    # node
    if current.name == "+":
        operator = Operator.PARALLEL
        label = None
    elif current.name == "->":
        operator = Operator.SEQUENCE
        label = None
    elif current.name == "X":
        operator = Operator.XOR
        label = None
    elif current.name == "*":
        operator = Operator.LOOP
        label = None
    else:
        operator = None
        label = current.name
    tree = ProcessTree(operator=operator, label=label)
    tree.parent = parent
    for child in current.children:        
        tree.children.append(convert_to_pm4py(child, parent=tree))
    return tree

def rename_nodes(tree, letter_to_activity, activities):# rename the nodes to the original activity names
    for node in tree.children:
        if node.name in letter_to_activity and node.name not in activities:
            node.name = letter_to_activity[node.name]
        rename_nodes(node, letter_to_activity, activities)
    return tree
    


def parse_tree(tree_str: str, activities: set[str]) -> ProcessTree:
    # relace all the activities with single letters
    activity_to_letter, letter_to_activity = map_activities_to_letters(activities)

    # sort activities by length (longest first) to avoid substring issues
    sorted_activities = sorted(activity_to_letter.keys(), key=len, reverse=True)

    # use regex word boundaries (`\b`) to match whole words
    for activity in sorted_activities:
        letter = activity_to_letter[activity]
        tree_str = re.sub(rf"\b{re.escape(activity)}\b", letter, tree_str)

    # remove quotes
    tree_str = tree_str.replace('"', "")
    tree_str = tree_str.replace("'", "")
    parsed_tree = parse_tree_str(tree_str)    
    # rename the nodes to the original activity names
    parsed_tree = rename_nodes(parsed_tree, letter_to_activity, activities)
    # convert the parsed tree and return a ProcessTree object 
    return convert_to_pm4py(parsed_tree, ProcessTree(operator=None, label="tree"))


def handler(signum, frame):
    raise TimeoutError("Timed out!")

def generate_traces_from_tree(tree_str, activities):
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(60)  # Set a timeout

    try:
        # generate ProcessTree from string
        tree = parse_tree(tree_str, activities)
        gen_tree = GenerationTree(tree)
        traces = generate_log(gen_tree, no_traces=100)
        string_traces = [[event["concept:name"] for event in trace] for trace in traces]
        return string_traces
    except TimeoutError:
        print("Timed out via signal!")
    finally:
        signal.alarm(0)  # Cancel alarm


def map_activities_to_letters(activities):
    # List of single letters A-Z
    letters = list(string.ascii_uppercase) + list(string.ascii_lowercase)
    # remove the 'x' letter as it is used for the xor operator
    letters.remove('x')
    letters.remove('a')
    letters.remove('X')

    # Extend with combinations if there are more than 50 activities
    if len(activities) > len(letters):
        combinations = [
            "a".join(pair)
            for pair in zip(
                letters * len(letters),
                letters * (len(activities) // len(letters) + 1),
            )
        ]
        letters.extend(combinations)

    # Create a mapping dictionary
    activity_to_letter = {
        activity: letters[i] for i, activity in enumerate(activities)
    }

    # and the reverse mapping
    letter_to_activity = {v: k for k, v in activity_to_letter.items()}

    return activity_to_letter, letter_to_activity