import ast

GENERAL_INTRO = """You are an advanced AI system specialized in solving process mining tasks. 
Below is a description of a specific task along with process-related data.
Please produce the correct answer exactly as requested."""

def build_few_shot_prompt_for_trace_anomaly(few_shot_examples, current_example):
    """
    Builds a prompt that includes:
      - The standard instruction text
      - If few_shot_examples is not empty, a section labeled "Examples:"
        with each shot's Set of process activities + Trace + Valid: {gold_answer}
      - Finally, the query for the current example (without the answer).
    """
    main_instruction = """Given a set of activities that constitute an organizational process and a sequence of activities, determine whether the sequence is a valid execution of the process.
The activities in the sequence must be performed in the correct order for the execution to be valid.
Provide either True or False as the answer and nothing else.\n
"""
    prompt_parts = [GENERAL_INTRO, "\n\n", main_instruction]

    if len(few_shot_examples) > 0:
        prompt_parts.append("Examples:\n")

        for shot in few_shot_examples:
            shot_activities = str(set(shot["unique_activities"]))
            shot_trace = str(shot["trace"])
            shot_label = str(shot["ds_labels"])
            prompt_parts.append(
                f"Set of process activities: {shot_activities}\n"
                f"Trace: {shot_trace}\n"
                f"Valid: {shot_label}\n\n"
            )

    # The question we want the model to answer
    current_activities = str(set(current_example["unique_activities"]))
    current_trace = str(current_example["trace"])
    prompt_parts.append(
        f"Set of process activities: {current_activities}\n"
        f"Trace: {current_trace}\n"
        f"Valid: "
    )

    return "".join(prompt_parts)


def build_few_shot_prompt_for_activity_anomaly(few_shot_examples, current_example):
    """
    Builds a multi-shot prompt for activity-level anomaly detection.
    Each example is shown as:
        Set of process activities: {...}
        1. Activity: {act1}
        2. Activity: {act2}
        Valid: {gold_answer}
    Then the final example is asked without the answer.
    """
    main_instruction = """You are given a set of activities that constitute an organizational process and two activities performed in a single process execution. Determine whether it is valid for the first activity to occur before the second.
Provide either True or False as the answer and nothing else.\n
"""
    prompt_parts = [GENERAL_INTRO, "\n\n", main_instruction]

    if len(few_shot_examples) > 0:
        prompt_parts.append("Examples:\n")

        for shot in few_shot_examples:
            shot_activities = str(set(shot["unique_activities"]))
            # 'eventually_follows' should be a list/tuple of two items
            shot_act1, shot_act2 = shot["eventually_follows"]
            shot_label = str(shot["ds_labels"])

            prompt_parts.append(
                f"Set of process activities: {shot_activities}\n"
                f"1. Activity: {shot_act1}\n"
                f"2. Activity: {shot_act2}\n"
                f"Valid: {shot_label}\n\n"
            )

    curr_activities = str(set(current_example["unique_activities"]))
    curr_act1, curr_act2 = current_example["eventually_follows"]
    prompt_parts.append(
        f"Set of process activities: {curr_activities}\n"
        f"1. Activity: {curr_act1}\n"
        f"2. Activity: {curr_act2}\n"
        f"Valid: "
    )

    return "".join(prompt_parts)


def build_few_shot_prompt_for_next_activity(few_shot_examples, current_example):
    """
    Builds a multi-shot prompt for next_activity.
    Each shot is shown as:
        Set of process activities: {...}
        Sequence of activities: {prefix}
        Answer: {gold_answer}
    Then the final query is asked without the answer.
    """
    main_instruction = """You are given a list of activities that constitute an organizational process and a sequence of activities that have been performed in the given order.
Which activity from the list should be performed next in the sequence?
The answer should be one activity from the list and nothing else.\n
"""

    prompt_parts = [GENERAL_INTRO, "\n\n", main_instruction]

    if len(few_shot_examples) > 0:
        prompt_parts.append("Examples:\n")

        for shot in few_shot_examples:
            shot_activities = str(set(shot["unique_activities"]))
            shot_prefix = str(shot["prefix"])
            shot_answer = str(shot["next"])

            prompt_parts.append(
                f"Set of process activities: {shot_activities}\n"
                f"Sequence of activities: {shot_prefix}\n"
                f"Answer: {shot_answer}\n\n"
            )

    curr_activities = str(set(current_example["unique_activities"]))
    curr_prefix = str(current_example["prefix"])
    prompt_parts.append(
        f"Set of process activities: {curr_activities}\n"
        f"Sequence of activities: {curr_prefix}\n"
        f"Answer: "
    )

    return "".join(prompt_parts)


def list_of_tuples_to_arrow_format(dfg_list):
    pairs_str = [f"{act1} -> {act2}" for (act1, act2) in dfg_list]
    return "\n".join(pairs_str) + "\n[END]"


def build_few_shot_prompt_for_dfg(few_shot_examples, current_example):
    """
    Builds a multi-shot prompt for dfg generation.
    Each shot is shown as:
        Set of process activities: {...}
        Pairs of activities: {dfg in arror representation}
    Then the final query is asked without the answer.
    """
    main_instruction = """Given a list of activities that constitute an organizational process, determine all pairs of activities that can reasonably follow each other directly in an execution of this process.
The output must be one pair per line, in the format: Activity A -> Activity B. End the output with [END] on a new line. Do not include any extra elements like list numbers, bullet points etc.\n
"""

    prompt_parts = [GENERAL_INTRO, "\n\n", main_instruction]

    # Few-shot examples
    if len(few_shot_examples) > 0:
        prompt_parts.append("Examples:\n")

        for shot in few_shot_examples:
            shot_activities = str(set(shot["unique_activities"]))
            dfg_list = ast.literal_eval(shot["dfg"])
            dfg_arrow_format = list_of_tuples_to_arrow_format(dfg_list)

            prompt_parts.append(
                f"Set of process activities: {shot_activities}\n"
                f"Pairs of activities:\n{dfg_arrow_format}\n\n"
            )

    current_activities = str(set(current_example["unique_activities"]))
    prompt_parts.append(
        f"Set of process activities: {current_activities}\n"
        f"Pairs of activities: "
    )

    return "".join(prompt_parts)


def build_few_shot_prompt_for_process_tree(few_shot_examples, current_example):
    """
    Builds a multi-shot prompt for process_tree generation.
    Each shot is shown as:
        Set of process activities: {...}
        Process tree: {pt}
    Then the final query is asked without the answer.
    """
    main_instruction = """Given a list of activities that constitute an organizational process, determine the process tree of the process. 
A process tree is a hierarchical process model.
The following operators are defined for process trees:
->( A, B ) tells that process tree A should be executed before process tree B.
X( A, B ) tells that there is an exclusive choice between executing process tree A and process tree B.
+( A, B ) tells that process tree A and process tree B are executed in true concurrency.
*( A, B ) tells that process tree A is executed, then either you exit the loop, or you execute B and then A again (this can happen several times until the loop is exited).
Leaves of the process tree are activities. Use commas to separate arguments inside operators.
An example process tree follows:
+( 'a', ->( 'b', 'c', 'd' ) )
It defines that you should execute b before executing c and c before d. In concurrency to this, you can execute a.
Provide the process tree using only the allowed operators and these activities, followed by [END].
Also make sure each activity is used exactly once and that each subtree has exactly one root node.\n
"""

    prompt_parts = [GENERAL_INTRO, "\n\n", main_instruction]

    if len(few_shot_examples) > 0:
        prompt_parts.append("Examples:\n")

        for shot in few_shot_examples:
            shot_activities = str(set(shot["unique_activities"]))
            shot_pt = str(shot["pt"])  # The correct process tree string
            prompt_parts.append(
                f"Set of process activities: {shot_activities}\n"
                f"Process tree: {shot_pt} [END]\n\n"
            )

    curr_activities = str(set(current_example["unique_activities"]))
    prompt_parts.append(
        f"Set of process activities: {curr_activities}\n"
        f"Process tree: "
    )

    return "".join(prompt_parts)
