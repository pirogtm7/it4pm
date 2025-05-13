# 6 base prompts

GENERAL_INTRO = """You are an advanced AI system specialized in solving process mining tasks. 
Below is a description of a specific task along with process-related data.
Please produce the correct answer exactly as requested."""

TASK_PROMPTS_VARIANTS = {
    "trace_anomaly": [
        {
            "template": """Given a set of activities that constitute an organizational process and a sequence of activities, determine whether the sequence is a valid execution of the process.
The activities in the sequence must be performed in the correct order for the execution to be valid.
Provide either True or False as the answer and nothing else.

Set of process activities: {activities}

Trace: {trace}

Valid: """,
            "answer": "{gold_answer}"
        },
        {
            "template": """You have a process with a given set of activities, and a proposed sequence of how they are performed.
Check if the sequence respects the correct ordering of these process activities.
Answer with True if correct, False otherwise.

Set of process activities: {activities}

Trace: {trace}

Valid: """,
            "answer": "{gold_answer}"
        },
        {
            "template": """A process is defined by these activities. Does the following sequence respect the valid ordering?
Return True or False.

Activities: {activities}

Sequence: {trace}

Valid: """,
            "answer": "{gold_answer}"
        },
        {
            "template": """Is this sequence of activities consistent with the required process order?
Answer True or False.

Process Activities: {activities}

Observed Sequence: {trace}

Valid: """,
            "answer": "{gold_answer}"
        },
        {
            "template": """Does the given trace properly follow the rules of the process?
Only respond with True or False.

Activities: {activities}

Trace: {trace}

Valid: """,
            "answer": "{gold_answer}"
        },
        {
            "template": """Determine if the sequence is valid for the given set of activities.
Output True or False.

Process Activities: {activities}

Sequence: {trace}

Valid: """,
            "answer": "{gold_answer}"
        }
    ],
    "activity_anomaly": [
        {
            "template": """You are given a set of activities that constitute an organizational process and two activities performed in a single process execution. Determine whether it is valid for the first activity to occur before the second.
Provide either True or False as the answer and nothing else.

Set of process activities: {activities}

1. Activity: {act1}
2. Activity: {act2}

Valid: """,
            "answer": "{gold_answer}"
        },
        {
            "template": """Check if having '{act1}' before '{act2}' is allowed in the process described by these activities.
Return True or False only.

Activities: {activities}

Valid: """,
            "answer": "{gold_answer}"
        },
        {
            "template": """We want to see if the ordering (first activity, then second) is valid under this process definition.
Respond with True or False.

All Activities: {activities}

(1) {act1}
(2) {act2}

Valid: """,
            "answer": "{gold_answer}"
        },
        {
            "template": """Given two activities, is it permissible for '{act1}' to occur before '{act2}' in this process?
Answer True or False.

Process Activities: {activities}

Valid: """,
            "answer": "{gold_answer}"
        },
        {
            "template": """Is the order (first: '{act1}', second: '{act2}') acceptable according to the given set of activities?
Answer with True or False.

Activities set: {activities}

Valid: """,
            "answer": "{gold_answer}"
        },
        {
            "template": """Determine whether '{act1}' preceding '{act2}' matches the process constraints.
Answer True or False.

Defined process activities: {activities}

Valid: """,
            "answer": "{gold_answer}"
        }
    ],
    "next_activity": [
        {
            "template": """You are given a list of activities that constitute an organizational process and a sequence of activities that have been performed in the given order.
Which activity from the list should be performed next in the sequence?
The answer should be one activity from the list and nothing else.

Set of process activities: {activities}

Sequence of activities: {prefix}

Answer: """,
            "answer": "{gold_answer}"
        },
        {
            "template": """We have a partial execution trace of activities. From the provided set, which single activity comes next?
Return only the activity name.

Activities: {activities}

So far executed: {prefix}

Answer: """,
            "answer": "{gold_answer}"
        },
        {
            "template": """Given these process activities and the current sequence, choose the next activity.
Output exactly one activity from the list.

All Activities: {activities}

Current sequence: {prefix}

Answer: """,
            "answer": "{gold_answer}"
        },
        {
            "template": """A set of activities defines a process. The following sequence has already occurred.
Which activity logically follows next?
Return just the name.

Possible activities: {activities}

Executed so far: {prefix}

Answer: """,
            "answer": "{gold_answer}"
        },
        {
            "template": """Identify the next valid activity in the sequence from this list.
Only give the single activity as the answer.

Process Activities: {activities}

Performed sequence: {prefix}

Answer: """,
            "answer": "{gold_answer}"
        },
        {
            "template": """Select the next activity after the given partial trace. Only one from the list is correct.

Activities: {activities}

Current partial trace: {prefix}

Answer: """,
            "answer": "{gold_answer}"
        }
    ],
    "dfg": [
        {
            "template": """Given a list of activities that constitute an organizational process, determine all pairs of activities that can reasonably follow each other directly in an execution of this process.

Output Format (STRICT):
- The output must be a list of tuples: [(Activity A, Activity B)]
- Do NOT use quotes (' or ") around activity names.
- End the output with [END]

Correct Example: [(Activity A, Activity B)] [END]
Incorrect Examples (DO NOT USE): '(Activity A, Activity B)', ['Activity A', 'Activity B']

Set of process activities: {activities}

Pairs of activities: """,
            "answer": "{gold_answer}"
        }
    ],
    "process_tree": [
        {
            "template": """Given a list of activities that constitute an organizational process, determine the process tree of the process. 
A process tree is a hierarchical process model.
The following operators are defined for process trees:
-> ( A, B ) tells that process tree A should be executed before process tree B.
X ( A, B ) tells that there is an exclusive choice between executing process tree A and process tree B.
+ ( A, B ) tells that process tree A and process tree B are executed in true concurrency.
* ( A, B ) tells that process tree A is executed, then either you exit the loop, or you execute B and then A again (this can happen several times until the loop is exited).
Leaves of the process tree are activities or silent steps (indicated by tau).
An example process tree follows:
+ ( 'a', -> ( 'b', 'c', 'd' ) )
It defines that you should execute b before executing c and c before d. In concurrency to this, you can execute a.
Provide the process tree using only the allowed operators and these activities, followed by [END].
Also make sure each activity is used exactly once and that each subtree has exactly one root node.

Set of process activities: {activities}

Process tree: """,
            "answer": "{gold_answer}"
        }
    ],
}

# here we substitute two prompts that ask for next/previous activity prediction with additional variants of a different prompt
TASK_PROMPTS_INVERTED_NEGATIVE = {
    "activity_anomaly": [
        {
            "template": """You are given a set of activities that constitute an organizational process and one activity performed during the process. Choose another activity from the set that clearly cannot eventually follow the given activity.

Set of process activities: {activities}

Given activity: {act1}

Invalid next activity: """,
            "answer": "{act2}"
        },
        {
            "template": """Based on this set of process activities and one activity performed during the process, choose another activity from the set that cannot eventually precede the given activity.

Set of process activities: {activities}

Given activity: {act2}

Invalid previous activity: """,
            "answer": "{act1}"
        }
    ],
    "trace_anomaly": [
        {
            "template": """We need one example of a sequence that is invalid for the given set of activities. 
Provide only this anomalous sequence.

Activities: {activities}

Anomalous trace: """,
            "answer": "{trace}"
        },
        {
            "template": """Propose a short sequence that clearly violates the process constraints. 
Return only that invalid trace.

Set of activities: {activities}

Anomalous trace: """,
            "answer": "{trace}"
        }
    ],
}

# here we substitute two prompts that ask for next/previous activity prediction with additional variants of a different prompt
TASK_PROMPTS_INVERTED_POSITIVE = {
    "activity_anomaly": [
        {
            "template": """You are given a set of activities that constitute an organizational process and one activity performed during the process. Choose another activity from the set that can follow the given activity.

Set of process activities: {activities}

Given activity: {act1}

Next activity: """,
            "answer": "{act2}"
        },
        {
            "template": """Based on this set of process activities and one activity performed during the process, choose another activity from the set that can precede the given activity.

Set of process activities: {activities}

Given activity: {act2}

Previous activity: """,
            "answer": "{act1}"
        }
    ],
    "trace_anomaly": [
        {
            "template": """We need one example of a sequence that is valid for the given set of activities. 
Provide only this valid sequence.

Activities: {activities}

Valid trace: """,
            "answer": "{trace}"
        },
        {
            "template": """Propose a sequence that follows the process constraints. 
Return only that valid trace.

Set of activities: {activities}

Valid trace: """,
            "answer": "{trace}"
        }
    ],
    "next_activity": [
        {
            "template": """Given a specific activity that has just occurred, propose a short partial trace that might logically come before it in the process. 
Return only that preceding trace.

Activities: {activities}

Last activity: {next_act}

Preceding trace: """,
            "answer": "{prefix}"
        },
        {
            "template": """We have a single activity that just happened. 
Please suggest a short sequence of activities that could have led up to it.
Return only that partial trace.

Process Activities: {activities}

Observed activity: {next_act}

Prior steps: """,
            "answer": "{prefix}"
        },
        {
            "template": """Given a specific activity that has just occurred, determine a short sequence of activities that might have preceded it.  
Return only that partial trace.  

Available activities: {activities}  

Current activity: {next_act}  

Earlier activities: """,
            "answer": "{prefix}"
        },
        {
            "template": """An activity has just taken place in a process.  
Suggest a short sequence of prior activities that could have led up to it.  
Return only that preceding trace.  

Set of activities: {activities}  

Current activity: {next_act}  

Previous trace: """,
            "answer": "{prefix}"
        }
    ],
}
