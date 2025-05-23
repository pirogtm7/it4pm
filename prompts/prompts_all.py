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
The output must be a list [] of pairs in the format ('Activity A', 'Activity B'), ending with [END].

Example: [('Activity A', 'Activity B')] [END]

Use only activities from the given list.

Set of process activities: {activities}

Pairs of activities: """,
            "answer": "{gold_answer}"
        },
        {
            "template": """Based on these activities, enumerate each direct follow pair (A -> B) that is valid in the process.
List them in the format [('A', 'B')] and end with [END].

Activities: {activities}

Pairs of activities: """,
            "answer": "{gold_answer}"
        },
        {
            "template": """Find every immediate successor relationship among the given activities.
Provide the results as a list of pairs [('Activity X', 'Activity Y')] and conclude with [END].

Activity set: {activities}

Pairs of activities: """,
            "answer": "{gold_answer}"
        },
        {
            "template": """Which pairs of activities can occur consecutively in this process?
Return them as a list [('Activity A', 'Activity B')], then write [END].

Available activities: {activities}

Pairs of activities: """,
            "answer": "{gold_answer}"
        },
        {
            "template": """List all directly-follows connections (A, B) using the activities below.
Format them as [('A', 'B')] and append [END].

Process activities: {activities}

Pairs of activities: """,
            "answer": "{gold_answer}"
        },
        {
            "template": """From this set of activities, figure out all direct follow pairs.
Provide them as [('Activity X', 'Activity Y')], then conclude with [END].

Defined activities: {activities}

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
        },
        {
            "template": """Construct a process tree to represent a process model using these operators:
-> ( A, B ) for sequence,
X ( A, B ) for exclusive choice,
+ ( A, B ) for concurrency,
* ( A, B ) for loops.
Include all given activities exactly once, and end with [END]. If there is a silent step, use tau as a leaf.

Activities: {activities}

Process tree: """,
            "answer": "{gold_answer}"
        },
        {
            "template": """From this set of activities, create a valid process tree that defines their structure. 
Use ->, X, +, or * and conclude with [END].

-> ( A, B ) tells that process tree A should be executed before process tree B.
X ( A, B ) tells that there is an exclusive choice between executing process tree A and process tree B.
+ ( A, B ) tells that process tree A and process tree B are executed in true concurrency.
* ( A, B ) tells that process tree A is executed, then either you exit the loop, or you execute B and then A again (this can happen several times until the loop is exited).
Leaves of the process tree are activities or silent steps (indicated by tau).

Process activities: {activities}

Process tree: """,
            "answer": "{gold_answer}"
        },
        {
            "template": """Build a single-root process tree with the operators ->, X, +, and * from the following activities.
For sequence: -> ( A, B ), for exclusive choice: X ( A, B ), for concurrency: + ( A, B ), for loops: * ( A, B ).
The leaves are activities or tau, and each subtree must be properly parenthesized.
End the final answer with [END].

Available activities: {activities}

Process tree: """,
            "answer": "{gold_answer}"
        },
        {
            "template": """We want a hierarchical process tree with these activities as leaves. 
Use ->, X, +, * operators. Each activity must appear once. End with [END].
An example process tree follows:
+ ( 'a', -> ( 'b', 'c', 'd' ) )
It defines that you should execute b before executing c and c before d. In concurrency to this, you can execute a.
Additionally, you can use * ( x, y ) for loops. If a silent step is required, use tau.

Set of process activities: {activities}

Process tree: """,
            "answer": "{gold_answer}"
        },
        {
            "template": """Provide a valid process tree covering all listed activities exactly once, using ->, X, +, and *.
Finish with [END].

-> ( A, B ) tells that process tree A should be executed before process tree B.
X ( A, B ) tells that there is an exclusive choice between executing process tree A and process tree B.
+ ( A, B ) tells that process tree A and process tree B are executed in true concurrency.
* ( A, B ) tells that process tree A is executed, then either you exit the loop, or you execute B and then A again (this can happen several times until the loop is exited).
Leaves of the process tree are activities or silent steps (indicated by tau).

All activities: {activities}

Process tree: """,
            "answer": "{gold_answer}"
        }
    ],
}

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
        },
        {
            "template": """From this set of activities, produce a single pair that definitely cannot be performed in a single process execution. 
Return only that anomalous pair, nothing else.

Activities: {activities}

Anomalous pair: """,
            "answer": "{eventually_follows}"
        },
        {
            "template": """Identify a pair of activities from this set that cannot occur in the same process execution, either directly or in sequence.
Provide only that anomalous pair and nothing else.

Set of activities: {activities}

Anomalous pair: """,
            "answer": "{eventually_follows}"
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
            "template": """Propose a sequence that clearly violates the process constraints. 
Return only that invalid trace.

Set of activities: {activities}

Anomalous trace: """,
            "answer": "{trace}"
        },
        {
            "template": """Given the following process activities: {activities}, create a trace that would not be accepted by the process.

Only provide the invalid sequence.

Anomalous trace: """,
            "answer": "{trace}"
        },
        {
            "template": """Use the given set of activities to generate a sequence that breaks the execution rules of the process.

Return just the incorrect trace.

Activities: {activities}

Invalid trace: """,
            "answer": "{trace}"
        }
    ],
    "next_activity": [
        {
            "template": """An ongoing process execution is missing an important activity that should have occurred somewhere earlier in the sequence.  

Identify the missing activity that should have been included before reaching the last activity, at any point in the sequence.

Here is the observed sequence: {reduced_prefix}.

Set of activitites: {activities}.

Missing activity: """,
            "answer": "{act_from_reduced_prefix}"
        },
        {
            "template": """A process consists of the following possible activities: {activities}.

Identify an activity that does not fit the expected order in the trace.

Here is an ongoing execution trace: {extended_prefix}.

Activity: """,
            "answer": "{act_from_extended_prefix}"
        },
        {
            "template": """The following sequence is part of an ongoing process, but one crucial activity is missing.

Use the list of valid process activities to identify which activity should have occurred earlier.

Sequence: {reduced_prefix}

Valid activities: {activities}

Missing activity: """,
            "answer": "{act_from_reduced_prefix}"
        },
        {
            "template": """An incorrect activity appears in the current execution trace, violating expected process behavior.

Based on the activity list: {activities}

Here is the problematic trace: {extended_prefix}

Which activity should be flagged as incorrect?

Activity: """,
            "answer": "{act_from_extended_prefix}"
        }
    ],
    "dfg": [
        {
            "template": """A process consists of the following activities: {activities}.
Below is a proposed directly-follows graph: {extended_dfg}.

Identify a transition that does not belong in the process. Answer with a pair of activities and nothing else.

Transition: """,
            "answer": "{pair_from_extended_dfg}"
        },
        {
            "template": """The following process contains a set of activities: {activities}.
Below is a discovered directly-follows graph (DFG), which shows possible activity transitions:
Discovered DFG: {reduced_dfg}

However, one possible transition is missing from this graph.
Identify the missing activity pair that should be included to accurately represent the process.

Missing activity pair: """,
            "answer": "{pair_from_reduced_dfg}"
        },
        {
            "template": """From the following process model activities: {activities},
a directly-follows graph was proposed: {extended_dfg}.

Determine which transition in this graph is invalid and should not be part of the model.

Transition: """,
            "answer": "{pair_from_extended_dfg}"
        },
        {
            "template": """Given these activities: {activities}, a directly-follows graph (DFG) has been discovered with the following transitions:
{reduced_dfg}

One transition that should exist is missing from this DFG.
State the missing activity pair.

Missing activity pair: """,
            "answer": "{pair_from_reduced_dfg}"
        }
    ]
}


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
        },
        {
            "template": """We want an example of a pair that can be performed in a single process execution in this process. 
Pick one pair of activities that does not violate the rules. 
Output only that pair.

Process Activities: {activities}

Valid pair: """,
            "answer": "{eventually_follows}"
        },
        {
            "template": """Choose a pair of activities from this set that can be performed within the same process execution without violating any constraints.
Provide only the valid pair as your answer.

Set of activities: {activities}

Valid pair: """,
            "answer": "{eventually_follows}"
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
        },
        {
            "template": """Imagine a correct execution of the process using the provided activities.
List only a valid sequence of those activities.

Activities: {activities}

Valid trace: """,
            "answer": "{trace}"
        },
        {
            "template": """Using the given set of activities, provide a sequence that would be accepted as valid by the process.
Just return that valid trace.

Set of activities: {activities}

Valid trace: """,
            "answer": "{trace}"
        }
    ],
    "next_activity": [
        {
            "template": """Given a specific activity that has just occurred, propose a partial trace that might logically come before it in the process. 
Return only that preceding trace.

Activities: {activities}

Last activity: {next_act}

Preceding trace: """,
            "answer": "{prefix}"
        },
        {
            "template": """We have a single activity that just happened. 
Please suggest a sequence of activities that could have led up to it.
Return only that partial trace.

Process Activities: {activities}

Observed activity: {next_act}

Prior steps: """,
            "answer": "{prefix}"
        },
        {
            "template": """Given a specific activity that has just occurred, determine a sequence of activities that might have preceded it.  
Return only that partial trace.  

Available activities: {activities}  

Current activity: {next_act}  

Earlier activities: """,
            "answer": "{prefix}"
        },
        {
            "template": """An activity has just taken place in a process.  
Suggest a sequence of prior activities that could have led up to it.  
Return only that preceding trace.  

Set of activities: {activities}  

Current activity: {next_act}  

Previous trace: """,
            "answer": "{prefix}"
        }
    ],
}