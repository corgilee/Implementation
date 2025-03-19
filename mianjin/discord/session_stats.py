import json
import pprint
from datetime import datetime, timedelta
from collections import defaultdict, Counter


# not necessary
def parse_timestamp(timestamp):
    """Convert timestamp string to datetime object."""
    return datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")

def process_log_file(file_path):
    # first to Dictionary to store sessions per user
    user_sessions = defaultdict(list) 

    # Read the JSON log file line by line
    with open(file_path, "r") as file:
        #events = [json.loads(line.strip()) for line in file]
        events = [json.loads(line) for line in file]

    # events are list of dictionary
    # print(f'type of events: {type(events)}')
    # print(f'type of events: {type(events[0])}')
    
    # Sort events by timestamp (important for session grouping)
    #events.sort(key=lambda x: (x["user_id"], parse_timestamp(x["timestamp"])))
    events.sort(key=lambda x: (x["user_id"], x["timestamp"]))

    #pprint.pprint(events)

    # Process each event
    for event in events:
        user_id = event["user_id"]
        timestamp = parse_timestamp(event["timestamp"])
        recipient = event["recipient"]

        # If current user has no sessions, create the first one
        if user_id not in  user_sessions:
            user_sessions[user_id].append([(timestamp, recipient)])
            # skip the following steps
            continue

        # Check the last session for this user
        last_session = user_sessions[user_id][-1]
        session_start_time = last_session[0][0] # session_time of the first record

        # If the new event falls within the 30-minute session window, add to the current session
        if timestamp <= session_start_time + timedelta(minutes=30):
            last_session.append((timestamp, recipient))
        else:
            # Otherwise, start a new session
            user_sessions[user_id].append([(timestamp, recipient)])

    pprint.pprint(user_sessions)
    # Compute session statistics, use a list
    session_results = []
    for user_id, sessions in user_sessions.items():
        for session in sessions:
            event_count = len(session)
            recipient_counts = Counter(recipient for _, recipient in session)
            most_frequent_recipient = recipient_counts.most_common(1)[0][0] 
            
            session_results.append({
                "user_id": user_id,
                "session_start": session[0][0].isoformat(),
                "event_count": event_count,
                "most_frequent_recipient": most_frequent_recipient
            })

    return session_results

# Example usage:
file_path = "test_json.json"  # Replace with your actual file path
session_data = process_log_file(file_path)

import pandas as pd

df = pd.DataFrame(session_data)
print(df)

