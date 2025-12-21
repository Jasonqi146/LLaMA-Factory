from datasets import load_dataset
import json

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("snorkelai/tb-plus-plus", split="train")

data = []
filtered_count = 0

for item in ds:
    convo = item["conversations"]
    
    # Normalize messages to exact format: {"role": ..., "content": ...}
    messages = []
    for msg in convo:
        role = msg.get("role", "user")
        if role not in ("user", "assistant", "system"):
            role = "user"
        # Ensure key order is role, then content
        messages.append({"role": role, "content": msg.get("content", "")})
    
    # Extract system message if present
    system_msg = None
    if messages and messages[0]["role"] == "system":
        system_msg = messages[0]
        messages = messages[1:]
    
    # Fix: If conversation starts with assistant, prepend a user message
    # (The assistant's first message becomes context for user's task)
    if messages and messages[0]["role"] == "assistant":
        # Move assistant's initial response to be part of the flow
        # We need user -> assistant alternation, so skip this malformed start
        # or restructure: treat the first assistant message as context in system
        if system_msg:
            # Append the assistant's initial response to system context
            system_msg["content"] += "\n\n[Initial context]\n" + messages[0]["content"]
            messages = messages[1:]
        else:
            # Skip this conversation - no good way to fix without system
            filtered_count += 1
            continue
    
    # Ensure alternation: user, assistant, user, assistant...
    fixed_messages = []
    expected_role = "user"
    for msg in messages:
        if msg["role"] == expected_role:
            fixed_messages.append(msg)
            expected_role = "assistant" if expected_role == "user" else "user"
        elif msg["role"] == expected_role:
            fixed_messages.append(msg)
        # Skip messages that break alternation
    
    # Must have at least one user-assistant pair
    if len(fixed_messages) < 2:
        filtered_count += 1
        continue
    
    # Must end with assistant - truncate if ends with user
    if fixed_messages[-1]["role"] == "user":
        fixed_messages = fixed_messages[:-1]
    
    # Must have even count (user-assistant pairs)
    if len(fixed_messages) % 2 != 0:
        fixed_messages = fixed_messages[:-1]
    
    # Must still have content
    if len(fixed_messages) < 2:
        filtered_count += 1
        continue
    
    # Rebuild messages with system at front
    final_messages = []
    if system_msg:
        final_messages.append(system_msg)
    final_messages.extend(fixed_messages)
    
    # Final validation: no consecutive same roles
    has_consecutive = False
    for j in range(len(final_messages) - 1):
        curr = final_messages[j]["role"]
        next_r = final_messages[j + 1]["role"]
        # system -> user is OK, user -> assistant is OK, assistant -> user is OK
        if curr == next_r:
            has_consecutive = True
            break
        if curr == "system" and next_r != "user":
            has_consecutive = True
            break
    
    if has_consecutive:
        filtered_count += 1
        continue
    
    data.append({"messages": final_messages})

print(f"Kept samples: {len(data)}")
print(f"Filtered out samples: {filtered_count}")

with open("data/tb_plus_12_20_train.jsonl", "w") as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Output written to: data/tb_plus_12_20_train.jsonl")
