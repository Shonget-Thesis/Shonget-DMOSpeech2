import json

transcript_path = r'C:\Users\Mark Limpahan\.gemini\antigravity-ide\brain\d80be0eb-766b-42b8-9c72-0231a1a61655\.system_generated\logs\transcript_full.jsonl'
output_path = r'c:\Users\Mark Limpahan\Downloads\Thesis Shits\Shonget-DMOSpeech2\CONVERSATION_HISTORY.md'

messages = []
with open(transcript_path, 'r', encoding='utf-8') as f:
    for line in f:
        line_s = line.strip()
        if not line_s:
            continue
        try:
            item = json.loads(line_s)
        except Exception:
            continue
        
        step_type = item.get('type')
        content = item.get('content', '')
        source = item.get('source', '')
        
        if step_type == 'USER_INPUT' or source == 'USER_EXPLICIT':
            text = content
            if '<USER_REQUEST>' in text:
                parts = text.split('<USER_REQUEST>')
                if len(parts) > 1:
                    text = parts[1].split('</USER_REQUEST>')[0].strip()
            if text.startswith('<SYSTEM_MESSAGE>') or '[Notice]' in text:
                continue
            if text:
                messages.append(('User', text.strip()))
        elif step_type == 'PLANNER_RESPONSE' or source == 'MODEL':
            if not content or not content.strip():
                continue
            lines = [l.strip() for l in content.strip().splitlines() if l.strip()]
            if not lines:
                continue
            first_line = lines[0]
            # Ignore raw tool output artifacts echoed as model responses
            if first_line.startswith('Created At:') or first_line.startswith('{"name":') or first_line.startswith('File Path:'):
                continue
            if 'I am extracting the full conversation history' in content:
                continue
            messages.append(('Assistant', content.strip()))

# Combine consecutive turns of the same role
merged_messages = []
for role, text in messages:
    if merged_messages and merged_messages[-1][0] == role:
        merged_messages[-1] = (role, merged_messages[-1][1] + '\n\n' + text)
    else:
        merged_messages.append([role, text])

with open(output_path, 'w', encoding='utf-8') as out:
    out.write('# Complete Dialogue History: TAGS-Flow & DMOSpeech 2\n\n')
    out.write('> **Repository:** `DMOSpeech 2 / Shonget-DMOSpeech2`\n')
    out.write('> **Topic:** Inference Architecture, TAGS-Flow Mathematical Foundation, Cadence Scaling, and Implementation Details\n\n---\n\n')
    for role, text in merged_messages:
        out.write(f'## 👤 {role}\n\n' if role == 'User' else f'## 🤖 {role}\n\n')
        out.write(f'{text}\n\n---\n\n')

print(f'Cleanly saved {len(merged_messages)} dialogue turns to {output_path}')
