#!/usr/bin/env python3
"""
Quick fix for web interface - remove CPU warning dialog, use unified endpoint
"""

import re

html_file = "/root/everplan.ai-vmm/web/static/index.html"

with open(html_file, 'r') as f:
    content = f.read()

# Find and replace the entire if/else block for CPU vs GPU
old_pattern = r'''                try:\s+let response;\s+
                    
                    if \(useGPU\) \{.*?// Using IPEX-LLM container model\s+\}\);\s+\}\) else \{.*?// User wants to try CPU anyway.*?\}\s+\}'''

new_code = '''                try {
                    // Use unified chat completions endpoint with device parameter
                    const response = await fetch(`${API_BASE}/api/chat/completions`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            messages: [
                                { role: 'user', content: prompt }
                            ],
                            max_tokens: maxTokens,
                            temperature: temperature,
                            model: 'tinyllama',
                            device: useGPU ? 'gpu' : 'cpu'
                        })
                    });'''

# Use re.DOTALL to match across lines
content_new = re.sub(old_pattern, new_code, content, flags=re.DOTALL)

if content_new != content:
    with open(html_file, 'w') as f:
        f.write(content_new)
    print("✅ Updated web interface to use unified endpoint")
else:
    print("❌ Pattern not found, doing manual update...")
    # Manual approach: find the line and do targeted replacement
    lines = content.split('\n')
    in_block = False
    output_lines = []
    skip_until_closing = 0
    
    for i, line in enumerate(lines):
        if 'let response;' in line and 'try {' in lines[i-1]:
            # Start of block - replace entire section
            output_lines.append('                try {')
            output_lines.append('                    // Use unified chat completions endpoint with device parameter')
            output_lines.append('                    const response = await fetch(`${API_BASE}/api/chat/completions`, {')
            output_lines.append('                        method: \'POST\',')
            output_lines.append('                        headers: {')
            output_lines.append('                            \'Content-Type\': \'application/json\'')
            output_lines.append('                        },')
            output_lines.append('                        body: JSON.stringify({')
            output_lines.append('                            messages: [')
            output_lines.append('                                { role: \'user\', content: prompt }')
            output_lines.append('                            ],')
            output_lines.append('                            max_tokens: maxTokens,')
            output_lines.append('                            temperature: temperature,')
            output_lines.append('                            model: \'tinyllama\',')
            output_lines.append('                            device: useGPU ? \'gpu\' : \'cpu\'')
            output_lines.append('                        })')
            output_lines.append('                    });')
            in_block = True
            skip_until_closing = 0
        elif in_block:
            # Count braces to find end of block
            skip_until_closing += line.count('{') - line.count('}')
            if skip_until_closing <= 0 and '});' in line and 'fetch' not in line:
                in_block = False
                # Don't add this line, we already added the closing
                continue
        elif not in_block:
            output_lines.append(line)
    
    with open(html_file, 'w') as f:
        f.write('\n'.join(output_lines))
    print("✅ Manual update completed")

print("Done!")
